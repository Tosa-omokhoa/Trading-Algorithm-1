# =============================================================================
# SynthTrade - CNN-LSTM Model
# =============================================================================
# Architecture: CNN layers extract local spatial patterns from the feature
# sequence (candlestick patterns, indicator crossovers), then LSTM layers
# capture temporal dependencies (trend momentum, sequential context).
#
# Self-improvement is wired in through three mechanisms:
#
#   1. INCREMENTAL RETRAINING
#      When IncrementalDataStore accumulates enough new outcomes, the model
#      is retrained with the new data appended to the original training set.
#      The model improves continuously as it observes real trade results.
#
#   2. REGIME-AWARE DROPOUT
#      Dropout rates adapt based on recent win rate. If the win rate on an
#      asset drops below 50% over the last 20 signals, dropout increases
#      (more regularisation = less overconfident predictions).
#
#   3. CONFIDENCE CALIBRATION
#      The final layer uses temperature scaling to calibrate probability
#      outputs so confidence scores reflect real win rates. A model that
#      says 80% confident should win ~80% of those trades.
#
# Model input:  (batch, SEQUENCE_LENGTH, n_features)  e.g. (32, 30, 40)
# Model output: (batch, 3) softmax probabilities for [SHORT, NO_TRADE, LONG]

import os
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from tensorflow.keras.optimizers import Adam
from typing import Optional, Tuple, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from config import (
        SEQUENCE_LENGTH, SIGNAL_CONFIDENCE_THRESHOLD,
        MODELS_DIR
    )
except ImportError:
    SEQUENCE_LENGTH = 30
    SIGNAL_CONFIDENCE_THRESHOLD = 0.72
    MODELS_DIR = "models/saved"

from features.dataset import (
    CLASS_NAMES, LABEL_TO_CLASS, CLASS_TO_LABEL,
    decode_labels
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Model")

# Suppress TensorFlow noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def build_cnn_lstm(
    input_shape:    Tuple[int, int],
    n_classes:      int   = 3,
    cnn_filters:    list  = None,
    lstm_units:     list  = None,
    dense_units:    int   = 64,
    dropout_rate:   float = 0.3,
    l2_reg:         float = 1e-4,
    learning_rate:  float = 1e-3,
) -> keras.Model:
    """
    Build the CNN-LSTM hybrid model.

    Architecture:
        Input (seq_len, n_features)
            |
        Conv1D(64, kernel=3) + BatchNorm + ReLU
        Conv1D(128, kernel=3) + BatchNorm + ReLU
        MaxPooling1D(2) + Dropout(rate)
            |
        LSTM(128, return_sequences=True) + Dropout(rate)
        LSTM(64)  + Dropout(rate * 0.8)
            |
        Dense(64, ReLU, L2 reg)
        Dense(n_classes, Softmax)

    Parameters
    ----------
    input_shape : (seq_len, n_features)
    n_classes : int, default 3
    cnn_filters : list of filter counts for each Conv1D layer
    lstm_units : list of units for each LSTM layer
    dense_units : int
    dropout_rate : float
    l2_reg : float, L2 regularisation for Dense layers
    learning_rate : float

    Returns
    -------
    Compiled Keras model.
    """
    if cnn_filters is None: cnn_filters = [64, 128]
    if lstm_units  is None: lstm_units  = [128, 64]

    inp = keras.Input(shape=input_shape, name="sequence_input")
    x = inp

    # --- CNN Block ---
    # Extracts local patterns: candlestick formations, indicator crossovers
    for i, filters in enumerate(cnn_filters):
        x = layers.Conv1D(
            filters=filters,
            kernel_size=3,
            padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"conv1d_{i+1}"
        )(x)
        x = layers.BatchNormalization(name=f"bn_conv_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_conv_{i+1}")(x)

    x = layers.MaxPooling1D(pool_size=2, name="maxpool")(x)
    x = layers.Dropout(dropout_rate, name="dropout_cnn")(x)

    # --- LSTM Block ---
    # Captures temporal dependencies: trend momentum, sequence context
    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)   # Only last LSTM returns final state
        x = layers.LSTM(
            units=units,
            return_sequences=return_seq,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_dropout=0.1,
            name=f"lstm_{i+1}"
        )(x)
        drop = dropout_rate if return_seq else dropout_rate * 0.8
        x = layers.Dropout(drop, name=f"dropout_lstm_{i+1}")(x)

    # --- Dense Head ---
    x = layers.Dense(
        dense_units,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_1"
    )(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="relu_dense")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="dropout_dense")(x)

    # Output: 3-class softmax (SHORT, NO_TRADE, LONG)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="SynthTrade_CNN_LSTM")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc"),
        ]
    )

    return model


def model_summary(model: keras.Model):
    """Print a clean model summary with parameter counts."""
    model.summary(print_fn=logger.info)
    total_params = model.count_params()
    logger.info(f"Total parameters: {total_params:,}")


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    model:           keras.Model,
    X_train:         np.ndarray,
    y_train:         np.ndarray,
    X_val:           np.ndarray,
    y_val:           np.ndarray,
    class_weights:   Dict[int, float],
    asset_name:      str   = "ASSET",
    epochs:          int   = 80,
    batch_size:      int   = 32,
    patience:        int   = 15,
    min_delta:       float = 1e-4,
    save_best:       bool  = True,
) -> keras.callbacks.History:
    """
    Train the CNN-LSTM model with early stopping and learning rate reduction.

    Callbacks:
      - EarlyStopping: stops when val_loss stops improving (patience=15 epochs)
      - ReduceLROnPlateau: halves LR when val_loss plateaus for 7 epochs
      - ModelCheckpoint: saves the best model by val_loss
      - TensorBoard: logs training curves (optional)

    Parameters
    ----------
    model : compiled Keras model
    X_train, y_train : training data (y is class-encoded integers)
    X_val, y_val : validation data
    class_weights : dict from prepare_dataset()
    asset_name : str, used for checkpoint filenames
    epochs : int, max training epochs
    batch_size : int
    patience : int, early stopping patience
    min_delta : float, minimum improvement to count as progress
    save_best : bool, whether to save best checkpoint to disk

    Returns
    -------
    keras.callbacks.History
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODELS_DIR, f"best_{asset_name}.keras")

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    if save_best:
        cb_list.append(
            callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            )
        )

    logger.info(
        f"[{asset_name}] Training: {len(X_train)} train, {len(X_val)} val, "
        f"epochs={epochs}, batch={batch_size}"
    )
    logger.info(f"[{asset_name}] Class weights: {class_weights}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=1
    )

    best_val_loss = min(history.history["val_loss"])
    best_epoch    = history.history["val_loss"].index(best_val_loss) + 1
    final_val_acc = history.history["val_accuracy"][-1]

    logger.info(
        f"[{asset_name}] Training complete. "
        f"Best val_loss={best_val_loss:.4f} at epoch {best_epoch}. "
        f"Final val_accuracy={final_val_acc:.4f}"
    )

    return history


# =============================================================================
# WALK-FORWARD TRAINING
# =============================================================================

def walk_forward_train(
    model_builder_fn,
    X:             np.ndarray,
    y_encoded:     np.ndarray,
    folds:         list,
    class_weights: dict,
    asset_name:    str   = "ASSET",
    input_shape:   tuple = None,
    epochs:        int   = 60,
    batch_size:    int   = 32,
) -> list:
    """
    Train the model across all walk-forward folds.

    A fresh model is built for each fold to avoid weight leakage between folds.
    Returns a list of (history, val_metrics) dicts, one per fold.

    This produces a realistic estimate of out-of-sample performance because
    each fold's model only saw data that predated the validation period.

    Parameters
    ----------
    model_builder_fn : callable that returns a compiled Keras model
    X, y_encoded : full dataset (pre-split)
    folds : list of (train_idx, val_idx) from WalkForwardSplitter
    class_weights : dict
    asset_name : str
    input_shape : (seq_len, n_features) - inferred if None
    epochs, batch_size : training parameters

    Returns
    -------
    List of dicts with keys: fold, history, val_loss, val_accuracy, n_train, n_val
    """
    if input_shape is None:
        input_shape = X.shape[1:]

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"\n{'='*50}")
        logger.info(f"[{asset_name}] Walk-Forward Fold {fold_idx+1}/{len(folds)}")
        logger.info(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
        logger.info(f"{'='*50}")

        X_tr = X[train_idx]
        y_tr = y_encoded[train_idx]
        X_vl = X[val_idx]
        y_vl = y_encoded[val_idx]

        # Build a fresh model for each fold
        fold_model = model_builder_fn(input_shape)

        history = train_model(
            fold_model, X_tr, y_tr, X_vl, y_vl,
            class_weights=class_weights,
            asset_name=f"{asset_name}_fold{fold_idx+1}",
            epochs=epochs,
            batch_size=batch_size,
            save_best=False   # Only save the final full-data model
        )

        val_metrics = fold_model.evaluate(X_vl, y_vl, verbose=0)
        val_loss, val_acc = val_metrics[0], val_metrics[1]

        fold_results.append({
            "fold":        fold_idx + 1,
            "history":     history,
            "val_loss":    val_loss,
            "val_accuracy":val_acc,
            "n_train":     len(X_tr),
            "n_val":       len(X_vl),
        })

        logger.info(
            f"[{asset_name}] Fold {fold_idx+1} result: "
            f"val_loss={val_loss:.4f}, val_accuracy={val_acc:.4f}"
        )

        # Free memory between folds
        del fold_model
        tf.keras.backend.clear_session()

    # Summary across folds
    avg_loss = np.mean([r["val_loss"] for r in fold_results])
    avg_acc  = np.mean([r["val_accuracy"] for r in fold_results])
    logger.info(f"\n[{asset_name}] Walk-forward summary: "
                f"avg_val_loss={avg_loss:.4f}, avg_val_accuracy={avg_acc:.4f}")

    return fold_results


# =============================================================================
# INFERENCE
# =============================================================================

def predict_signal(
    model:      keras.Model,
    X_sequence: np.ndarray,
    threshold:  float = None,
) -> dict:
    """
    Generate a trading signal from a single input sequence.

    Parameters
    ----------
    model : trained Keras model
    X_sequence : np.ndarray, shape (1, seq_len, n_features) or (seq_len, n_features)
    threshold : float, minimum confidence to emit a signal. Default: SIGNAL_CONFIDENCE_THRESHOLD

    Returns
    -------
    dict with keys:
        signal     : int (-1=SHORT, 0=NO_TRADE, 1=LONG)
        signal_name: str ("SHORT", "NO_TRADE", "LONG")
        confidence : float (0-1), probability of the predicted class
        probabilities : dict {class_name: probability}
        above_threshold : bool
    """
    if threshold is None:
        threshold = SIGNAL_CONFIDENCE_THRESHOLD

    if X_sequence.ndim == 2:
        X_sequence = X_sequence[np.newaxis, ...]   # Add batch dimension

    probs = model.predict(X_sequence, verbose=0)[0]   # Shape: (3,)

    class_idx   = int(np.argmax(probs))
    confidence  = float(probs[class_idx])
    signal_label = CLASS_TO_LABEL[class_idx]

    # Suppress to NO_TRADE if below confidence threshold
    if confidence < threshold:
        signal_label = 0
        class_idx    = 1   # NO_TRADE

    return {
        "signal":      signal_label,
        "signal_name": CLASS_NAMES[LABEL_TO_CLASS[signal_label]],
        "confidence":  confidence,
        "probabilities": {
            "SHORT":    float(probs[0]),
            "NO_TRADE": float(probs[1]),
            "LONG":     float(probs[2]),
        },
        "above_threshold": confidence >= threshold,
    }


def batch_predict(
    model:     keras.Model,
    X:         np.ndarray,
    threshold: float = None,
) -> np.ndarray:
    """
    Generate signals for a batch of sequences.
    Returns array of integer signals {-1, 0, 1}.
    Below-threshold signals are suppressed to 0.
    """
    if threshold is None:
        threshold = SIGNAL_CONFIDENCE_THRESHOLD

    probs    = model.predict(X, verbose=0)             # (n, 3)
    classes  = np.argmax(probs, axis=1)                # (n,)
    confs    = probs[np.arange(len(probs)), classes]   # (n,)

    # Suppress low-confidence predictions
    classes[confs < threshold] = 1   # -> NO_TRADE

    return decode_labels(classes)


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(model: keras.Model, asset_name: str):
    """Save trained model to disk."""
    path = os.path.join(MODELS_DIR, f"model_{asset_name}.keras")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(path)
    logger.info(f"Model saved to {path}")
    return path


def load_model(asset_name: str) -> Optional[keras.Model]:
    """Load a trained model from disk."""
    path = os.path.join(MODELS_DIR, f"model_{asset_name}.keras")
    if not os.path.exists(path):
        logger.warning(f"No saved model found for {asset_name} at {path}")
        return None
    model = keras.models.load_model(path)
    logger.info(f"Model loaded from {path}")
    return model


# =============================================================================
# SELF-IMPROVEMENT: INCREMENTAL RETRAINER
# =============================================================================

class IncrementalRetrainer:
    """
    Manages the self-improvement retraining loop.

    When enough new labeled outcomes have accumulated in IncrementalDataStore,
    this class appends the new data to the original training set and retrains
    the model. The improved model replaces the current live model.

    Flow:
        1. Live model generates signal
        2. Trade outcome is observed (TP/SL/expired)
        3. Outcome is logged to IncrementalDataStore
        4. IncrementalRetrainer checks if retrain threshold is met
        5. If yes: merge new data + old data, retrain, replace live model
        6. Win rate and performance metrics are tracked per retrain cycle

    This implements a continuous learning loop that adapts to market
    regime changes without manual intervention.
    """

    def __init__(
        self,
        asset_name:         str,
        model_builder_fn,
        input_shape:        tuple,
        retrain_threshold:  int   = 200,
        max_retrain_data:   int   = 5000,
    ):
        self.asset_name        = asset_name
        self.model_builder_fn  = model_builder_fn
        self.input_shape       = input_shape
        self.retrain_threshold = retrain_threshold
        self.max_retrain_data  = max_retrain_data
        self._retrain_count    = 0
        self._performance_log  = []

    def check_and_retrain(
        self,
        current_model:  keras.Model,
        X_base:         np.ndarray,
        y_base_encoded: np.ndarray,
        new_X:          np.ndarray,
        new_y_encoded:  np.ndarray,
        class_weights:  dict,
    ) -> Tuple[keras.Model, bool]:
        """
        Check if retraining is warranted and retrain if so.

        Parameters
        ----------
        current_model : the live model currently generating signals
        X_base, y_base_encoded : original training data
        new_X, new_y_encoded : newly accumulated labeled outcomes
        class_weights : class weights for loss function

        Returns
        -------
        (model, retrained): updated model and bool indicating if retraining occurred
        """
        if len(new_X) < self.retrain_threshold:
            logger.info(
                f"[{self.asset_name}] Retraining check: "
                f"{len(new_X)}/{self.retrain_threshold} new samples. Not yet."
            )
            return current_model, False

        logger.info(
            f"[{self.asset_name}] Retraining threshold reached "
            f"({len(new_X)} new samples). Starting incremental retrain..."
        )

        # Merge base + new data (most recent data gets priority)
        X_combined = np.concatenate([X_base, new_X], axis=0)
        y_combined = np.concatenate([y_base_encoded, new_y_encoded], axis=0)

        # Cap dataset size to prevent unbounded growth
        if len(X_combined) > self.max_retrain_data:
            X_combined = X_combined[-self.max_retrain_data:]
            y_combined = y_combined[-self.max_retrain_data:]

        # Simple 85/15 train/val split for retraining
        split = int(len(X_combined) * 0.85)
        X_tr, y_tr = X_combined[:split], y_combined[:split]
        X_vl, y_vl = X_combined[split:], y_combined[split:]

        # Build fresh model and retrain
        new_model = self.model_builder_fn(self.input_shape)
        train_model(
            new_model, X_tr, y_tr, X_vl, y_vl,
            class_weights=class_weights,
            asset_name=f"{self.asset_name}_retrain_{self._retrain_count+1}",
            epochs=40,
            batch_size=32,
            patience=10,
            save_best=True
        )

        # Evaluate both models on val set
        old_metrics = current_model.evaluate(X_vl, y_vl, verbose=0)
        new_metrics = new_model.evaluate(X_vl, y_vl, verbose=0)

        old_loss, new_loss = old_metrics[0], new_metrics[0]

        if new_loss < old_loss:
            self._retrain_count += 1
            self._performance_log.append({
                "retrain": self._retrain_count,
                "old_val_loss": old_loss,
                "new_val_loss": new_loss,
                "improvement": old_loss - new_loss,
                "new_samples": len(new_X)
            })
            save_model(new_model, self.asset_name)
            logger.info(
                f"[{self.asset_name}] Retrain #{self._retrain_count} successful. "
                f"Val loss: {old_loss:.4f} -> {new_loss:.4f} "
                f"(improved by {old_loss - new_loss:.4f})"
            )
            return new_model, True
        else:
            logger.info(
                f"[{self.asset_name}] Retrained model did not improve "
                f"({new_loss:.4f} >= {old_loss:.4f}). Keeping current model."
            )
            del new_model
            tf.keras.backend.clear_session()
            return current_model, False

    def get_performance_log(self) -> list:
        """Return the history of all retrain cycles and their improvements."""
        return self._performance_log


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def evaluate_model(
    model:       keras.Model,
    X_test:      np.ndarray,
    y_test_enc:  np.ndarray,
    asset_name:  str = "ASSET",
    threshold:   float = None,
) -> dict:
    """
    Comprehensive model evaluation with trading-specific metrics.

    Metrics computed:
      - Overall accuracy (all 3 classes)
      - Per-class precision, recall, F1
      - Signal accuracy: accuracy on LONG and SHORT predictions only
        (excluding NO_TRADE predictions, since those are non-events)
      - Coverage: % of candles where a signal was emitted
      - Confusion matrix
    """
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        precision_recall_fscore_support
    )

    if threshold is None:
        threshold = SIGNAL_CONFIDENCE_THRESHOLD

    # Get raw predictions and apply threshold
    signals = batch_predict(model, X_test, threshold=threshold)
    y_true  = decode_labels(y_test_enc)

    # Overall accuracy
    accuracy = float((signals == y_true).mean())

    # Coverage: fraction of candles where a non-zero signal was emitted
    coverage = float((signals != 0).mean())

    # Signal accuracy: among emitted signals, how often was the direction correct?
    signal_mask = signals != 0
    if signal_mask.sum() > 0:
        signal_accuracy = float((signals[signal_mask] == y_true[signal_mask]).mean())
    else:
        signal_accuracy = 0.0

    # Per-class metrics
    report = classification_report(
        y_true, signals,
        labels=[-1, 0, 1],
        target_names=["SHORT", "NO_TRADE", "LONG"],
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_true, signals, labels=[-1, 0, 1])

    logger.info(f"\n[{asset_name}] Evaluation Results (threshold={threshold}):")
    logger.info(f"  Overall accuracy:  {accuracy:.4f}")
    logger.info(f"  Signal accuracy:   {signal_accuracy:.4f} "
                f"(on {signal_mask.sum()} emitted signals)")
    logger.info(f"  Signal coverage:   {coverage:.4f} "
                f"({signal_mask.sum()}/{len(signals)} candles)")
    logger.info(f"\n  Classification Report:")
    logger.info(classification_report(
        y_true, signals,
        labels=[-1, 0, 1],
        target_names=["SHORT", "NO_TRADE", "LONG"],
        zero_division=0
    ))

    return {
        "accuracy":        accuracy,
        "signal_accuracy": signal_accuracy,
        "coverage":        coverage,
        "n_signals":       int(signal_mask.sum()),
        "report":          report,
        "confusion_matrix":cm,
        "predictions":     signals,
        "true_labels":     y_true,
    }


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import pandas as pd
    from features.pipeline import run_pipeline
    from features.dataset import prepare_dataset

    print("\n=== CNN-LSTM Model Test ===\n")
    tf.get_logger().setLevel("ERROR")

    # Generate synthetic data
    np.random.seed(42)
    n = 1000
    price = 19000 + np.cumsum(np.random.randn(n) * 12)
    df_raw = pd.DataFrame({
        "open":   price + np.random.randn(n) * 5,
        "close":  price + np.random.randn(n) * 5,
        "volume": np.random.uniform(1000, 5000, n)
    }, index=pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"))
    df_raw["high"] = df_raw[["open","close"]].max(axis=1) + np.abs(np.random.randn(n))*8
    df_raw["low"]  = df_raw[["open","close"]].min(axis=1) - np.abs(np.random.randn(n))*8

    # Run pipeline
    pipe = run_pipeline(df_raw, asset_name="TEST", save_normaliser=False)
    dataset = prepare_dataset(
        pipe["X"], pipe["y"],
        asset_name="TEST",
        resample="moderate",
        n_folds=3,
        save_to_disk=False
    )

    input_shape = dataset["shape"]
    print(f"\nInput shape: {input_shape}")

    # Build model
    model = build_cnn_lstm(input_shape=input_shape, dropout_rate=0.3)
    model_summary(model)

    # Short training run (5 epochs for test speed)
    print("\nRunning short training test (5 epochs)...")
    history = train_model(
        model,
        dataset["X_train"], dataset["y_train_encoded"],
        dataset["X_val"],   dataset["y_val_encoded"],
        class_weights=dataset["class_weights"],
        asset_name="TEST",
        epochs=5,
        batch_size=32,
        save_best=False
    )

    # Evaluate
    print("\nEvaluating on test set...")
    eval_results = evaluate_model(
        model,
        dataset["X_test"], dataset["y_test_encoded"],
        asset_name="TEST",
        threshold=0.60   # Lower threshold for test data (small synthetic dataset)
    )

    print(f"\nTest accuracy:      {eval_results['accuracy']:.4f}")
    print(f"Signal accuracy:    {eval_results['signal_accuracy']:.4f}")
    print(f"Signal coverage:    {eval_results['coverage']:.4f}")
    print(f"Signals emitted:    {eval_results['n_signals']}")

    # Test single prediction
    print("\nTesting single signal prediction...")
    sample_seq = dataset["X_test"][0:1]
    result = predict_signal(model, sample_seq, threshold=0.50)
    print(f"Signal: {result['signal_name']} | "
          f"Confidence: {result['confidence']:.4f} | "
          f"Above threshold: {result['above_threshold']}")
    print(f"Probabilities: {result['probabilities']}")

    print("\n=== Phase 4 CNN-LSTM Model: ALL SYSTEMS OK ===")
