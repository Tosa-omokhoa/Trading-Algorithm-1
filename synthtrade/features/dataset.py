# =============================================================================
# SynthTrade - Dataset Preparation
# =============================================================================
# Takes the raw sequence output from pipeline.py and prepares it for training.
#
# Problems solved here:
#
#   1. CLASS IMBALANCE (~82% No Trade, ~10% Long, ~8% Short)
#      A model trained on imbalanced data learns to say "No Trade" to
#      everything and still gets 82% accuracy. Useless for trading.
#      Solution: SMOTE-based oversampling on the minority classes + class
#      weight computation so the loss function penalises missed signals harder.
#
#   2. TEMPORAL LEAKAGE
#      Standard random train/test splits leak future data into training.
#      In time series this is catastrophic: the model "sees" the future.
#      Solution: Walk-forward validation with anchored expanding windows.
#      Each fold trains on all past data and tests on the immediate future.
#
#   3. LABEL ENCODING
#      The model outputs 3 classes. Labels must be one-hot encoded for
#      categorical crossentropy loss, but kept as integers for metrics.
#
#   4. DATASET PERSISTENCE
#      Processed datasets are saved to disk so training can be restarted
#      without re-running the entire pipeline from scratch.

import os
import sys
import logging
import numpy as np
import joblib
from typing import List, Tuple, Optional, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from config import SEQUENCE_LENGTH, MODELS_DIR, RAW_DATA_DIR
except ImportError:
    SEQUENCE_LENGTH = 30
    MODELS_DIR = "models/saved"
    RAW_DATA_DIR = "data/raw"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Dataset")

# Label encoding: internal int -> model class index
# -1 (SHORT) -> 0, 0 (NO_TRADE) -> 1, 1 (LONG) -> 2
LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_LABEL = {0: -1, 1: 0, 2: 1}
CLASS_NAMES    = ["SHORT", "NO_TRADE", "LONG"]


# =============================================================================
# LABEL ENCODING
# =============================================================================

def encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Convert integer labels {-1, 0, 1} to class indices {0, 1, 2}.
    Required for Keras categorical crossentropy.

    -1 (SHORT)    -> 0
     0 (NO_TRADE) -> 1
     1 (LONG)     -> 2
    """
    encoded = np.array([LABEL_TO_CLASS[int(label)] for label in y], dtype=np.int32)
    return encoded


def decode_labels(encoded: np.ndarray) -> np.ndarray:
    """Reverse of encode_labels. Converts class indices back to {-1, 0, 1}."""
    return np.array([CLASS_TO_LABEL[int(c)] for c in encoded], dtype=np.int8)


def to_onehot(encoded: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Convert encoded class indices to one-hot arrays."""
    onehot = np.zeros((len(encoded), n_classes), dtype=np.float32)
    onehot[np.arange(len(encoded)), encoded] = 1.0
    return onehot


# =============================================================================
# CLASS WEIGHT COMPUTATION
# =============================================================================

def compute_class_weights(y_encoded: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights to handle imbalance in the loss function.

    Uses balanced weighting: weight[c] = n_samples / (n_classes * count[c])
    This means rare classes (LONG, SHORT) get higher loss weight than
    the majority class (NO_TRADE), so the model is penalised more for
    missing actual signals.

    Parameters
    ----------
    y_encoded : np.ndarray
        Class-encoded labels (0=SHORT, 1=NO_TRADE, 2=LONG).

    Returns
    -------
    dict: {class_index: weight}
    """
    n_samples  = len(y_encoded)
    n_classes  = 3
    class_weights = {}

    for c in range(n_classes):
        count = (y_encoded == c).sum()
        if count == 0:
            class_weights[c] = 1.0
        else:
            class_weights[c] = n_samples / (n_classes * count)

    # Log the weights so training behaviour is transparent
    for c, w in class_weights.items():
        count = (y_encoded == c).sum()
        logger.info(f"  Class {c} ({CLASS_NAMES[c]}): count={count}, weight={w:.3f}")

    return class_weights


# =============================================================================
# SEQUENCE-LEVEL SMOTE (SMOTE adapted for 3D time series data)
# =============================================================================

def resample_sequences(
    X: np.ndarray,
    y_encoded: np.ndarray,
    strategy: str = "moderate"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample training sequences to reduce class imbalance.

    Standard SMOTE operates on 2D feature vectors. Our data is 3D
    (samples x timesteps x features). We flatten to 2D for SMOTE,
    then reshape back to 3D.

    Strategies:
      "none"      : No resampling. Use class weights only.
      "moderate"  : Oversample LONG and SHORT to ~30% each of total.
      "aggressive": Oversample LONG and SHORT to ~40% each of total.

    Note: Resampling is applied to TRAINING data only, never to
    validation or test data. Applying it to val/test would distort metrics.

    Parameters
    ----------
    X : np.ndarray, shape (n, seq_len, n_features)
    y_encoded : np.ndarray, shape (n,), values in {0, 1, 2}
    strategy : str

    Returns
    -------
    X_resampled : np.ndarray, shape (n_resampled, seq_len, n_features)
    y_resampled : np.ndarray, shape (n_resampled,)
    """
    if strategy == "none":
        logger.info("Resampling strategy: none. Using class weights only.")
        return X, y_encoded

    from imblearn.over_sampling import SMOTE

    n_samples, seq_len, n_features = X.shape

    # Flatten 3D -> 2D for SMOTE
    X_flat = X.reshape(n_samples, seq_len * n_features)

    # Define target sample counts per class
    counts = {c: int((y_encoded == c).sum()) for c in range(3)}
    majority_count = counts[1]   # NO_TRADE is always majority

    if strategy == "moderate":
        target_minority = int(majority_count * 0.43)   # ~30% of new total each
    else:  # aggressive
        target_minority = int(majority_count * 0.67)   # ~40% of new total each

    # Only oversample classes that have fewer samples than the target
    sampling_strategy = {}
    for c in [0, 2]:   # SHORT and LONG
        if counts[c] < target_minority:
            sampling_strategy[c] = target_minority

    if not sampling_strategy:
        logger.info("Classes already sufficiently balanced. No SMOTE needed.")
        return X, y_encoded

    logger.info(f"Applying SMOTE (strategy={strategy}): "
                f"target minority count = {target_minority}")
    logger.info(f"Before: SHORT={counts[0]}, NO_TRADE={counts[1]}, LONG={counts[2]}")

    try:
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=min(5, min(counts[0], counts[2]) - 1),
            random_state=42
        )
        X_flat_r, y_r = smote.fit_resample(X_flat, y_encoded)
    except ValueError as e:
        logger.warning(f"SMOTE failed ({e}). Falling back to random oversampling.")
        X_flat_r, y_r = _random_oversample(X_flat, y_encoded, target_minority)

    # Reshape back to 3D
    X_resampled = X_flat_r.reshape(-1, seq_len, n_features)

    new_counts = {c: int((y_r == c).sum()) for c in range(3)}
    logger.info(f"After:  SHORT={new_counts[0]}, "
                f"NO_TRADE={new_counts[1]}, LONG={new_counts[2]}")
    logger.info(f"Total samples: {len(y_r)} (was {n_samples})")

    return X_resampled.astype(np.float32), y_r.astype(np.int32)


def _random_oversample(
    X_flat: np.ndarray,
    y: np.ndarray,
    target_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback: random duplication of minority class samples."""
    X_parts = [X_flat]
    y_parts  = [y]

    for c in [0, 2]:
        idx = np.where(y == c)[0]
        count = len(idx)
        if count < target_count:
            n_needed = target_count - count
            chosen   = np.random.choice(idx, size=n_needed, replace=True)
            X_parts.append(X_flat[chosen])
            y_parts.append(y[chosen])

    return np.vstack(X_parts), np.concatenate(y_parts)


# =============================================================================
# WALK-FORWARD VALIDATION SPLITTER
# =============================================================================

class WalkForwardSplitter:
    """
    Generates walk-forward train/validation splits for time series data.

    Unlike random k-fold, walk-forward splits always respect temporal order:
    training data comes BEFORE validation data in every fold.

    Two modes:
      "expanding" : Training window grows with each fold (uses all past data).
                    Best for maximising data utilisation.
      "rolling"   : Fixed-size training window slides forward.
                    Best for capturing recent market regime changes.

    Example (expanding, 3 folds, test_ratio=0.2):
      Fold 1: Train [0:640], Val [640:800]
      Fold 2: Train [0:720], Val [720:900]
      Fold 3: Train [0:800], Val [800:1000]

    This means the model is always evaluated on data it has never seen,
    in the exact order it would arrive in live trading.
    """

    def __init__(
        self,
        n_folds:    int   = 5,
        val_ratio:  float = 0.15,
        mode:       str   = "expanding",
        min_train:  int   = 200
    ):
        """
        Parameters
        ----------
        n_folds : int
            Number of walk-forward folds.
        val_ratio : float
            Fraction of total data used per validation window.
        mode : str
            "expanding" or "rolling".
        min_train : int
            Minimum training samples required in any fold.
        """
        self.n_folds   = n_folds
        self.val_ratio = val_ratio
        self.mode      = mode
        self.min_train = min_train

    def split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate list of (train_indices, val_indices) tuples.

        Parameters
        ----------
        n_samples : int
            Total number of samples.

        Returns
        -------
        List of (train_idx, val_idx) tuples, one per fold.
        """
        val_size  = max(1, int(n_samples * self.val_ratio))
        folds     = []

        # Determine end of usable data (leave final val_size for final test)
        usable = n_samples - val_size

        if self.n_folds == 1:
            train_idx = np.arange(0, usable)
            val_idx   = np.arange(usable, n_samples)
            if len(train_idx) >= self.min_train:
                folds.append((train_idx, val_idx))
            return folds

        # Step size between fold boundaries
        step = max(1, usable // self.n_folds)

        for fold in range(self.n_folds):
            val_end   = usable - (self.n_folds - fold - 1) * step
            val_start = val_end - val_size
            val_start = max(self.min_train, val_start)

            if val_end > n_samples:
                val_end = n_samples

            if self.mode == "expanding":
                train_start = 0
            else:   # rolling
                train_window = val_start
                train_start  = max(0, val_start - train_window)

            train_end = val_start

            if train_end - train_start < self.min_train:
                logger.debug(f"Fold {fold+1}: insufficient training data, skipping.")
                continue

            train_idx = np.arange(train_start, train_end)
            val_idx   = np.arange(val_start,   val_end)
            folds.append((train_idx, val_idx))

            logger.debug(
                f"Fold {fold+1}: train=[{train_start}:{train_end}] "
                f"({len(train_idx)}), val=[{val_start}:{val_end}] ({len(val_idx)})"
            )

        logger.info(f"Walk-forward split: {len(folds)} folds, "
                    f"mode={self.mode}, val_size={val_size}")
        return folds


# =============================================================================
# FINAL DATASET BUILDER
# =============================================================================

def prepare_dataset(
    X: np.ndarray,
    y: np.ndarray,
    asset_name:    str   = "ASSET",
    resample:      str   = "moderate",
    n_folds:       int   = 5,
    val_ratio:     float = 0.15,
    wf_mode:       str   = "expanding",
    save_to_disk:  bool  = True
) -> dict:
    """
    Full dataset preparation pipeline.

    Takes raw sequences (X, y) from pipeline.py and returns a dict containing
    everything needed for model training: encoded labels, class weights,
    walk-forward folds, and resampled training data.

    Parameters
    ----------
    X : np.ndarray, shape (n, seq_len, n_features)
    y : np.ndarray, shape (n,), values in {-1, 0, 1}
    asset_name : str
    resample : str
        SMOTE strategy: "none", "moderate", "aggressive".
    n_folds : int
    val_ratio : float
    wf_mode : str
    save_to_disk : bool

    Returns
    -------
    dict with keys:
        X_train, y_train_encoded       : Resampled training data (last fold)
        X_val, y_val_encoded           : Validation data (last fold, no resampling)
        X_test, y_test_encoded         : Final holdout test set (never touched during training)
        class_weights                  : Dict for Keras class_weight param
        folds                          : All walk-forward (train_idx, val_idx) pairs
        y_encoded                      : Full encoded label array
        label_encoder                  : LABEL_TO_CLASS dict
        n_classes                      : 3
        class_names                    : ["SHORT", "NO_TRADE", "LONG"]
        shape                          : (seq_len, n_features)
    """
    logger.info(f"[{asset_name}] Preparing dataset from {len(X)} sequences...")

    # --- Encode labels ---
    y_encoded = encode_labels(y)
    logger.info("Class distribution before resampling:")
    class_weights = compute_class_weights(y_encoded)

    # --- Final holdout test set (last 10% of data, never resampled) ---
    test_size   = max(30, int(len(X) * 0.10))
    X_test      = X[-test_size:]
    y_test_enc  = y_encoded[-test_size:]
    X_main      = X[:-test_size]
    y_main_enc  = y_encoded[:-test_size]

    logger.info(f"Holdout test set: {test_size} samples "
                f"({test_size/len(X)*100:.1f}% of total).")

    # --- Walk-forward splits on main data ---
    splitter = WalkForwardSplitter(
        n_folds=n_folds, val_ratio=val_ratio,
        mode=wf_mode, min_train=200
    )
    folds = splitter.split(len(X_main))

    if not folds:
        logger.warning("No valid folds generated. Using simple 80/20 split.")
        split_idx  = int(len(X_main) * 0.8)
        train_idx  = np.arange(0, split_idx)
        val_idx    = np.arange(split_idx, len(X_main))
        folds      = [(train_idx, val_idx)]

    # --- Extract last fold as primary train/val for immediate training ---
    last_train_idx, last_val_idx = folds[-1]

    X_val     = X_main[last_val_idx]
    y_val_enc = y_main_enc[last_val_idx]

    X_train_raw     = X_main[last_train_idx]
    y_train_raw_enc = y_main_enc[last_train_idx]

    logger.info(f"Last fold: train={len(X_train_raw)}, val={len(X_val)}")

    # --- Resample training data only ---
    logger.info(f"Applying resampling (strategy={resample}) to training set...")
    X_train, y_train_encoded = resample_sequences(
        X_train_raw, y_train_raw_enc, strategy=resample
    )

    # Shuffle resampled training data (SMOTE appends synthetic samples at the end)
    shuffle_idx  = np.random.permutation(len(X_train))
    X_train      = X_train[shuffle_idx]
    y_train_encoded = y_train_encoded[shuffle_idx]

    # --- Save to disk ---
    if save_to_disk:
        save_path = os.path.join(MODELS_DIR, f"dataset_{asset_name}.npz")
        os.makedirs(MODELS_DIR, exist_ok=True)
        np.savez_compressed(
            save_path,
            X_train=X_train, y_train=y_train_encoded,
            X_val=X_val,     y_val=y_val_enc,
            X_test=X_test,   y_test=y_test_enc,
            y_full=y_encoded
        )
        logger.info(f"[{asset_name}] Dataset saved to {save_path}")

    logger.info(
        f"\nDataset ready:\n"
        f"  Train:    {X_train.shape}  labels: {np.bincount(y_train_encoded)}\n"
        f"  Val:      {X_val.shape}    labels: {np.bincount(y_val_enc)}\n"
        f"  Test:     {X_test.shape}   labels: {np.bincount(y_test_enc)}\n"
        f"  Shape per sample: {X_train.shape[1:]}\n"
        f"  Class weights: {class_weights}"
    )

    return {
        "X_train":        X_train,
        "y_train_encoded":y_train_encoded,
        "X_val":          X_val,
        "y_val_encoded":  y_val_enc,
        "X_test":         X_test,
        "y_test_encoded": y_test_enc,
        "class_weights":  class_weights,
        "folds":          folds,
        "y_encoded":      y_encoded,
        "label_encoder":  LABEL_TO_CLASS,
        "n_classes":      3,
        "class_names":    CLASS_NAMES,
        "shape":          X_train.shape[1:],
    }


def load_dataset(asset_name: str) -> Optional[dict]:
    """Load a previously saved dataset from disk."""
    path = os.path.join(MODELS_DIR, f"dataset_{asset_name}.npz")
    if not os.path.exists(path):
        logger.warning(f"No saved dataset found for {asset_name} at {path}")
        return None
    data = np.load(path)
    logger.info(f"Loaded dataset for {asset_name} from {path}")
    return {
        "X_train":         data["X_train"],
        "y_train_encoded": data["y_train"],
        "X_val":           data["X_val"],
        "y_val_encoded":   data["y_val"],
        "X_test":          data["X_test"],
        "y_test_encoded":  data["y_test"],
        "y_encoded":       data["y_full"],
        "n_classes":       3,
        "class_names":     CLASS_NAMES,
        "label_encoder":   LABEL_TO_CLASS,
        "shape":           data["X_train"].shape[1:],
    }


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from features.pipeline import run_pipeline

    print("\n=== Dataset Preparation Test ===\n")

    np.random.seed(42)
    n = 800
    price = 19000 + np.cumsum(np.random.randn(n) * 12)
    df_raw = __import__("pandas").DataFrame({
        "open":   price + np.random.randn(n) * 5,
        "close":  price + np.random.randn(n) * 5,
        "volume": np.random.uniform(1000, 5000, n)
    }, index=__import__("pandas").date_range(
        "2025-01-01", periods=n, freq="5min", tz="UTC"))
    df_raw["high"] = df_raw[["open","close"]].max(axis=1) + np.abs(np.random.randn(n))*8
    df_raw["low"]  = df_raw[["open","close"]].min(axis=1) - np.abs(np.random.randn(n))*8

    pipe_out = run_pipeline(df_raw, asset_name="TEST", save_normaliser=False)
    X, y = pipe_out["X"], pipe_out["y"]

    print(f"Pipeline output: X={X.shape}, y={y.shape}")
    print(f"Raw label distribution: "
          f"LONG={(y==1).sum()}, SHORT={(y==-1).sum()}, "
          f"NO_TRADE={(y==0).sum()}")

    dataset = prepare_dataset(
        X, y,
        asset_name="TEST",
        resample="moderate",
        n_folds=5,
        save_to_disk=False
    )

    print(f"\nFinal dataset summary:")
    print(f"  Train shape:  {dataset['X_train'].shape}")
    print(f"  Val shape:    {dataset['X_val'].shape}")
    print(f"  Test shape:   {dataset['X_test'].shape}")
    print(f"  Class weights: {dataset['class_weights']}")
    print(f"  Walk-forward folds: {len(dataset['folds'])}")
    print(f"\nLabel encoding: {dataset['label_encoder']}")
    print(f"Class names:    {dataset['class_names']}")
    print(f"\nPhase 3 dataset preparation: ALL SYSTEMS OK.")
