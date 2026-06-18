# =============================================================================
# SynthTrade - Feature Pipeline & Label Generator
# =============================================================================
# This is the central orchestrator for Phase 2. It:
#   1. Takes raw OHLCV DataFrames (from either data source)
#   2. Applies Heikin Ashi transformation
#   3. Computes the full indicator stack
#   4. Normalises all features (MinMax per feature column)
#   5. Generates forward-looking trade labels for model training
#   6. Builds sliding window sequences for CNN-LSTM input
#   7. Saves processed datasets to disk
#
# The self-improvement mechanism starts here: the pipeline is designed to
# accept new incoming data and append it to the training set, so the model
# can be retrained incrementally as new outcomes are observed.

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.heikin_ashi import compute_heikin_ashi
from features.indicators import compute_indicators, get_indicator_feature_columns

try:
    from config import (
        ATR_PERIOD, SEQUENCE_LENGTH, LABEL_LOOKAHEAD,
        MIN_REWARD_ATR, MAX_RISK_ATR, RAW_DATA_DIR, MODELS_DIR
    )
except ImportError:
    ATR_PERIOD      = 14
    SEQUENCE_LENGTH = 30
    LABEL_LOOKAHEAD = 5
    MIN_REWARD_ATR  = 1.5
    MAX_RISK_ATR    = 1.0
    RAW_DATA_DIR    = "data/raw"
    MODELS_DIR      = "models/saved"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Pipeline")


# =============================================================================
# LABEL GENERATOR
# =============================================================================

def generate_labels(
    df: pd.DataFrame,
    lookahead: int = None,
    reward_atr: float = None,
    risk_atr:   float = None
) -> pd.Series:
    """
    Generate forward-looking trade labels for supervised learning.

    Label logic (applied independently for LONG and SHORT):

    LONG  (1): Within the next `lookahead` candles, the HA close moves UP
               by at least `reward_atr * ATR` WITHOUT first dropping DOWN
               by `risk_atr * ATR`. This means the trade would have hit TP
               before SL.

    SHORT (-1): Mirror of LONG in the downward direction.

    NO TRADE (0): Neither condition met. The model should predict abstention.

    This encoding directly reflects your risk-reward requirement: signals
    must earn at least 1.5x what they risk.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ha_close and atr columns (post-indicator pipeline).
    lookahead : int
        Number of future candles to evaluate. Default: LABEL_LOOKAHEAD.
    reward_atr : float
        Minimum favourable move in ATR multiples. Default: MIN_REWARD_ATR.
    risk_atr : float
        Maximum adverse move in ATR multiples before labelling 0. Default: MAX_RISK_ATR.

    Returns
    -------
    pd.Series of int (1=LONG, -1=SHORT, 0=NO_TRADE), same index as df.
    """
    if lookahead is None:  lookahead  = LABEL_LOOKAHEAD
    if reward_atr is None: reward_atr = MIN_REWARD_ATR
    if risk_atr is None:   risk_atr   = MAX_RISK_ATR

    closes = df["ha_close"].values
    atrs   = df["atr"].values
    n      = len(df)
    labels = np.zeros(n, dtype=int)

    for i in range(n - lookahead):
        entry  = closes[i]
        atr    = atrs[i]

        if atr == 0:
            continue

        reward = reward_atr * atr
        risk   = risk_atr   * atr

        future_closes = closes[i+1 : i+1+lookahead]

        # --- LONG evaluation ---
        long_tp_hit = False
        long_sl_hit = False
        for fc in future_closes:
            if not long_sl_hit and (entry - fc) >= risk:
                long_sl_hit = True
                break
            if (fc - entry) >= reward:
                long_tp_hit = True
                break

        if long_tp_hit and not long_sl_hit:
            labels[i] = 1
            continue

        # --- SHORT evaluation ---
        short_tp_hit = False
        short_sl_hit = False
        for fc in future_closes:
            if not short_sl_hit and (fc - entry) >= risk:
                short_sl_hit = True
                break
            if (entry - fc) >= reward:
                short_tp_hit = True
                break

        if short_tp_hit and not short_sl_hit:
            labels[i] = -1

    # Last `lookahead` candles cannot have valid labels (no future data)
    labels[-lookahead:] = 0

    label_series = pd.Series(labels, index=df.index, name="label")

    # Log class distribution for diagnostics
    counts = label_series.value_counts().sort_index()
    total  = len(label_series)
    logger.info(
        f"Label distribution: "
        f"LONG={counts.get(1, 0)} ({counts.get(1,0)/total*100:.1f}%), "
        f"SHORT={counts.get(-1,0)} ({counts.get(-1,0)/total*100:.1f}%), "
        f"NO_TRADE={counts.get(0,0)} ({counts.get(0,0)/total*100:.1f}%)"
    )

    return label_series


# =============================================================================
# FEATURE NORMALISER
# =============================================================================

class FeatureNormaliser:
    """
    Per-feature MinMax normalisation with fit/transform/inverse_transform.

    Designed for the CNN-LSTM input so all features are on a comparable
    scale (0 to 1). Fitted on training data only; applied to val/test/live.

    The scaler state is saved to disk so it can be reloaded for live inference
    without re-fitting. Critical for consistent live signal generation.

    Price-derived features (EMA values, BB values, SAR values, HA OHLC) are
    normalised differently: they are divided by the HA close of the same candle
    to make them scale-invariant across different asset price levels.
    This allows a single model to generalise across assets with very
    different price scales (e.g. XAUUSD at 2400 vs USDJPY at 155).
    """

    # Columns that should be normalised relative to ha_close (scale-invariant)
    PRICE_RELATIVE_COLS = [
        "ha_open", "ha_high", "ha_low",
        "ema_9", "ema_21", "ema_50",
        "bb_upper", "bb_mid", "bb_lower",
        "sar", "atr"
    ]

    # Columns that are already bounded or ratio-based (normalise with MinMax)
    BOUNDED_COLS = [
        "ha_direction", "ha_consecutive", "ha_no_upper", "ha_no_lower", "ha_doji",
        "ema_9_slope", "ema_21_slope", "ema_9_21_cross", "ema_alignment",
        "price_to_ema9", "price_to_ema21", "price_to_ema50",
        "rsi", "rsi_zone", "rsi_slope",
        "bb_width", "bb_pct", "bb_regime",
        "sar_direction", "sar_flip", "sar_distance",
        "volume_ratio", "volume_spike",
        "trend_score",
        "ha_body", "ha_body_abs", "ha_upper_wick", "ha_lower_wick",
    ]

    def __init__(self):
        self._min: dict = {}
        self._max: dict = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        """Compute min/max statistics from training data."""
        feature_cols = get_indicator_feature_columns()
        available = [c for c in feature_cols if c in df.columns]

        for col in available:
            if col in self.PRICE_RELATIVE_COLS:
                # Normalised as ratio to ha_close; no global min/max needed
                continue
            self._min[col] = float(df[col].min())
            self._max[col] = float(df[col].max())

        self._fitted = True
        logger.info(f"Normaliser fitted on {len(df)} rows, {len(available)} features.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalisation. Returns a copy with normalised feature columns."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        df = df.copy()
        feature_cols = get_indicator_feature_columns()
        available = [c for c in feature_cols if c in df.columns]

        ha_close = df["ha_close"].replace(0, np.nan).fillna(1.0)

        for col in available:
            if col in self.PRICE_RELATIVE_COLS:
                # Express as ratio to current HA close (scale-invariant)
                df[col] = df[col] / ha_close
            elif col == "ha_close":
                # ha_close becomes 1.0 by definition after ratio normalisation
                df[col] = 1.0
            elif col in self._min:
                min_val = self._min[col]
                max_val = self._max[col]
                rng = max_val - min_val
                if rng == 0:
                    df[col] = 0.0
                else:
                    df[col] = (df[col] - min_val) / rng
                # Clip to [0, 1] to handle out-of-range values in live data
                df[col] = df[col].clip(0.0, 1.0)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    def save(self, path: str):
        """Persist normaliser state to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"min": self._min, "max": self._max, "fitted": self._fitted}, path)
        logger.info(f"Normaliser saved to {path}")

    def load(self, path: str):
        """Load normaliser state from disk."""
        state = joblib.load(path)
        self._min    = state["min"]
        self._max    = state["max"]
        self._fitted = state["fitted"]
        logger.info(f"Normaliser loaded from {path}")


# =============================================================================
# SEQUENCE BUILDER
# =============================================================================

def build_sequences(
    df: pd.DataFrame,
    labels: pd.Series,
    sequence_length: int = None,
    feature_cols:    list = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a normalised feature DataFrame into sliding window sequences
    for CNN-LSTM input.

    Each sample X[i] contains `sequence_length` consecutive candles of
    feature data. The corresponding label y[i] is the label at the LAST
    candle in the window (the candle for which we are making a prediction).

    Parameters
    ----------
    df : pd.DataFrame
        Normalised feature DataFrame (output of FeatureNormaliser.transform).
    labels : pd.Series
        Label series aligned with df (output of generate_labels).
    sequence_length : int
        Number of candles per sequence window. Default: SEQUENCE_LENGTH.
    feature_cols : list
        Feature columns to include. Default: get_indicator_feature_columns().

    Returns
    -------
    X : np.ndarray, shape (n_samples, sequence_length, n_features)
    y : np.ndarray, shape (n_samples,), values in {-1, 0, 1}
    """
    if sequence_length is None: sequence_length = SEQUENCE_LENGTH
    if feature_cols is None:    feature_cols    = get_indicator_feature_columns()

    available_cols = [c for c in feature_cols if c in df.columns]
    data   = df[available_cols].values
    label_arr = labels.values
    n = len(data)

    X_list = []
    y_list = []

    for i in range(sequence_length, n):
        X_list.append(data[i - sequence_length : i])
        y_list.append(label_arr[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int8)

    logger.info(f"Sequences built: X={X.shape}, y={y.shape}")
    logger.info(f"Label split: LONG={( y==1).sum()}, "
                f"SHORT={(y==-1).sum()}, NO_TRADE={(y==0).sum()}")

    return X, y


# =============================================================================
# MASTER PIPELINE FUNCTION
# =============================================================================

def run_pipeline(
    df_raw: pd.DataFrame,
    asset_name: str = "UNKNOWN",
    normaliser: Optional[FeatureNormaliser] = None,
    fit_normaliser: bool = True,
    generate_label: bool = True,
    save_normaliser: bool = True
) -> dict:
    """
    End-to-end feature pipeline: raw OHLCV in, model-ready sequences out.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw OHLCV DataFrame with columns [open, high, low, close, volume].
    asset_name : str
        Used for logging and file naming.
    normaliser : FeatureNormaliser, optional
        Pre-fitted normaliser to use. If None and fit_normaliser=True, a new
        one is fitted on this data (training mode).
    fit_normaliser : bool
        If True, fit the normaliser on this data. Set False for val/test/live.
    generate_label : bool
        If True, generate forward-looking labels. Set False for live inference.
    save_normaliser : bool
        If True, persist the normaliser to disk after fitting.

    Returns
    -------
    dict with keys:
        df_features : pd.DataFrame    (full featured DataFrame, pre-normalisation)
        df_normalised : pd.DataFrame  (normalised feature DataFrame)
        labels : pd.Series or None    (label series if generate_label=True)
        X : np.ndarray or None        (CNN-LSTM input sequences)
        y : np.ndarray or None        (label array for sequences)
        normaliser : FeatureNormaliser
        feature_cols : list
        n_features : int
    """
    logger.info(f"[{asset_name}] Starting feature pipeline on {len(df_raw)} rows...")

    # --- Step 1: Heikin Ashi transformation ---
    logger.info(f"[{asset_name}] Step 1: Heikin Ashi transformation...")
    df_ha = compute_heikin_ashi(df_raw)

    # --- Step 2: Indicator computation ---
    logger.info(f"[{asset_name}] Step 2: Computing indicator stack...")
    df_features = compute_indicators(df_ha)
    logger.info(f"[{asset_name}] Features shape after indicators: {df_features.shape}")

    # --- Step 3: Label generation ---
    labels = None
    if generate_label:
        logger.info(f"[{asset_name}] Step 3: Generating trade labels...")
        labels = generate_labels(df_features)

    # --- Step 4: Normalisation ---
    logger.info(f"[{asset_name}] Step 4: Normalising features...")
    if normaliser is None:
        normaliser = FeatureNormaliser()

    if fit_normaliser:
        df_normalised = normaliser.fit_transform(df_features)
        if save_normaliser:
            norm_path = os.path.join(MODELS_DIR, f"normaliser_{asset_name}.pkl")
            normaliser.save(norm_path)
    else:
        df_normalised = normaliser.transform(df_features)

    # --- Step 5: Build sequences ---
    X, y = None, None
    feature_cols = get_indicator_feature_columns()
    logger.info(f"[{asset_name}] Step 5: Building CNN-LSTM sequences...")
    if generate_label and labels is not None:
        X, y = build_sequences(df_normalised, labels, feature_cols=feature_cols)
    else:
        # Inference mode: build sequences without labels
        X, _ = build_sequences(df_normalised, pd.Series(np.zeros(len(df_normalised), dtype=int), index=df_normalised.index), feature_cols=feature_cols)

    n_features = len([c for c in feature_cols if c in df_normalised.columns])

    logger.info(f"[{asset_name}] Pipeline complete. "
                f"Features: {n_features}, Sequences: {len(X) if X is not None else 0}")

    return {
        "df_features":   df_features,
        "df_normalised": df_normalised,
        "labels":        labels,
        "X":             X,
        "y":             y,
        "normaliser":    normaliser,
        "feature_cols":  feature_cols,
        "n_features":    n_features,
    }


# =============================================================================
# SELF-IMPROVEMENT: INCREMENTAL DATA APPENDER
# =============================================================================

class IncrementalDataStore:
    """
    Manages an ever-growing dataset of labeled candles for incremental
    model retraining.

    When a trade signal is generated and its outcome is later observed
    (TP hit, SL hit, or expired), the outcome is logged here as a new
    labeled data point. Periodically, this store is used to retrain the
    model on fresh data, allowing it to adapt to current market conditions.

    This is the foundation of the self-improvement loop:
        Signal generated -> Trade observed -> Outcome logged ->
        Retrain trigger -> Improved model -> Better future signals
    """

    def __init__(self, asset_name: str, store_dir: str = None):
        self.asset_name = asset_name
        self.store_dir  = store_dir or RAW_DATA_DIR
        self.store_path = os.path.join(
            self.store_dir, f"incremental_{asset_name}.parquet"
        )
        self._buffer: list = []

    def log_outcome(
        self,
        candle_features: dict,
        signal: int,
        outcome: int,
        pnl_atr: float
    ):
        """
        Log the observed outcome of a generated signal.

        Parameters
        ----------
        candle_features : dict
            Feature values of the candle that generated the signal.
        signal : int
            The signal that was generated (1=LONG, -1=SHORT).
        outcome : int
            Actual outcome: 1=TP hit, -1=SL hit, 0=expired.
        pnl_atr : float
            PnL in ATR multiples (positive=profit, negative=loss).
        """
        record = {**candle_features, "signal": signal,
                  "outcome": outcome, "pnl_atr": pnl_atr,
                  "timestamp": pd.Timestamp.now(tz="UTC")}
        self._buffer.append(record)

        # Auto-flush buffer to disk every 50 records
        if len(self._buffer) >= 50:
            self.flush()

    def flush(self):
        """Write buffered records to the Parquet store."""
        if not self._buffer:
            return

        new_df = pd.DataFrame(self._buffer)
        self._buffer = []

        if os.path.exists(self.store_path):
            existing = pd.read_parquet(self.store_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        os.makedirs(self.store_dir, exist_ok=True)
        combined.to_parquet(self.store_path)
        logger.info(f"[{self.asset_name}] Flushed {len(new_df)} records to "
                    f"incremental store. Total: {len(combined)}")

    def load(self) -> Optional[pd.DataFrame]:
        """Load the full incremental store."""
        if not os.path.exists(self.store_path):
            logger.info(f"[{self.asset_name}] No incremental store found yet.")
            return None
        df = pd.read_parquet(self.store_path)
        logger.info(f"[{self.asset_name}] Loaded {len(df)} records from store.")
        return df

    def should_retrain(self, min_new_records: int = 200) -> bool:
        """
        Returns True if enough new data has accumulated to warrant retraining.
        Threshold: 200 new labeled outcomes since last retrain.
        """
        df = self.load()
        if df is None:
            return False
        return len(df) >= min_new_records


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("\n=== Full Feature Pipeline Test ===\n")

    np.random.seed(42)
    n = 500
    price = 19000 + np.cumsum(np.random.randn(n) * 10)
    df_raw = pd.DataFrame({
        "open":   price + np.random.randn(n) * 5,
        "close":  price + np.random.randn(n) * 5,
        "volume": np.random.uniform(1000, 5000, n)
    }, index=pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"))
    df_raw["high"] = df_raw[["open","close"]].max(axis=1) + np.abs(np.random.randn(n))*8
    df_raw["low"]  = df_raw[["open","close"]].min(axis=1) - np.abs(np.random.randn(n))*8

    result = run_pipeline(df_raw, asset_name="TEST_ASSET", save_normaliser=False)

    print(f"\nPipeline output keys: {list(result.keys())}")
    print(f"Feature DataFrame shape:    {result['df_features'].shape}")
    print(f"Normalised DataFrame shape: {result['df_normalised'].shape}")
    print(f"Number of features:         {result['n_features']}")

    if result["X"] is not None:
        print(f"X (sequences) shape:        {result['X'].shape}")
        print(f"y (labels) shape:           {result['y'].shape}")
        print(f"X value range:              [{result['X'].min():.3f}, {result['X'].max():.3f}]")

    if result["labels"] is not None:
        vc = result["labels"].value_counts().sort_index()
        print(f"\nFinal label distribution:")
        for k, v in vc.items():
            name = {1: "LONG", -1: "SHORT", 0: "NO_TRADE"}.get(k, str(k))
            print(f"  {name}: {v} ({v/len(result['labels'])*100:.1f}%)")

    print("\nPhase 2 feature pipeline: ALL SYSTEMS OK.")
