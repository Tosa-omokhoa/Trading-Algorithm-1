# =============================================================================
# SynthTrade - Heikin Ashi Transformation
# =============================================================================
# Converts standard OHLCV data into Heikin Ashi candles.
#
# Heikin Ashi formulas:
#   HA_Close  = (Open + High + Low + Close) / 4
#   HA_Open   = (prev_HA_Open + prev_HA_Close) / 2   [first bar: (Open + Close) / 2]
#   HA_High   = max(High, HA_Open, HA_Close)
#   HA_Low    = min(Low,  HA_Open, HA_Close)
#
# Why Heikin Ashi for this system:
#   - Smooths out noise in volatile synthetic indices (VIX75 especially)
#   - Makes trend direction visually and mathematically cleaner
#   - Consecutive same-coloured HA candles are a strong trend continuation signal
#   - Works particularly well with EMA and SAR confluence strategies
#
# The original OHLCV columns are preserved alongside HA columns so the
# feature pipeline has access to both raw and smoothed price data.

import numpy as np
import pandas as pd
from typing import Optional


def compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Heikin Ashi candles from a standard OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close.
        Volume is passed through unchanged.

    Returns
    -------
    pd.DataFrame
        Original columns preserved, plus:
        - ha_open, ha_high, ha_low, ha_close
        - ha_body       : HA close - HA open (positive = bullish, negative = bearish)
        - ha_body_abs   : Absolute body size (magnitude of move)
        - ha_upper_wick : HA high - max(HA open, HA close)
        - ha_lower_wick : min(HA open, HA close) - HA low
        - ha_direction  : 1 (bullish), -1 (bearish), 0 (doji)
        - ha_consecutive: Count of consecutive same-direction HA candles
        - ha_no_upper   : 1 if upper wick is near zero (strong bullish signal)
        - ha_no_lower   : 1 if lower wick is near zero (strong bearish signal)
        - ha_doji       : 1 if body is very small relative to total range (indecision)
    """
    df = df.copy()

    n = len(df)
    ha_close = np.zeros(n)
    ha_open  = np.zeros(n)
    ha_high  = np.zeros(n)
    ha_low   = np.zeros(n)

    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    # --- Compute HA values row by row ---
    # First candle: HA open is average of open and close
    ha_close[0] = (opens[0] + highs[0] + lows[0] + closes[0]) / 4.0
    ha_open[0]  = (opens[0] + closes[0]) / 2.0
    ha_high[0]  = max(highs[0], ha_open[0], ha_close[0])
    ha_low[0]   = min(lows[0],  ha_open[0], ha_close[0])

    for i in range(1, n):
        ha_close[i] = (opens[i] + highs[i] + lows[i] + closes[i]) / 4.0
        ha_open[i]  = (ha_open[i-1] + ha_close[i-1]) / 2.0
        ha_high[i]  = max(highs[i], ha_open[i], ha_close[i])
        ha_low[i]   = min(lows[i],  ha_open[i], ha_close[i])

    df["ha_open"]  = ha_open
    df["ha_high"]  = ha_high
    df["ha_low"]   = ha_low
    df["ha_close"] = ha_close

    # --- Derived HA features ---

    # Body: signed (positive = bullish candle, negative = bearish)
    df["ha_body"]     = df["ha_close"] - df["ha_open"]
    df["ha_body_abs"] = df["ha_body"].abs()

    # Wicks
    df["ha_upper_wick"] = df["ha_high"] - df[["ha_open", "ha_close"]].max(axis=1)
    df["ha_lower_wick"] = df[["ha_open", "ha_close"]].min(axis=1) - df["ha_low"]

    # Direction: 1 = bullish, -1 = bearish, 0 = doji
    df["ha_direction"] = np.sign(df["ha_body"]).astype(int)

    # Consecutive same-direction candle count (trend strength indicator)
    consecutive = np.ones(n, dtype=int)
    for i in range(1, n):
        if df["ha_direction"].iloc[i] == df["ha_direction"].iloc[i-1] and \
           df["ha_direction"].iloc[i] != 0:
            consecutive[i] = consecutive[i-1] + 1
        else:
            consecutive[i] = 1
    df["ha_consecutive"] = consecutive

    # Total candle range (high to low)
    total_range = df["ha_high"] - df["ha_low"]
    total_range = total_range.replace(0, np.nan)   # Avoid divide by zero

    # Wick ratio thresholds: a wick is "near zero" if it is < 10% of total range
    wick_threshold = 0.10
    df["ha_no_upper"] = (
        (df["ha_upper_wick"] / total_range) < wick_threshold
    ).astype(int)

    df["ha_no_lower"] = (
        (df["ha_lower_wick"] / total_range) < wick_threshold
    ).astype(int)

    # Doji: body is < 5% of total range (indecision / potential reversal)
    df["ha_doji"] = (
        (df["ha_body_abs"] / total_range) < 0.05
    ).astype(int)

    # Fill any NaN introduced by division
    df.fillna(0, inplace=True)

    return df


def get_ha_signal(df: pd.DataFrame) -> pd.Series:
    """
    Generate a simple Heikin Ashi trend signal based on candle sequence.

    Signal logic:
      - LONG  (1):  2+ consecutive bullish HA candles with no lower wick
      - SHORT (-1): 2+ consecutive bearish HA candles with no upper wick
      - NONE  (0):  Everything else

    This is used as a confirmation component in the full signal pipeline,
    not as a standalone signal.

    Returns
    -------
    pd.Series of int (1, -1, or 0), same index as df.
    """
    signal = pd.Series(0, index=df.index)

    long_cond  = (df["ha_direction"] == 1) & \
                 (df["ha_consecutive"] >= 2) & \
                 (df["ha_no_lower"] == 1)

    short_cond = (df["ha_direction"] == -1) & \
                 (df["ha_consecutive"] >= 2) & \
                 (df["ha_no_upper"] == 1)

    signal[long_cond]  = 1
    signal[short_cond] = -1

    return signal


# -----------------------------------------------------------------------------
# QUICK TEST
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    print("\n=== Heikin Ashi Transformation Test ===\n")

    # Generate synthetic price data that mimics a trending market
    np.random.seed(42)
    n = 200
    price = 19000 + np.cumsum(np.random.randn(n) * 10)
    df_test = pd.DataFrame({
        "open":   price + np.random.randn(n) * 5,
        "high":   price + np.abs(np.random.randn(n)) * 15,
        "low":    price - np.abs(np.random.randn(n)) * 15,
        "close":  price + np.random.randn(n) * 5,
        "volume": np.random.uniform(1000, 5000, n)
    }, index=pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"))

    # Ensure high >= low
    df_test["high"] = df_test[["open", "high", "close"]].max(axis=1) + 1
    df_test["low"]  = df_test[["open", "low", "close"]].min(axis=1) - 1

    result = compute_heikin_ashi(df_test)

    ha_cols = ["ha_open", "ha_high", "ha_low", "ha_close",
               "ha_body", "ha_direction", "ha_consecutive",
               "ha_no_upper", "ha_no_lower", "ha_doji"]

    print(f"Output shape: {result.shape}")
    print(f"\nHA columns added: {ha_cols}")
    print(f"\nSample output (last 5 rows):")
    print(result[ha_cols].tail(5).to_string())

    signal = get_ha_signal(result)
    long_signals  = (signal == 1).sum()
    short_signals = (signal == -1).sum()
    print(f"\nHA signals generated: {long_signals} LONG, {short_signals} SHORT "
          f"out of {n} candles")
    print(f"Signal rate: {(long_signals + short_signals) / n * 100:.1f}%")
