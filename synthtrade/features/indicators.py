# =============================================================================
# SynthTrade - Technical Indicators
# =============================================================================
# Computes the full indicator stack on Heikin Ashi OHLCV data.
#
# Indicators computed:
#   - EMA 9, 21, 50         (trend alignment across three horizons)
#   - Parabolic SAR          (trend continuation / reversal detection)
#   - ATR 14                 (volatility measurement, SL/TP sizing)
#   - RSI 14                 (momentum filter)
#   - Bollinger Bands 20,2   (regime detection: trending vs ranging)
#   - Volume Ratio           (current volume vs 20-period average)
#
# All indicators are computed on HA candles (ha_open, ha_high, ha_low,
# ha_close) rather than raw OHLCV. This is intentional: it keeps the
# smoothing effect of Heikin Ashi consistent across both price and
# indicator inputs.
#
# Additional derived features are computed after indicators to maximise
# the information density of each input vector to the CNN-LSTM.

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Optional

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from config import (
        EMA_PERIODS, RSI_PERIOD, ATR_PERIOD,
        BB_PERIOD, BB_STD, SAR_ACCELERATION, SAR_MAXIMUM,
        VOLUME_MA_PERIOD
    )
except ImportError:
    # Fallback defaults if run standalone
    EMA_PERIODS      = [9, 21, 50]
    RSI_PERIOD       = 14
    ATR_PERIOD       = 14
    BB_PERIOD        = 20
    BB_STD           = 2.0
    SAR_ACCELERATION = 0.02
    SAR_MAXIMUM      = 0.2
    VOLUME_MA_PERIOD = 20


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full indicator stack on a Heikin Ashi OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain HA columns: ha_open, ha_high, ha_low, ha_close.
        Must also contain: volume.
        Produced by features/heikin_ashi.py.

    Returns
    -------
    pd.DataFrame
        Original columns preserved, plus all indicator columns below.

    Added columns
    -------------
    Trend:
        ema_9, ema_21, ema_50
        ema_9_slope, ema_21_slope         (rate of change per candle)
        ema_9_21_cross                    (1=bullish cross, -1=bearish, 0=none)
        price_to_ema9, price_to_ema21     (distance normalised by ATR)
        price_to_ema50
        ema_alignment                     (1=all aligned bullish, -1=bearish, 0=mixed)

    Momentum:
        rsi
        rsi_zone    (1=overbought >70, -1=oversold <30, 0=neutral)
        rsi_slope   (rate of change of RSI)

    Volatility:
        atr
        bb_upper, bb_mid, bb_lower
        bb_width                          (band width normalised by mid)
        bb_pct                            (price position within bands, 0-1)
        bb_regime   (1=trending/wide, 0=ranging/narrow, threshold=0.03)

    Trend continuation:
        sar                               (SAR value)
        sar_direction   (1=price above SAR bullish, -1=price below SAR bearish)
        sar_flip        (1=just flipped bullish, -1=just flipped bearish, 0=none)
        sar_distance    (distance from price to SAR, normalised by ATR)

    Volume:
        volume_ma
        volume_ratio    (current volume / volume_ma)
        volume_spike    (1 if volume_ratio > 2.0, else 0)

    Composite:
        trend_score     (sum of aligned signals: EMA + SAR + HA direction)
                        Range: -3 to +3. Used as a quick confluence gauge.
    """
    df = df.copy()

    # Use HA columns as the price source for all indicators
    ha_high  = df["ha_high"]
    ha_low   = df["ha_low"]
    ha_close = df["ha_close"]
    ha_open  = df["ha_open"]
    volume   = df["volume"]

    # -------------------------------------------------------------------------
    # EMA STACK
    # -------------------------------------------------------------------------
    for period in EMA_PERIODS:
        df[f"ema_{period}"] = ta.ema(ha_close, length=period)

    # EMA slopes: rate of change over last 3 candles (normalised by price)
    df["ema_9_slope"]  = df["ema_9"].diff(3)  / df["ema_9"]  * 100
    df["ema_21_slope"] = df["ema_21"].diff(3) / df["ema_21"] * 100

    # EMA 9/21 crossover signal
    prev_diff = (df["ema_9"] - df["ema_21"]).shift(1)
    curr_diff =  df["ema_9"] - df["ema_21"]
    df["ema_9_21_cross"] = 0
    df.loc[(prev_diff <= 0) & (curr_diff > 0), "ema_9_21_cross"] = 1   # Bullish cross
    df.loc[(prev_diff >= 0) & (curr_diff < 0), "ema_9_21_cross"] = -1  # Bearish cross

    # EMA alignment: all three EMAs in order (9 > 21 > 50 = bullish, reverse = bearish)
    df["ema_alignment"] = 0
    df.loc[
        (df["ema_9"] > df["ema_21"]) & (df["ema_21"] > df["ema_50"]),
        "ema_alignment"
    ] = 1
    df.loc[
        (df["ema_9"] < df["ema_21"]) & (df["ema_21"] < df["ema_50"]),
        "ema_alignment"
    ] = -1

    # -------------------------------------------------------------------------
    # ATR (computed before price-to-EMA distances since those need ATR normalisation)
    # -------------------------------------------------------------------------
    atr_series = ta.atr(ha_high, ha_low, ha_close, length=ATR_PERIOD)
    df["atr"] = atr_series

    # Safe ATR for division (avoid zero division on synthetics / flat periods)
    safe_atr = df["atr"].replace(0, np.nan).ffill().fillna(1.0)

    # Price distances to each EMA, normalised by ATR (scale-invariant)
    df["price_to_ema9"]  = (ha_close - df["ema_9"])  / safe_atr
    df["price_to_ema21"] = (ha_close - df["ema_21"]) / safe_atr
    df["price_to_ema50"] = (ha_close - df["ema_50"]) / safe_atr

    # -------------------------------------------------------------------------
    # RSI
    # -------------------------------------------------------------------------
    df["rsi"] = ta.rsi(ha_close, length=RSI_PERIOD)

    df["rsi_zone"] = 0
    df.loc[df["rsi"] >= 70, "rsi_zone"] = 1    # Overbought
    df.loc[df["rsi"] <= 30, "rsi_zone"] = -1   # Oversold

    df["rsi_slope"] = df["rsi"].diff(3)         # RSI momentum

    # -------------------------------------------------------------------------
    # BOLLINGER BANDS
    # -------------------------------------------------------------------------
    bb = ta.bbands(ha_close, length=BB_PERIOD, std=BB_STD)

    if bb is not None and not bb.empty:
        # pandas-ta names columns: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        col_map = {
            "lower": [c for c in bb.columns if c.startswith("BBL")],
            "mid":   [c for c in bb.columns if c.startswith("BBM")],
            "upper": [c for c in bb.columns if c.startswith("BBU")],
            "width": [c for c in bb.columns if c.startswith("BBB")],
            "pct":   [c for c in bb.columns if c.startswith("BBP")],
        }
        df["bb_lower"] = bb[col_map["lower"][0]].values if col_map["lower"] else np.nan
        df["bb_mid"]   = bb[col_map["mid"][0]].values   if col_map["mid"]   else np.nan
        df["bb_upper"] = bb[col_map["upper"][0]].values if col_map["upper"] else np.nan
        df["bb_width"] = bb[col_map["width"][0]].values if col_map["width"] else np.nan
        df["bb_pct"]   = bb[col_map["pct"][0]].values   if col_map["pct"]   else np.nan
    else:
        df["bb_lower"] = np.nan
        df["bb_mid"]   = np.nan
        df["bb_upper"] = np.nan
        df["bb_width"] = np.nan
        df["bb_pct"]   = np.nan

    # Regime: wide bands = trending, narrow bands = ranging
    # bb_width from pandas-ta is already (upper-lower)/mid * 100
    df["bb_regime"] = (df["bb_width"] > 3.0).astype(int)

    # -------------------------------------------------------------------------
    # PARABOLIC SAR
    # -------------------------------------------------------------------------
    sar_result = ta.psar(ha_high, ha_low, ha_close,
                         af0=SAR_ACCELERATION,
                         af=SAR_ACCELERATION,
                         max_af=SAR_MAXIMUM)

    if sar_result is not None and not sar_result.empty:
        # pandas-ta psar returns long and short SAR series separately
        # PSARl = SAR when trending up (price above SAR)
        # PSARs = SAR when trending down (price below SAR)
        long_col  = [c for c in sar_result.columns if "PSARl" in c]
        short_col = [c for c in sar_result.columns if "PSARs" in c]

        if long_col and short_col:
            sar_long  = sar_result[long_col[0]]
            sar_short = sar_result[short_col[0]]

            # Unified SAR: use long SAR when in uptrend, short SAR when in downtrend
            df["sar"] = np.where(sar_long.notna(), sar_long, sar_short)
            df["sar"] = pd.Series(df["sar"], index=df.index).ffill()

            # Direction: 1 if price above SAR (uptrend), -1 if below (downtrend)
            df["sar_direction"] = np.where(ha_close > df["sar"], 1, -1)

            # SAR flip: detect direction change
            prev_dir = df["sar_direction"].shift(1)
            df["sar_flip"] = 0
            df.loc[(prev_dir == -1) & (df["sar_direction"] == 1),  "sar_flip"] = 1
            df.loc[(prev_dir == 1)  & (df["sar_direction"] == -1), "sar_flip"] = -1

            # Distance from close to SAR, normalised by ATR
            df["sar_distance"] = (ha_close - df["sar"]) / safe_atr
        else:
            df["sar"] = np.nan
            df["sar_direction"] = 0
            df["sar_flip"] = 0
            df["sar_distance"] = 0.0
    else:
        df["sar"] = np.nan
        df["sar_direction"] = 0
        df["sar_flip"] = 0
        df["sar_distance"] = 0.0

    # -------------------------------------------------------------------------
    # VOLUME
    # -------------------------------------------------------------------------
    df["volume_ma"]    = ta.sma(volume, length=VOLUME_MA_PERIOD)
    safe_vol_ma        = df["volume_ma"].replace(0, np.nan).fillna(1.0)
    df["volume_ratio"] = volume / safe_vol_ma
    df["volume_spike"] = (df["volume_ratio"] > 2.0).astype(int)

    # -------------------------------------------------------------------------
    # COMPOSITE TREND SCORE
    # Combines EMA alignment, SAR direction, and HA direction into one value.
    # Range: -3 (strongly bearish) to +3 (strongly bullish)
    # This is a fast signal-quality pre-filter for the dashboard heatmap.
    # -------------------------------------------------------------------------
    ha_dir = df.get("ha_direction", pd.Series(0, index=df.index))

    df["trend_score"] = (
        df["ema_alignment"].fillna(0) +
        df["sar_direction"].fillna(0) +
        ha_dir.fillna(0)
    )

    # -------------------------------------------------------------------------
    # FINAL CLEANUP
    # Drop rows with NaN introduced by indicator warm-up periods.
    # The warm-up period is determined by the longest indicator (EMA 50).
    # -------------------------------------------------------------------------
    warmup = max(EMA_PERIODS) + ATR_PERIOD + BB_PERIOD + 5
    df = df.iloc[warmup:].copy()
    df.reset_index(drop=False, inplace=True)

    # Restore datetime as index if it was reset
    if "datetime" in df.columns:
        df.set_index("datetime", inplace=True)
    elif "index" in df.columns:
        df.set_index("index", inplace=True)

    # Final NaN fill for any stragglers
    df.fillna(0, inplace=True)

    return df


def get_indicator_feature_columns() -> list:
    """
    Returns the ordered list of feature column names that the CNN-LSTM
    model expects as input. Call this to get the exact feature vector spec.
    """
    return [
        # Heikin Ashi price features
        "ha_open", "ha_high", "ha_low", "ha_close",
        "ha_body", "ha_body_abs", "ha_upper_wick", "ha_lower_wick",
        "ha_direction", "ha_consecutive", "ha_no_upper", "ha_no_lower", "ha_doji",

        # EMA features
        "ema_9", "ema_21", "ema_50",
        "ema_9_slope", "ema_21_slope",
        "ema_9_21_cross", "ema_alignment",
        "price_to_ema9", "price_to_ema21", "price_to_ema50",

        # ATR
        "atr",

        # RSI
        "rsi", "rsi_zone", "rsi_slope",

        # Bollinger Bands
        "bb_upper", "bb_mid", "bb_lower",
        "bb_width", "bb_pct", "bb_regime",

        # Parabolic SAR
        "sar", "sar_direction", "sar_flip", "sar_distance",

        # Volume
        "volume_ratio", "volume_spike",

        # Composite
        "trend_score",
    ]


# -----------------------------------------------------------------------------
# QUICK TEST
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== Indicator Stack Test ===\n")
    from heikin_ashi import compute_heikin_ashi

    np.random.seed(42)
    n = 300
    price = 19000 + np.cumsum(np.random.randn(n) * 10)
    df_test = pd.DataFrame({
        "open":   price + np.random.randn(n) * 5,
        "close":  price + np.random.randn(n) * 5,
        "volume": np.random.uniform(1000, 5000, n)
    }, index=pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"))
    df_test["high"] = df_test[["open", "close"]].max(axis=1) + np.abs(np.random.randn(n)) * 8
    df_test["low"]  = df_test[["open", "close"]].min(axis=1) - np.abs(np.random.randn(n)) * 8

    df_ha   = compute_heikin_ashi(df_test)
    df_full = compute_indicators(df_ha)

    feature_cols = get_indicator_feature_columns()

    print(f"Output shape: {df_full.shape}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"\nSample (last 3 rows, key features):")
    key_cols = ["ha_close", "ema_9", "ema_21", "ema_50", "ema_alignment",
                "rsi", "sar_direction", "sar_flip", "trend_score", "volume_ratio"]
    print(df_full[key_cols].tail(3).to_string())

    print(f"\nTrend score distribution:")
    print(df_full["trend_score"].value_counts().sort_index())

    missing = [c for c in feature_cols if c not in df_full.columns]
    if missing:
        print(f"\nWARNING: Missing feature columns: {missing}")
    else:
        print(f"\nAll {len(feature_cols)} feature columns present. Indicator stack OK.")
