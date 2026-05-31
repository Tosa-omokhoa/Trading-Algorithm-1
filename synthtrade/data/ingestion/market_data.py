# =============================================================================
# SynthTrade - Real Market Data Fetcher
# =============================================================================
# Fetches OHLCV data for real indices, forex pairs, and metals using yfinance.
# Handles rate limiting, caching to disk, and a unified DataFrame format that
# matches the output of the Deriv WebSocket fetcher so the feature pipeline
# sees identical structure regardless of asset source.

import os
import time
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional

# Adjust import path when running standalone vs as part of the package
try:
    from config import (
        REAL_ASSETS, YF_INTERVALS, HISTORICAL_CANDLES,
        RAW_DATA_DIR, PRIMARY_TF, BIAS_TF, ENTRY_TF
    )
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from config import (
        REAL_ASSETS, YF_INTERVALS, HISTORICAL_CANDLES,
        RAW_DATA_DIR, PRIMARY_TF, BIAS_TF, ENTRY_TF
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MarketData")


# -----------------------------------------------------------------------------
# PERIOD MAPPING
# Converts a candle count + interval into a yfinance-compatible period string.
# yfinance has strict rules: 1m data only available for last 7 days, etc.
# -----------------------------------------------------------------------------

def _get_yf_period(interval: str, n_candles: int) -> str:
    """
    Calculate a safe yfinance period string given interval and candle count.
    yfinance limits: 1m = max 7d, 5m/15m = max 60d, 1h = max 730d.
    """
    minutes_per_candle = {
        "1m": 1, "2m": 2, "3m": 3, "5m": 5, "15m": 15,
        "30m": 30, "1h": 60, "4h": 240, "1d": 1440
    }
    mins = minutes_per_candle.get(interval, 5)
    total_minutes = mins * n_candles
    total_days = total_minutes / (60 * 6.5)   # approx trading hours per day

    # Respect yfinance hard limits per interval
    hard_limits = {
        "1m": 7, "2m": 60, "3m": 60, "5m": 60,
        "15m": 60, "30m": 60, "1h": 730, "4h": 730, "1d": 3650
    }
    max_days = hard_limits.get(interval, 60)
    days = min(int(total_days) + 1, max_days)

    if days <= 7:
        return "7d"
    elif days <= 30:
        return f"{days}d"
    elif days <= 60:
        return "60d"
    elif days <= 180:
        return "6mo"
    elif days <= 365:
        return "1y"
    elif days <= 730:
        return "2y"
    else:
        return "5y"


# -----------------------------------------------------------------------------
# CORE FETCH FUNCTION
# -----------------------------------------------------------------------------

def fetch_ohlcv(
    asset_name: str,
    interval: str = "5m",
    n_candles: int = 2000,
    use_cache: bool = True,
    cache_max_age_minutes: int = 15
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a real market asset.

    Parameters
    ----------
    asset_name : str
        Internal asset name, e.g. "US100", "XAUUSD", "GBPJPY".
        Must be a key in REAL_ASSETS in config.py.
    interval : str
        Candle interval. One of: 1m, 3m, 5m, 15m, 1h, 4h, 1d.
    n_candles : int
        Approximate number of candles to retrieve.
    use_cache : bool
        If True, loads from disk cache if a recent file exists.
    cache_max_age_minutes : int
        Maximum age (in minutes) of a cached file before re-fetching.

    Returns
    -------
    pd.DataFrame with columns: [open, high, low, close, volume]
        Index is a UTC-aware DatetimeIndex.
        Returns None if fetch fails.
    """
    if asset_name not in REAL_ASSETS:
        logger.error(f"Unknown asset: {asset_name}. Add it to REAL_ASSETS in config.py.")
        return None

    ticker_symbol = REAL_ASSETS[asset_name]

    # --- Check disk cache ---
    cache_path = os.path.join(RAW_DATA_DIR, f"{asset_name}_{interval}.parquet")
    if use_cache and os.path.exists(cache_path):
        age_minutes = (time.time() - os.path.getmtime(cache_path)) / 60
        if age_minutes < cache_max_age_minutes:
            logger.info(f"[{asset_name}] Loading from cache ({age_minutes:.1f} min old).")
            df = pd.read_parquet(cache_path)
            return df

    # --- Fetch from yfinance ---
    period = _get_yf_period(interval, n_candles)
    logger.info(f"[{asset_name}] Fetching {n_candles} candles at {interval} "
                f"(period={period}) from yfinance...")

    try:
        ticker = yf.Ticker(ticker_symbol)
        raw = ticker.history(period=period, interval=interval, auto_adjust=True)

        if raw is None or raw.empty:
            logger.warning(f"[{asset_name}] No data returned from yfinance.")
            return None

        # --- Standardise column names ---
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]

        # --- Ensure UTC-aware index ---
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.index.name = "datetime"

        # --- Drop incomplete final candle (still forming) ---
        df = df.iloc[:-1]

        # --- Drop NaN rows ---
        df.dropna(inplace=True)

        # --- Trim to requested candle count ---
        if len(df) > n_candles:
            df = df.iloc[-n_candles:]

        logger.info(f"[{asset_name}] Retrieved {len(df)} candles. "
                    f"Range: {df.index[0]} to {df.index[-1]}")

        # --- Save to cache ---
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info(f"[{asset_name}] Cached to {cache_path}")

        return df

    except Exception as e:
        logger.error(f"[{asset_name}] Fetch failed: {e}")
        return None


# -----------------------------------------------------------------------------
# MULTI-TIMEFRAME FETCH
# Fetches all three required timeframes (entry, primary, bias) for one asset.
# Returns a dict keyed by timeframe string.
# -----------------------------------------------------------------------------

def fetch_multi_timeframe(
    asset_name: str,
    timeframes: list = None,
    use_cache: bool = True
) -> dict:
    """
    Fetch OHLCV data across multiple timeframes for a single asset.

    Parameters
    ----------
    asset_name : str
        Internal asset name from REAL_ASSETS.
    timeframes : list
        List of interval strings. Defaults to [ENTRY_TF, PRIMARY_TF, BIAS_TF].
    use_cache : bool
        Whether to use disk cache.

    Returns
    -------
    dict: {interval_string: pd.DataFrame}
        Missing timeframes are omitted from the dict (fetch failed silently).
    """
    if timeframes is None:
        timeframes = [ENTRY_TF, PRIMARY_TF, BIAS_TF]

    results = {}
    for tf in timeframes:
        n = HISTORICAL_CANDLES.get(tf, 2000)
        df = fetch_ohlcv(asset_name, interval=tf, n_candles=n, use_cache=use_cache)
        if df is not None:
            results[tf] = df
        time.sleep(0.5)   # Polite rate limiting between yfinance calls

    return results


# -----------------------------------------------------------------------------
# BATCH FETCH ALL REAL ASSETS
# Fetches all active real assets across all required timeframes.
# Returns a nested dict: {asset_name: {timeframe: DataFrame}}
# -----------------------------------------------------------------------------

def fetch_all_real_assets(
    asset_names: list = None,
    timeframes: list = None,
    use_cache: bool = True
) -> dict:
    """
    Batch fetch all real market assets across all required timeframes.

    Parameters
    ----------
    asset_names : list
        Subset of REAL_ASSETS keys to fetch. Defaults to all.
    timeframes : list
        Timeframes to fetch for each asset.
    use_cache : bool
        Whether to use disk cache.

    Returns
    -------
    dict: {asset_name: {timeframe: pd.DataFrame}}
    """
    if asset_names is None:
        asset_names = list(REAL_ASSETS.keys())
    if timeframes is None:
        timeframes = [ENTRY_TF, PRIMARY_TF, BIAS_TF]

    all_data = {}
    for asset in asset_names:
        logger.info(f"--- Fetching all timeframes for {asset} ---")
        all_data[asset] = fetch_multi_timeframe(asset, timeframes, use_cache)
        time.sleep(1.0)   # Rate limiting between assets

    return all_data


# -----------------------------------------------------------------------------
# DATA QUALITY CHECK
# Simple validation to catch corrupt or insufficient data before modelling.
# -----------------------------------------------------------------------------

def validate_dataframe(df: pd.DataFrame, asset_name: str, min_rows: int = 100) -> bool:
    """
    Validate a fetched OHLCV DataFrame for basic quality requirements.

    Checks:
    - Minimum row count
    - No all-NaN columns
    - OHLCV columns present
    - High >= Low (no corrupt candles)
    - No duplicate index entries
    """
    required_cols = {"open", "high", "low", "close", "volume"}

    if df is None or df.empty:
        logger.warning(f"[{asset_name}] DataFrame is None or empty.")
        return False

    if len(df) < min_rows:
        logger.warning(f"[{asset_name}] Insufficient rows: {len(df)} < {min_rows}.")
        return False

    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        logger.warning(f"[{asset_name}] Missing columns: {missing}")
        return False

    if df[list(required_cols)].isna().all().any():
        logger.warning(f"[{asset_name}] One or more columns are all NaN.")
        return False

    corrupt_candles = (df["high"] < df["low"]).sum()
    if corrupt_candles > 0:
        logger.warning(f"[{asset_name}] {corrupt_candles} candles with High < Low detected.")
        return False

    if df.index.duplicated().any():
        logger.warning(f"[{asset_name}] Duplicate timestamps detected. Deduplicating.")
        # Not a failure, just a warning. Caller should deduplicate.

    logger.info(f"[{asset_name}] Data validation passed. {len(df)} rows.")
    return True


# -----------------------------------------------------------------------------
# QUICK TEST (run this file directly to verify everything works)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== SynthTrade: Real Market Data Fetcher Test ===\n")

    test_assets = ["US100", "XAUUSD", "USDJPY"]

    for asset in test_assets:
        print(f"\nTesting {asset}...")
        df = fetch_ohlcv(asset, interval="5m", n_candles=500, use_cache=False)
        if df is not None:
            valid = validate_dataframe(df, asset)
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            print(f"  Valid: {valid}")
            print(df.tail(3))
        else:
            print(f"  FAILED to fetch {asset}")
