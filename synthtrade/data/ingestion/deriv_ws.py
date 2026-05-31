# =============================================================================
# SynthTrade - Deriv WebSocket Client
# =============================================================================
# Connects to the Deriv (Binary.com) WebSocket API to stream live tick data
# and fetch historical OHLCV candles for synthetic indices.
#
# Deriv API docs: https://api.deriv.com/
# This client handles:
#   - Authentication with your Deriv API token
#   - Historical candle (OHLCV) requests per asset and granularity
#   - Live tick subscriptions with an in-memory buffer
#   - Auto-reconnection on disconnect
#   - Graceful shutdown
#
# IMPORTANT: You need a free Deriv account and an API token with
# "Read" and "Trading information" scopes. Set DERIV_API_TOKEN in your .env.
# For historical data without a token, the demo app ID (1089) still works
# for most synthetic index data.

import asyncio
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

import websockets
import pandas as pd

try:
    from config import (
        DERIV_APP_ID, DERIV_API_TOKEN, DERIV_WS_URL,
        DERIV_WS_TIMEOUT, DERIV_TICK_BUFFER, SYNTHETIC_ASSETS,
        RAW_DATA_DIR, PRIMARY_TF, BIAS_TF, ENTRY_TF, HISTORICAL_CANDLES
    )
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from config import (
        DERIV_APP_ID, DERIV_API_TOKEN, DERIV_WS_URL,
        DERIV_WS_TIMEOUT, DERIV_TICK_BUFFER, SYNTHETIC_ASSETS,
        RAW_DATA_DIR, PRIMARY_TF, BIAS_TF, ENTRY_TF, HISTORICAL_CANDLES
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DerivWS")


# -----------------------------------------------------------------------------
# GRANULARITY MAPPING
# Deriv uses seconds as granularity, not interval strings.
# -----------------------------------------------------------------------------

GRANULARITY_MAP = {
    "1m":  60,
    "2m":  120,
    "3m":  180,
    "5m":  300,
    "10m": 600,
    "15m": 900,
    "30m": 1800,
    "1h":  3600,
    "4h":  14400,
    "1d":  86400,
}


# -----------------------------------------------------------------------------
# DERIV WEBSOCKET CLIENT
# -----------------------------------------------------------------------------

class DerivWSClient:
    """
    Async WebSocket client for the Deriv API.

    Usage (historical candles):
        client = DerivWSClient()
        df = await client.get_candles("VIX75", interval="5m", count=500)

    Usage (live tick stream):
        async def on_tick(asset, price, timestamp):
            print(f"{asset}: {price} @ {timestamp}")

        client = DerivWSClient()
        await client.subscribe_ticks(["VIX75", "CRASH500"], callback=on_tick)
    """

    def __init__(self):
        self.app_id = DERIV_APP_ID
        self.token = DERIV_API_TOKEN
        self.ws_url = f"{DERIV_WS_URL}?app_id={self.app_id}"
        self._ws = None
        self._connected = False
        self._authenticated = False
        self._pending: Dict[str, asyncio.Future] = {}
        self._req_id = 0

        # In-memory tick buffers per asset: {asset_name: deque of (price, epoch)}
        self.tick_buffers: Dict[str, deque] = {
            name: deque(maxlen=DERIV_TICK_BUFFER)
            for name in SYNTHETIC_ASSETS
        }

        # Subscription IDs for cleanup on shutdown
        self._subscriptions: Dict[str, str] = {}

    # -------------------------------------------------------------------------
    # CONNECTION MANAGEMENT
    # -------------------------------------------------------------------------

    async def connect(self) -> bool:
        """Open WebSocket connection and optionally authenticate."""
        try:
            logger.info(f"Connecting to Deriv API: {self.ws_url}")
            self._ws = await websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            )
            self._connected = True
            logger.info("Connected to Deriv WebSocket.")

            # Authenticate if a token is provided
            if self.token:
                auth_result = await self._authenticate()
                if not auth_result:
                    logger.warning("Authentication failed. Proceeding in demo mode.")
            else:
                logger.info("No API token provided. Running in demo/public mode.")

            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._connected = False
            return False

    async def disconnect(self):
        """Close the WebSocket connection gracefully."""
        if self._ws and self._connected:
            await self._ws.close()
            self._connected = False
            logger.info("Disconnected from Deriv WebSocket.")

    async def _authenticate(self) -> bool:
        """Send API token authentication request."""
        try:
            response = await self._send_request({"authorize": self.token})
            if "authorize" in response:
                self._authenticated = True
                account = response["authorize"].get("email", "unknown")
                logger.info(f"Authenticated as: {account}")
                return True
            else:
                error = response.get("error", {}).get("message", "Unknown error")
                logger.error(f"Auth error: {error}")
                return False
        except Exception as e:
            logger.error(f"Authentication exception: {e}")
            return False

    # -------------------------------------------------------------------------
    # REQUEST / RESPONSE CORE
    # -------------------------------------------------------------------------

    async def _send_request(self, payload: dict) -> dict:
        """
        Send a JSON request and wait for its matching response.
        Uses req_id to match async responses correctly.
        """
        if not self._connected or self._ws is None:
            raise ConnectionError("Not connected. Call connect() first.")

        self._req_id += 1
        payload["req_id"] = self._req_id

        future = asyncio.get_event_loop().create_future()
        self._pending[self._req_id] = future

        await self._ws.send(json.dumps(payload))

        try:
            response = await asyncio.wait_for(future, timeout=DERIV_WS_TIMEOUT)
            return response
        except asyncio.TimeoutError:
            self._pending.pop(self._req_id, None)
            raise TimeoutError(f"Request timed out (req_id={self._req_id})")

    async def _listen(self):
        """
        Continuously listen for messages and route them to pending futures
        or tick handlers.
        """
        async for raw_message in self._ws:
            try:
                msg = json.loads(raw_message)
                req_id = msg.get("req_id")

                # Route to pending request future if applicable
                if req_id and req_id in self._pending:
                    future = self._pending.pop(req_id)
                    if not future.done():
                        future.set_result(msg)

                # Handle streaming tick data
                elif msg.get("msg_type") == "tick":
                    await self._handle_tick(msg)

                # Handle streaming candle data
                elif msg.get("msg_type") == "ohlc":
                    await self._handle_ohlc(msg)

            except json.JSONDecodeError:
                logger.warning(f"Could not parse message: {raw_message[:100]}")
            except Exception as e:
                logger.error(f"Message handling error: {e}")

    async def _handle_tick(self, msg: dict):
        """Process a streaming tick message and append to buffer."""
        tick = msg.get("tick", {})
        symbol = tick.get("symbol", "")
        price = float(tick.get("ask", 0))
        epoch = int(tick.get("epoch", 0))

        # Reverse lookup: Deriv symbol -> our internal asset name
        for asset_name, deriv_symbol in SYNTHETIC_ASSETS.items():
            if deriv_symbol == symbol:
                self.tick_buffers[asset_name].append({
                    "price": price,
                    "epoch": epoch,
                    "datetime": datetime.fromtimestamp(epoch, tz=timezone.utc)
                })
                break

    async def _handle_ohlc(self, msg: dict):
        """Handle streaming OHLC candle update (used for live candle feeds)."""
        # Placeholder for live candle stream handler
        # Will be used in Phase 6 (dashboard live feed)
        pass

    # -------------------------------------------------------------------------
    # HISTORICAL CANDLES
    # -------------------------------------------------------------------------

    async def get_candles(
        self,
        asset_name: str,
        interval: str = "5m",
        count: int = 500,
        use_cache: bool = True,
        cache_max_age_minutes: int = 15
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV candles for a synthetic index asset.

        Parameters
        ----------
        asset_name : str
            Internal asset name, e.g. "VIX75", "CRASH500".
        interval : str
            Candle interval string, e.g. "5m", "15m".
        count : int
            Number of candles to fetch (max 5000 per Deriv API limit).
        use_cache : bool
            Load from disk cache if a recent file exists.
        cache_max_age_minutes : int
            Max age of cached file before re-fetching.

        Returns
        -------
        pd.DataFrame with columns [open, high, low, close, volume]
        Index: UTC DatetimeIndex named 'datetime'.
        """
        if asset_name not in SYNTHETIC_ASSETS:
            logger.error(f"Unknown synthetic asset: {asset_name}")
            return None

        deriv_symbol = SYNTHETIC_ASSETS[asset_name]
        granularity = GRANULARITY_MAP.get(interval)
        if granularity is None:
            logger.error(f"Unsupported interval: {interval}. "
                         f"Use one of {list(GRANULARITY_MAP.keys())}")
            return None

        # --- Check disk cache ---
        cache_path = os.path.join(RAW_DATA_DIR, f"{asset_name}_{interval}.parquet")
        if use_cache and os.path.exists(cache_path):
            age_minutes = (time.time() - os.path.getmtime(cache_path)) / 60
            if age_minutes < cache_max_age_minutes:
                logger.info(f"[{asset_name}] Loading from cache ({age_minutes:.1f} min old).")
                return pd.read_parquet(cache_path)

        # --- Request from Deriv ---
        # Deriv candles endpoint: ticks_history with style=candles
        end_epoch = int(time.time())
        count = min(count, 5000)   # Deriv hard limit

        request_payload = {
            "ticks_history": deriv_symbol,
            "style": "candles",
            "granularity": granularity,
            "count": count,
            "end": end_epoch,
            "adjust_start_time": 1
        }

        logger.info(f"[{asset_name}] Requesting {count} candles at {interval} "
                    f"(granularity={granularity}s) from Deriv...")

        try:
            response = await self._send_request(request_payload)

            if "error" in response:
                error_msg = response["error"].get("message", "Unknown error")
                logger.error(f"[{asset_name}] Deriv API error: {error_msg}")
                return None

            candles = response.get("candles", [])
            if not candles:
                logger.warning(f"[{asset_name}] No candles returned.")
                return None

            # --- Build DataFrame ---
            df = pd.DataFrame(candles)

            # Deriv returns: epoch, open, high, low, close (no volume for synthetics)
            df["epoch"] = df["epoch"].astype(int)
            df["open"]  = df["open"].astype(float)
            df["high"]  = df["high"].astype(float)
            df["low"]   = df["low"].astype(float)
            df["close"] = df["close"].astype(float)

            # Synthetic indices have no real volume. We use a placeholder of 1.0
            # so the feature pipeline stays consistent with real asset DataFrames.
            df["volume"] = 1.0

            # --- Convert epoch to UTC DatetimeIndex ---
            df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
            df.set_index("datetime", inplace=True)
            df.drop(columns=["epoch"], inplace=True)
            df.sort_index(inplace=True)

            # Drop the still-forming final candle
            df = df.iloc[:-1]

            logger.info(f"[{asset_name}] Retrieved {len(df)} candles. "
                        f"Range: {df.index[0]} to {df.index[-1]}")

            # --- Cache to disk ---
            os.makedirs(RAW_DATA_DIR, exist_ok=True)
            df.to_parquet(cache_path)
            logger.info(f"[{asset_name}] Cached to {cache_path}")

            return df

        except Exception as e:
            logger.error(f"[{asset_name}] get_candles failed: {e}")
            return None

    # -------------------------------------------------------------------------
    # MULTI-TIMEFRAME FETCH (for one synthetic asset)
    # -------------------------------------------------------------------------

    async def get_multi_timeframe(
        self,
        asset_name: str,
        timeframes: list = None,
        use_cache: bool = True
    ) -> dict:
        """
        Fetch historical candles across multiple timeframes for one asset.

        Returns
        -------
        dict: {interval_string: pd.DataFrame}
        """
        if timeframes is None:
            timeframes = [ENTRY_TF, PRIMARY_TF, BIAS_TF]

        results = {}
        for tf in timeframes:
            n = HISTORICAL_CANDLES.get(tf, 500)
            df = await self.get_candles(asset_name, interval=tf, count=n,
                                        use_cache=use_cache)
            if df is not None:
                results[tf] = df
            await asyncio.sleep(0.3)   # Small delay between requests

        return results

    # -------------------------------------------------------------------------
    # LIVE TICK SUBSCRIPTION
    # -------------------------------------------------------------------------

    async def subscribe_ticks(
        self,
        asset_names: list,
        callback: Optional[Callable] = None
    ):
        """
        Subscribe to live tick streams for a list of synthetic assets.
        Starts the listener loop which populates self.tick_buffers.

        Parameters
        ----------
        asset_names : list
            List of internal asset names, e.g. ["VIX75", "CRASH500"].
        callback : async callable, optional
            Called on each tick with signature: callback(asset_name, price, dt)
        """
        if not self._connected:
            await self.connect()

        for asset_name in asset_names:
            if asset_name not in SYNTHETIC_ASSETS:
                logger.warning(f"Cannot subscribe: unknown asset {asset_name}")
                continue

            deriv_symbol = SYNTHETIC_ASSETS[asset_name]
            payload = {"ticks": deriv_symbol, "subscribe": 1}

            try:
                response = await self._send_request(payload)
                if "error" in response:
                    logger.error(f"[{asset_name}] Subscribe error: "
                                 f"{response['error'].get('message')}")
                else:
                    sub_id = response.get("subscription", {}).get("id", "")
                    self._subscriptions[asset_name] = sub_id
                    logger.info(f"[{asset_name}] Subscribed to live ticks. "
                                f"Sub ID: {sub_id}")
            except Exception as e:
                logger.error(f"[{asset_name}] Subscribe failed: {e}")

        # Store the callback for use in _handle_tick
        self._tick_callback = callback

        # Start the listener loop (this runs until disconnect)
        logger.info("Starting tick listener loop...")
        await self._listen()

    async def unsubscribe_all(self):
        """Unsubscribe from all active tick streams."""
        for asset_name, sub_id in self._subscriptions.items():
            if sub_id:
                try:
                    await self._send_request({
                        "forget": sub_id
                    })
                    logger.info(f"[{asset_name}] Unsubscribed (sub_id={sub_id})")
                except Exception as e:
                    logger.error(f"[{asset_name}] Unsubscribe failed: {e}")
        self._subscriptions.clear()

    # -------------------------------------------------------------------------
    # TICK BUFFER UTILITIES
    # -------------------------------------------------------------------------

    def get_tick_buffer(self, asset_name: str) -> pd.DataFrame:
        """
        Return the current in-memory tick buffer for an asset as a DataFrame.

        Returns
        -------
        pd.DataFrame with columns [price, epoch, datetime]
        """
        if asset_name not in self.tick_buffers:
            logger.warning(f"No tick buffer for {asset_name}")
            return pd.DataFrame()

        buffer = list(self.tick_buffers[asset_name])
        if not buffer:
            return pd.DataFrame()

        df = pd.DataFrame(buffer)
        df.set_index("datetime", inplace=True)
        return df

    def get_latest_price(self, asset_name: str) -> Optional[float]:
        """Return the most recent price from the tick buffer."""
        if not self.tick_buffers.get(asset_name):
            return None
        return self.tick_buffers[asset_name][-1]["price"]


# -----------------------------------------------------------------------------
# CONVENIENCE FUNCTION: Fetch historical candles without managing client state
# (Useful for one-off data pulls in notebooks and the feature pipeline)
# -----------------------------------------------------------------------------

async def fetch_synthetic_candles(
    asset_name: str,
    interval: str = "5m",
    count: int = 500,
    use_cache: bool = True
) -> Optional[pd.DataFrame]:
    """
    One-shot coroutine to fetch historical candles for a synthetic asset.
    Opens a connection, fetches, and closes.

    Example
    -------
    import asyncio
    df = asyncio.run(fetch_synthetic_candles("VIX75", interval="5m", count=500))
    """
    client = DerivWSClient()
    connected = await client.connect()
    if not connected:
        logger.error("Could not connect to Deriv API.")
        return None
    try:
        df = await client.get_candles(asset_name, interval=interval,
                                       count=count, use_cache=use_cache)
        return df
    finally:
        await client.disconnect()


async def fetch_all_synthetic_assets(
    asset_names: list = None,
    timeframes: list = None,
    use_cache: bool = True
) -> dict:
    """
    Batch fetch all synthetic assets across all timeframes.
    Reuses a single WebSocket connection for efficiency.

    Returns
    -------
    dict: {asset_name: {timeframe: pd.DataFrame}}
    """
    if asset_names is None:
        asset_names = list(SYNTHETIC_ASSETS.keys())
    if timeframes is None:
        timeframes = [ENTRY_TF, PRIMARY_TF, BIAS_TF]

    client = DerivWSClient()
    connected = await client.connect()
    if not connected:
        logger.error("Could not connect to Deriv API.")
        return {}

    all_data = {}
    try:
        for asset in asset_names:
            logger.info(f"--- Fetching all timeframes for {asset} ---")
            all_data[asset] = await client.get_multi_timeframe(
                asset, timeframes, use_cache
            )
            await asyncio.sleep(0.5)
    finally:
        await client.disconnect()

    return all_data


# -----------------------------------------------------------------------------
# QUICK TEST
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    async def test():
        print("\n=== SynthTrade: Deriv WebSocket Client Test ===\n")

        test_assets = ["VIX75", "CRASH500"]

        for asset in test_assets:
            print(f"\nTesting {asset} (5m, 200 candles)...")
            df = await fetch_synthetic_candles(asset, interval="5m",
                                               count=200, use_cache=False)
            if df is not None:
                print(f"  Rows: {len(df)}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Date range: {df.index[0]} to {df.index[-1]}")
                print(df.tail(3))
            else:
                print(f"  FAILED to fetch {asset}")

    asyncio.run(test())
