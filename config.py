# =============================================================================
# SynthTrade - Central Configuration
# =============================================================================
# All asset lists, API settings, model thresholds, and feature parameters
# live here. Change values here and the entire pipeline updates automatically.

import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# API CREDENTIALS (loaded from .env file - never hardcode these)
# -----------------------------------------------------------------------------
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")        # Default is Deriv demo app ID
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", "")       # Your personal Deriv API token
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "") # Optional: Twelve Data for live forex/metals

# -----------------------------------------------------------------------------
# ASSET REGISTRY
# -----------------------------------------------------------------------------

# Deriv synthetic indices - these use Deriv WebSocket API
SYNTHETIC_ASSETS = {
    "VIX75":        "R_75",          # Volatility 75 Index
    "VIX25":        "R_25",          # Volatility 25 Index
    "CRASH500":     "CRASH500N",     # Crash 500 Index
    "CRASH1000":    "CRASH1000N",    # Crash 1000 Index
    "BOOM500":      "BOOM500N",      # Boom 500 Index
    "BOOM1000":     "BOOM1000N",     # Boom 1000 Index
    "STEP":         "stpRNG",        # Step Index
}

# Real market assets - fetched via yfinance (historical) or Twelve Data (live)
REAL_ASSETS = {
    # Indices
    "US100":    "NQ=F",      # Nasdaq 100 Futures (yfinance)
    "US30":     "YM=F",      # Dow Jones Futures (yfinance)

    # JPY Forex Pairs
    "USDJPY":   "JPY=X",
    "GBPJPY":   "GBPJPY=X",
    "EURJPY":   "EURJPY=X",

    # Metals
    "XAUUSD":   "GC=F",      # Gold Futures
    "XAGUSD":   "SI=F",      # Silver Futures
}

# All assets combined for unified processing
ALL_ASSETS = {**SYNTHETIC_ASSETS, **REAL_ASSETS}

# Assets to actively monitor (edit this to reduce load during development)
ACTIVE_ASSETS = [
    "VIX75", "CRASH500", "BOOM500",          # Synthetics
    "US100", "US30",                          # Indices
    "USDJPY", "GBPJPY", "EURJPY",            # Forex
    "XAUUSD", "XAGUSD",                      # Metals
]

# -----------------------------------------------------------------------------
# TIMEFRAME SETTINGS
# -----------------------------------------------------------------------------

# Primary signal generation timeframe (CNN-LSTM runs here)
PRIMARY_TF = "5m"

# Trend bias filter timeframe (no trades against this structure)
BIAS_TF = "15m"

# Entry precision timeframe (used for fine-tuning entry price)
ENTRY_TF = "1m"

# yfinance interval mapping
YF_INTERVALS = {
    "1m":  "1m",
    "3m":  "3m",
    "5m":  "5m",
    "15m": "15m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
}

# How many candles to fetch for each timeframe during historical download
HISTORICAL_CANDLES = {
    "1m":  1500,
    "3m":  1500,
    "5m":  2000,
    "15m": 2000,
    "1h":  2000,
    "4h":  1000,
    "1d":  500,
}

# Lookback window fed into the CNN-LSTM (number of candles per sequence)
SEQUENCE_LENGTH = 30

# -----------------------------------------------------------------------------
# FEATURE ENGINEERING PARAMETERS
# -----------------------------------------------------------------------------

EMA_PERIODS      = [9, 21, 50]     # Three EMA horizons
RSI_PERIOD       = 14
ATR_PERIOD       = 14
BB_PERIOD        = 20
BB_STD           = 2.0
SAR_ACCELERATION = 0.02
SAR_MAXIMUM      = 0.2
VOLUME_MA_PERIOD = 20              # For volume ratio calculation

# -----------------------------------------------------------------------------
# LABEL GENERATION PARAMETERS
# -----------------------------------------------------------------------------

# Minimum move to qualify as a valid signal (in ATR multiples)
MIN_REWARD_ATR   = 1.5    # Price must move 1.5x ATR in signal direction
MAX_RISK_ATR     = 1.0    # Without first hitting 1.0x ATR against signal

# How many forward candles to evaluate the label over
LABEL_LOOKAHEAD  = 5

# -----------------------------------------------------------------------------
# MODEL AND SIGNAL THRESHOLDS
# -----------------------------------------------------------------------------

# Minimum model confidence to surface a signal on the dashboard
SIGNAL_CONFIDENCE_THRESHOLD = 0.72

# Risk-reward ratio used for SL/TP calculation on live signals
SIGNAL_RR_RATIO  = 1.5    # TP = 1.5x the SL distance

# SL distance from entry (in ATR multiples)
SL_ATR_MULTIPLE  = 1.0
TP_ATR_MULTIPLE  = 1.5

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR    = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR      = os.path.join(BASE_DIR, "models", "saved")
RESULTS_DIR     = os.path.join(BASE_DIR, "backtest", "results")

# -----------------------------------------------------------------------------
# DERIV WEBSOCKET
# -----------------------------------------------------------------------------

DERIV_WS_URL    = "wss://ws.binaryws.com/websockets/v3"
DERIV_WS_TIMEOUT = 30       # seconds before reconnect attempt
DERIV_TICK_BUFFER = 5000    # max ticks to hold in memory per asset

# -----------------------------------------------------------------------------
# DISPLAY
# -----------------------------------------------------------------------------

SIGNAL_COLORS = {
    "LONG":    "#00C896",
    "SHORT":   "#FF4B4B",
    "NO_TRADE": "#888888",
}
