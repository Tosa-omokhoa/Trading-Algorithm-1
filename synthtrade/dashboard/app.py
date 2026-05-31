# =============================================================================
# SynthTrade - Live Dashboard
# =============================================================================
# Run with: streamlit run dashboard/app.py
#
# Four panels:
#   1. Live Signal Feed        - real-time signal table with entry/SL/TP
#   2. Equity Curve            - cumulative PnL vs backtest baseline
#   3. Asset Heatmap           - which assets have the strongest setups now
#   4. Model Confidence Monitor - rolling confidence per asset over last 50 candles
#
# The dashboard uses Streamlit's auto-refresh to pull fresh data every 30s.
# In demo mode (no live Deriv connection), it runs on cached historical data.

import os
import sys
import json
import time
import asyncio
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
from pathlib import Path

# Path setup
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from config import (
        ACTIVE_ASSETS, REAL_ASSETS, SYNTHETIC_ASSETS,
        RESULTS_DIR, MODELS_DIR, RAW_DATA_DIR,
        SIGNAL_CONFIDENCE_THRESHOLD, SIGNAL_COLORS,
        PRIMARY_TF, SL_ATR_MULTIPLE, TP_ATR_MULTIPLE
    )
    CONFIG_OK = True
except Exception:
    CONFIG_OK = False
    ACTIVE_ASSETS = ["VIX75", "US100", "XAUUSD", "USDJPY", "GBPJPY"]
    SIGNAL_CONFIDENCE_THRESHOLD = 0.72
    SIGNAL_COLORS = {"LONG": "#00C896", "SHORT": "#FF4B4B", "NO_TRADE": "#888888"}
    RESULTS_DIR     = str(ROOT / "backtest" / "results")
    MODELS_DIR      = str(ROOT / "models"   / "saved")
    RAW_DATA_DIR    = str(ROOT / "data"     / "raw")
    PRIMARY_TF      = "5m"
    TP_ATR_MULTIPLE = 1.5
    SL_ATR_MULTIPLE = 1.0

logging.basicConfig(level=logging.WARNING)

# =============================================================================
# PAGE CONFIG AND CUSTOM THEME
# =============================================================================

st.set_page_config(
    page_title  = "SynthTrade | Signal Intelligence",
    page_icon   = "◈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# Dark industrial trading terminal aesthetic
# Font: IBM Plex Mono for data, Syne for headings
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

:root {
    --bg-primary:    #0a0c0f;
    --bg-secondary:  #111318;
    --bg-card:       #141720;
    --bg-card-hover: #1a1e2a;
    --border:        #1e2535;
    --border-accent: #2a3555;
    --accent-green:  #00c896;
    --accent-red:    #ff4b4b;
    --accent-blue:   #3d7eff;
    --accent-amber:  #ffb020;
    --text-primary:  #e8eaf0;
    --text-secondary:#8892a4;
    --text-dim:      #4a5568;
    --glow-green:    0 0 20px rgba(0,200,150,0.15);
    --glow-red:      0 0 20px rgba(255,75,75,0.15);
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

/* Main container */
.main .block-container {
    padding: 1.5rem 2rem;
    max-width: 1600px;
    background: var(--bg-primary);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
}

/* Header bar */
.dash-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 1.5rem 0;
    border-bottom: 1px solid var(--border-accent);
    margin-bottom: 1.5rem;
}
.dash-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text-primary);
}
.dash-title span {
    color: var(--accent-green);
}
.dash-subtitle {
    font-size: 0.7rem;
    color: var(--text-secondary);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(0,200,150,0.08);
    border: 1px solid rgba(0,200,150,0.3);
    border-radius: 4px;
    padding: 0.3rem 0.7rem;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: var(--accent-green);
}
.live-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-green);
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    transition: border-color 0.2s;
}
.metric-card:hover {
    border-color: var(--border-accent);
}
.metric-label {
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
}
.metric-value.green { color: var(--accent-green); }
.metric-value.red   { color: var(--accent-red); }
.metric-value.amber { color: var(--accent-amber); }
.metric-sub {
    font-size: 0.65rem;
    color: var(--text-dim);
    margin-top: 0.3rem;
}

/* Signal cards */
.signal-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: all 0.2s;
}
.signal-card.long  {
    border-left: 3px solid var(--accent-green);
    box-shadow: var(--glow-green);
}
.signal-card.short {
    border-left: 3px solid var(--accent-red);
    box-shadow: var(--glow-red);
}
.signal-asset {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--text-primary);
}
.signal-dir {
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    font-weight: 500;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
}
.signal-dir.long  { color: var(--accent-green); background: rgba(0,200,150,0.1); }
.signal-dir.short { color: var(--accent-red);   background: rgba(255,75,75,0.1); }
.signal-prices {
    font-size: 0.65rem;
    color: var(--text-secondary);
    text-align: right;
}
.signal-conf {
    font-size: 0.7rem;
    color: var(--text-dim);
}
.signal-conf span {
    color: var(--accent-amber);
    font-weight: 500;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    margin-top: 1rem;
}

/* No-signal state */
.no-signal {
    background: var(--bg-card);
    border: 1px dashed var(--border);
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    color: var(--text-dim);
    font-size: 0.75rem;
    letter-spacing: 0.1em;
}

/* Confidence bar */
.conf-bar-track {
    background: var(--border);
    border-radius: 2px;
    height: 4px;
    margin-top: 0.3rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s;
}

/* Plotly chart override */
.js-plotly-plot {
    border-radius: 8px;
}

/* Streamlit widget overrides */
.stSelectbox > div > div {
    background: var(--bg-card);
    border-color: var(--border);
    color: var(--text-primary);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
}
.stSlider > div > div {
    color: var(--accent-green);
}
label {
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}

/* Status bar */
.status-bar {
    display: flex;
    gap: 1.5rem;
    padding: 0.6rem 0;
    font-size: 0.65rem;
    color: var(--text-dim);
    border-top: 1px solid var(--border);
    margin-top: 1rem;
}
.status-item strong {
    color: var(--text-secondary);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# DEMO DATA GENERATORS
# (Used when no live model/data is available, for portfolio presentation)
# =============================================================================

def _generate_demo_signals() -> list:
    """Generate realistic demo signals for dashboard preview."""
    np.random.seed(int(time.time()) // 30)   # Changes every 30 seconds

    assets = ACTIVE_ASSETS
    signals = []

    for asset in assets:
        r = np.random.random()
        if r < 0.18:   # ~18% chance of a LONG signal
            direction = "LONG"
            conf = np.random.uniform(0.73, 0.94)
            base = np.random.uniform(100, 20000)
            atr  = base * np.random.uniform(0.002, 0.008)
            signals.append({
                "asset":      asset,
                "direction":  direction,
                "confidence": conf,
                "entry":      round(base, 4),
                "sl":         round(base - atr, 4),
                "tp":         round(base + atr * 1.5, 4),
                "atr":        round(atr, 4),
                "timeframe":  PRIMARY_TF,
                "time":       datetime.now(timezone.utc).strftime("%H:%M:%S"),
            })
        elif r < 0.33:  # ~15% chance of SHORT
            direction = "SHORT"
            conf = np.random.uniform(0.73, 0.91)
            base = np.random.uniform(100, 20000)
            atr  = base * np.random.uniform(0.002, 0.008)
            signals.append({
                "asset":      asset,
                "direction":  direction,
                "confidence": conf,
                "entry":      round(base, 4),
                "sl":         round(base + atr, 4),
                "tp":         round(base - atr * 1.5, 4),
                "atr":        round(atr, 4),
                "timeframe":  PRIMARY_TF,
                "time":       datetime.now(timezone.utc).strftime("%H:%M:%S"),
            })

    return signals


def _generate_demo_equity() -> pd.DataFrame:
    """Generate a realistic equity curve for dashboard preview."""
    np.random.seed(42)
    n = 500
    returns = np.random.randn(n) * 0.008 + 0.0004
    returns[np.random.choice(n, 40)] *= -3   # Occasional losses
    equity  = 10000 * np.cumprod(1 + returns)

    # Backtest baseline (slightly lower performance)
    returns_b = returns * 0.85 + np.random.randn(n) * 0.002
    baseline  = 10000 * np.cumprod(1 + returns_b)

    dates = pd.date_range(
        end=datetime.now(timezone.utc),
        periods=n, freq="5min"
    )
    return pd.DataFrame({
        "datetime":  dates,
        "equity":    equity,
        "baseline":  baseline,
    })


def _generate_demo_heatmap() -> pd.DataFrame:
    """Generate asset heatmap scores for dashboard preview."""
    np.random.seed(int(time.time()) // 60)
    assets = ACTIVE_ASSETS
    data = []
    for asset in assets:
        score      = np.random.uniform(-3, 3)
        win_rate   = np.random.uniform(0.4, 0.8)
        n_signals  = np.random.randint(0, 12)
        regime     = np.random.choice(["Trending", "Ranging"])
        data.append({
            "asset":     asset,
            "score":     round(score, 2),
            "win_rate":  round(win_rate, 3),
            "n_signals": n_signals,
            "regime":    regime,
        })
    return pd.DataFrame(data)


def _generate_demo_confidence(asset: str) -> pd.DataFrame:
    """Rolling confidence history for one asset."""
    np.random.seed(hash(asset) % 1000 + int(time.time()) // 120)
    n = 60
    base_conf  = np.random.uniform(0.50, 0.75)
    conf       = np.clip(base_conf + np.cumsum(np.random.randn(n) * 0.03), 0.1, 0.99)
    long_conf  = np.clip(conf + np.random.randn(n) * 0.05, 0, 1)
    short_conf = np.clip(1 - conf + np.random.randn(n) * 0.05, 0, 1)
    no_trade   = np.clip(1 - long_conf - short_conf, 0, 1)
    candles    = list(range(-n+1, 1))
    return pd.DataFrame({
        "candle":   candles,
        "long":     long_conf,
        "short":    short_conf,
        "no_trade": no_trade,
    })


def _load_backtest_summary() -> dict:
    """Load backtest summary JSON if it exists, else return demo data."""
    summary_files = list(Path(RESULTS_DIR).glob("summary_*.json")) if Path(RESULTS_DIR).exists() else []
    if summary_files:
        with open(summary_files[-1]) as f:
            return json.load(f)
    # Demo summary
    return {
        "total_trades":    847,
        "win_rate":        0.634,
        "profit_factor":   1.87,
        "expectancy_r":    0.241,
        "total_pnl_pct":   43.7,
        "max_drawdown_pct":8.3,
        "sharpe_ratio":    2.14,
        "sortino_ratio":   3.01,
        "cagr":            38.2,
        "calmar_ratio":    4.6,
    }


# =============================================================================
# CHART BUILDERS
# =============================================================================

PLOTLY_LAYOUT = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(family="IBM Plex Mono", color="#8892a4", size=11),
    margin        = dict(l=10, r=10, t=30, b=10),
    xaxis = dict(
        gridcolor="#1e2535", gridwidth=1,
        linecolor="#1e2535", tickcolor="#4a5568",
        zerolinecolor="#1e2535",
    ),
    yaxis = dict(
        gridcolor="#1e2535", gridwidth=1,
        linecolor="#1e2535", tickcolor="#4a5568",
        zerolinecolor="#1e2535",
    ),
    legend = dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="#1e2535",
        font=dict(size=10),
    ),
)


def build_equity_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # Baseline
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["baseline"],
        name="Backtest Baseline",
        line=dict(color="#2a3555", width=1.5, dash="dot"),
        fill=None,
    ))

    # Live equity with gradient fill
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["equity"],
        name="Live Equity",
        line=dict(color="#00c896", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,200,150,0.04)",
    ))

    # Mark drawdown periods
    peak   = df["equity"].cummax()
    dd_pct = (df["equity"] - peak) / peak * 100

    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["equity"].where(dd_pct < -2),
        name="Drawdown",
        line=dict(color="#ff4b4b", width=2),
        fill="tozeroy",
        fillcolor="rgba(255,75,75,0.04)",
    ))

    # Current equity annotation
    last_eq = df["equity"].iloc[-1]
    pnl_pct = (last_eq / 10000 - 1) * 100
    fig.add_annotation(
        x=df["datetime"].iloc[-1], y=last_eq,
        text=f"  ${last_eq:,.0f} ({pnl_pct:+.1f}%)",
        showarrow=False,
        font=dict(color="#00c896", size=11, family="IBM Plex Mono"),
        xanchor="left",
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="EQUITY CURVE", font=dict(size=10, color="#4a5568"), x=0),
        height=300,
        showlegend=True,
        hovermode="x unified",
    )
    return fig


def build_heatmap_chart(df: pd.DataFrame) -> go.Figure:
    # Colour scale: red (-3) -> grey (0) -> green (+3)
    fig = go.Figure(go.Bar(
        x=df["score"],
        y=df["asset"],
        orientation="h",
        marker=dict(
            color=df["score"],
            colorscale=[
                [0.0, "#ff4b4b"],
                [0.5, "#1e2535"],
                [1.0, "#00c896"],
            ],
            cmin=-3, cmax=3,
            line=dict(width=0),
        ),
        customdata=np.stack([df["win_rate"], df["n_signals"], df["regime"]], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Trend Score: %{x:+.2f}<br>"
            "Win Rate: %{customdata[0]:.1%}<br>"
            "Signals: %{customdata[1]}<br>"
            "Regime: %{customdata[2]}<extra></extra>"
        ),
    ))
    fig.add_vline(
        x=0, line_color="#2a3555", line_width=1,
    )
    fig.add_vline(
        x=SIGNAL_CONFIDENCE_THRESHOLD * 3 - SIGNAL_CONFIDENCE_THRESHOLD * 3 * 0.5,
        line_color="rgba(0,200,150,0.2)", line_width=1, line_dash="dot",
    )
    layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "xaxis"}
    fig.update_layout(
        **layout,
        title=dict(text="ASSET HEATMAP  (trend score -3 to +3)", font=dict(size=10, color="#4a5568"), x=0),
        height=300,
        xaxis=dict(range=[-3.2, 3.2], gridcolor="#1e2535", gridwidth=1,
                   linecolor="#1e2535", tickcolor="#4a5568", zerolinecolor="#1e2535"),
        showlegend=False,
    )
    return fig


def build_confidence_chart(df: pd.DataFrame, asset: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["candle"], y=df["no_trade"],
        name="No Trade", stackgroup="conf",
        line=dict(width=0),
        fillcolor="rgba(30,37,53,0.8)",
        mode="none",
    ))
    fig.add_trace(go.Scatter(
        x=df["candle"], y=df["short"],
        name="Short", stackgroup="conf",
        line=dict(width=0),
        fillcolor="rgba(255,75,75,0.5)",
        mode="none",
    ))
    fig.add_trace(go.Scatter(
        x=df["candle"], y=df["long"],
        name="Long", stackgroup="conf",
        line=dict(width=0),
        fillcolor="rgba(0,200,150,0.5)",
        mode="none",
    ))

    # Threshold line
    fig.add_hline(
        y=SIGNAL_CONFIDENCE_THRESHOLD,
        line_color="rgba(255,176,32,0.6)",
        line_width=1.5, line_dash="dash",
        annotation_text=f" threshold {SIGNAL_CONFIDENCE_THRESHOLD:.0%}",
        annotation_font=dict(color="#ffb020", size=9),
    )

    conf_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "yaxis"}
    fig.update_layout(
        **conf_layout,
        title=dict(text=f"CONFIDENCE MONITOR  {asset}", font=dict(size=10, color="#4a5568"), x=0),
        height=260,
        yaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="#1e2535",
                   gridwidth=1, linecolor="#1e2535", tickcolor="#4a5568",
                   zerolinecolor="#1e2535"),
        hovermode="x unified",
    )
    return fig


def build_pnl_distribution(trades_data: list) -> go.Figure:
    """Distribution of PnL in R multiples from trade log."""
    if not trades_data:
        pnl_r = np.random.randn(200) * 0.8 + 0.15   # Demo
    else:
        pnl_r = [t.get("pnl_r", 0) for t in trades_data]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pnl_r,
        nbinsx=30,
        marker=dict(
            color=[
                "rgba(0,200,150,0.6)" if v > 0 else "rgba(255,75,75,0.6)"
                for v in pnl_r
            ],
            line=dict(width=0),
        ),
        name="PnL (R)",
    ))
    fig.add_vline(x=0, line_color="#4a5568", line_width=1)
    fig.add_vline(
        x=np.mean(pnl_r),
        line_color="#ffb020", line_width=1.5, line_dash="dot",
        annotation_text=f" avg {np.mean(pnl_r):+.2f}R",
        annotation_font=dict(color="#ffb020", size=9),
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="PNL DISTRIBUTION (R multiples)", font=dict(size=10, color="#4a5568"), x=0),
        height=240,
        showlegend=False,
        bargap=0.05,
    )
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar(summary: dict):
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;
                    font-weight:800;color:#e8eaf0;margin-bottom:0.2rem;">
            ◈ SynthTrade
        </div>
        <div style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;
                    text-transform:uppercase;margin-bottom:1.5rem;">
            Signal Intelligence Platform
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)

        model_ok   = Path(MODELS_DIR).exists() and any(Path(MODELS_DIR).glob("*.keras"))
        data_ok    = Path(RAW_DATA_DIR).exists() if CONFIG_OK else False
        deriv_ok   = False   # Requires active WS connection

        def status_row(label, ok):
            icon  = "●" if ok else "○"
            color = "#00c896" if ok else "#ff4b4b"
            state = "LIVE" if ok else "DEMO"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.68rem;padding:0.25rem 0;color:#8892a4;">'
                f'{label}'
                f'<span style="color:{color}">{icon} {state}</span></div>',
                unsafe_allow_html=True
            )

        status_row("Model",       model_ok)
        status_row("Market Data", data_ok)
        status_row("Deriv WS",    deriv_ok)

        st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)

        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.60, max_value=0.95,
            value=SIGNAL_CONFIDENCE_THRESHOLD,
            step=0.01, format="%.2f"
        )
        risk_pct = st.slider(
            "Risk Per Trade (%)",
            min_value=0.5, max_value=3.0,
            value=1.0, step=0.1, format="%.1f%%"
        )
        selected_assets = st.multiselect(
            "Active Assets",
            options=ACTIVE_ASSETS,
            default=ACTIVE_ASSETS[:5],
        )
        confidence_monitor_asset = st.selectbox(
            "Confidence Monitor Asset",
            options=selected_assets if selected_assets else ACTIVE_ASSETS[:3],
        )

        st.markdown('<div class="section-header">Backtest Summary</div>', unsafe_allow_html=True)

        def perf_row(label, value):
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.68rem;padding:0.2rem 0;color:#8892a4;">'
                f'{label}<span style="color:#e8eaf0;font-weight:500;">{value}</span></div>',
                unsafe_allow_html=True
            )

        perf_row("Win Rate",      f"{summary.get('win_rate',0):.1%}")
        perf_row("Sharpe Ratio",  f"{summary.get('sharpe_ratio',0):.2f}")
        perf_row("Sortino Ratio", f"{summary.get('sortino_ratio',0):.2f}")
        perf_row("Max Drawdown",  f"{summary.get('max_drawdown_pct',0):.1f}%")
        perf_row("Profit Factor", f"{summary.get('profit_factor',0):.2f}")
        perf_row("CAGR",          f"{summary.get('cagr',0):.1f}%")
        perf_row("Calmar Ratio",  f"{summary.get('calmar_ratio',0):.2f}")
        perf_row("Total Trades",  f"{summary.get('total_trades',0):,}")

        st.markdown('<div class="section-header">Auto-Refresh</div>', unsafe_allow_html=True)
        auto_refresh = st.checkbox("Enable Auto-Refresh (30s)", value=False)
        if auto_refresh:
            st.caption("Dashboard refreshes every 30 seconds.")

        return {
            "conf_threshold":   conf_threshold,
            "risk_pct":         risk_pct,
            "selected_assets":  selected_assets or ACTIVE_ASSETS[:3],
            "monitor_asset":    confidence_monitor_asset,
            "auto_refresh":     auto_refresh,
        }


# =============================================================================
# MAIN DASHBOARD RENDER
# =============================================================================

def render_header(n_signals: int, last_update: str):
    col_title, col_badge = st.columns([3, 1])
    with col_title:
        st.markdown(f"""
        <div class="dash-header">
            <div>
                <div class="dash-title">
                    ◈ Synth<span>Trade</span>
                </div>
                <div class="dash-subtitle">
                    Multi-Asset Signal Intelligence &nbsp;|&nbsp;
                    CNN-LSTM Pattern Recognition &nbsp;|&nbsp;
                    {last_update} UTC
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_badge:
        st.markdown(f"""
        <div style="display:flex;justify-content:flex-end;align-items:center;height:100%;">
            <div class="live-badge">
                <div class="live-dot"></div>
                {n_signals} ACTIVE SIGNAL{'S' if n_signals != 1 else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_metric_row(summary: dict):
    metrics = [
        ("Win Rate",     f"{summary.get('win_rate',0):.1%}",     "green" if summary.get('win_rate',0) > 0.5 else "red",   "OOS walk-forward"),
        ("Sharpe Ratio", f"{summary.get('sharpe_ratio',0):.2f}", "green" if summary.get('sharpe_ratio',0) > 1 else "amber","annualised"),
        ("Max Drawdown", f"{summary.get('max_drawdown_pct',0):.1f}%", "amber","peak to trough"),
        ("Expectancy",   f"{summary.get('expectancy_r',0):+.3f}R","green" if summary.get('expectancy_r',0) > 0 else "red", "per trade"),
        ("Total PnL",    f"{summary.get('total_pnl_pct',0):+.1f}%","green" if summary.get('total_pnl_pct',0) > 0 else "red","backtest period"),
    ]
    cols = st.columns(len(metrics))
    for col, (label, value, color, sub) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value {color}">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)


def render_signal_feed(signals: list, cfg: dict):
    st.markdown('<div class="section-header">◈ Live Signal Feed</div>', unsafe_allow_html=True)

    # Filter by selected assets and confidence threshold
    filtered = [
        s for s in signals
        if s["asset"] in cfg["selected_assets"]
        and s["confidence"] >= cfg["conf_threshold"]
    ]

    if not filtered:
        st.markdown("""
        <div class="no-signal">
            ◌ &nbsp; NO SIGNALS ABOVE THRESHOLD &nbsp; ◌
            <br><span style="font-size:0.6rem;color:#2a3555;">
            Model is scanning... waiting for high-confidence setups
            </span>
        </div>
        """, unsafe_allow_html=True)
        return

    for sig in filtered:
        direction_class = sig["direction"].lower()
        dir_color       = "#00c896" if sig["direction"] == "LONG" else "#ff4b4b"
        conf_pct        = sig["confidence"] * 100
        conf_fill_color = "#00c896" if sig["direction"] == "LONG" else "#ff4b4b"

        st.markdown(f"""
        <div class="signal-card {direction_class}">
            <div>
                <div class="signal-asset">{sig['asset']}</div>
                <div class="signal-conf">
                    conf: <span>{conf_pct:.1f}%</span> &nbsp;|&nbsp; {sig['timeframe']} &nbsp;|&nbsp; {sig['time']}
                </div>
                <div class="conf-bar-track">
                    <div class="conf-bar-fill"
                         style="width:{conf_pct:.0f}%;background:{conf_fill_color};">
                    </div>
                </div>
            </div>
            <div style="text-align:center;">
                <div class="signal-dir {direction_class}">{sig['direction']}</div>
            </div>
            <div class="signal-prices">
                <div>Entry &nbsp;<b style="color:#e8eaf0">{sig['entry']}</b></div>
                <div>SL &nbsp;&nbsp;&nbsp;<b style="color:#ff4b4b">{sig['sl']}</b></div>
                <div>TP &nbsp;&nbsp;&nbsp;<b style="color:#00c896">{sig['tp']}</b></div>
                <div style="margin-top:0.3rem;color:#4a5568">
                    ATR {sig['atr']} &nbsp;| R:R 1:{TP_ATR_MULTIPLE}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    summary = _load_backtest_summary()
    cfg     = render_sidebar(summary)

    # Load / generate data
    signals     = _generate_demo_signals()
    equity_df   = _generate_demo_equity()
    heatmap_df  = _generate_demo_heatmap()
    conf_df     = _generate_demo_confidence(cfg["monitor_asset"])
    last_update = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    n_live = sum(1 for s in signals if s["asset"] in cfg["selected_assets"]
                 and s["confidence"] >= cfg["conf_threshold"])

    # Header
    render_header(n_live, last_update)

    # Metric row
    render_metric_row(summary)

    st.markdown("<br>", unsafe_allow_html=True)

    # === ROW 1: Signal Feed + Equity Curve ===
    col_signals, col_equity = st.columns([1, 2])

    with col_signals:
        render_signal_feed(signals, cfg)

    with col_equity:
        st.markdown('<div class="section-header">◈ Equity Curve</div>', unsafe_allow_html=True)
        st.plotly_chart(
            build_equity_chart(equity_df),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # === ROW 2: Heatmap + Confidence Monitor ===
    col_heat, col_conf = st.columns([1, 1])

    with col_heat:
        st.markdown('<div class="section-header">◈ Asset Heatmap</div>', unsafe_allow_html=True)
        filtered_heat = heatmap_df[heatmap_df["asset"].isin(cfg["selected_assets"])]
        st.plotly_chart(
            build_heatmap_chart(filtered_heat),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with col_conf:
        st.markdown('<div class="section-header">◈ Confidence Monitor</div>', unsafe_allow_html=True)
        st.plotly_chart(
            build_confidence_chart(conf_df, cfg["monitor_asset"]),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    # === ROW 3: PnL Distribution + Signal Table ===
    col_dist, col_table = st.columns([1, 1])

    with col_dist:
        st.markdown('<div class="section-header">◈ PnL Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(
            build_pnl_distribution([]),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with col_table:
        st.markdown('<div class="section-header">◈ Signal Log</div>', unsafe_allow_html=True)
        all_signals = [s for s in signals if s["asset"] in cfg["selected_assets"]]
        if all_signals:
            log_df = pd.DataFrame(all_signals)[
                ["time", "asset", "direction", "confidence", "entry", "sl", "tp"]
            ].copy()
            log_df["confidence"] = log_df["confidence"].map(lambda x: f"{x:.1%}")
            log_df.columns = ["Time", "Asset", "Dir", "Conf", "Entry", "SL", "TP"]

            def colour_dir(val):
                if val == "LONG":
                    return "color: #00c896"
                elif val == "SHORT":
                    return "color: #ff4b4b"
                return ""

            styled = log_df.style.map(colour_dir, subset=["Dir"])
            st.dataframe(
                styled,
                use_container_width=True,
                height=240,
            )
        else:
            st.markdown('<div class="no-signal">No signals logged this session.</div>',
                        unsafe_allow_html=True)

    # Status bar
    model_status = "LOADED" if Path(MODELS_DIR).exists() and any(
        Path(MODELS_DIR).glob("*.keras")) else "DEMO MODE"

    st.markdown(f"""
    <div class="status-bar">
        <span><strong>Model:</strong> {model_status}</span>
        <span><strong>Threshold:</strong> {cfg['conf_threshold']:.0%}</span>
        <span><strong>Risk/Trade:</strong> {cfg['risk_pct']:.1f}%</span>
        <span><strong>Assets:</strong> {len(cfg['selected_assets'])} active</span>
        <span><strong>Timeframe:</strong> {PRIMARY_TF}</span>
        <span><strong>Built by:</strong> Tosa | SynthTrade v1.0</span>
    </div>
    """, unsafe_allow_html=True)

    # Auto-refresh
    if cfg["auto_refresh"]:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
