# =============================================================================
# SynthTrade - Backtesting Engine
# =============================================================================
# Vectorised backtester that replays model signals against historical data
# and computes a full suite of trading performance metrics.
#
# Design principles:
#   - NO lookahead bias: each signal only uses data available at that candle
#   - ATR-based dynamic SL/TP: adapts to current volatility, not fixed pips
#   - Risk-per-trade sizing: position size scales so every trade risks the
#     same % of current equity (realistic money management)
#   - Slippage and spread simulation: synthetic indices have near-zero spread
#     but real forex/metals have meaningful spread costs
#   - Consecutive loss tracking: the "max consecutive losses" stat tells you
#     how psychologically brutal a drawdown period would be in live trading
#
# Metrics computed:
#   Win rate, total trades, profit factor
#   Sharpe Ratio (annualised), Sortino Ratio (annualised)
#   Max drawdown (%), Max drawdown duration (candles)
#   Expectancy (avg PnL per trade in R multiples)
#   CAGR (compound annual growth rate)
#   Calmar Ratio (CAGR / max drawdown)
#   Equity curve (full time series)
#   Per-asset breakdown
#   Walk-forward out-of-sample performance

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from config import (
        SL_ATR_MULTIPLE, TP_ATR_MULTIPLE, SIGNAL_CONFIDENCE_THRESHOLD,
        RESULTS_DIR, SIGNAL_RR_RATIO
    )
except ImportError:
    SL_ATR_MULTIPLE  = 1.0
    TP_ATR_MULTIPLE  = 1.5
    SIGNAL_CONFIDENCE_THRESHOLD = 0.72
    SIGNAL_RR_RATIO  = 1.5
    RESULTS_DIR      = "backtest/results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Backtest")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Trade:
    """Represents a single completed trade."""
    trade_id:       int
    asset:          str
    signal:         int           # 1=LONG, -1=SHORT
    entry_price:    float
    sl_price:       float
    tp_price:       float
    entry_idx:      int           # candle index of entry
    exit_idx:       int           # candle index of exit
    exit_price:     float
    exit_reason:    str           # "TP", "SL", "EXPIRED", "SIGNAL_FLIP"
    pnl_r:          float         # PnL in R multiples (1R = SL distance)
    pnl_pct:        float         # PnL as % of equity at entry
    equity_at_entry:float
    equity_at_exit: float
    confidence:     float         # Model confidence score for this signal
    atr_at_entry:   float
    candle_time:    Optional[str] = None


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""
    asset_name:         str   = "ASSET"
    initial_equity:     float = 10_000.0
    risk_per_trade_pct: float = 1.0        # % of equity risked per trade
    sl_atr_mult:        float = SL_ATR_MULTIPLE
    tp_atr_mult:        float = TP_ATR_MULTIPLE
    max_hold_candles:   int   = 20         # Expire trade after this many candles
    spread_pct:         float = 0.0        # Spread as % of price (0 for synthetics)
    commission_pct:     float = 0.0        # Commission per trade (%)
    confidence_threshold: float = SIGNAL_CONFIDENCE_THRESHOLD
    candles_per_year:   int   = 26_280     # 5-min candles in a year (~252 days * 6.5h * 12)
    allow_pyramiding:   bool  = False      # Whether to allow multiple open positions
    min_atr_filter:     float = 0.0        # Skip signals if ATR below this (avoids flat markets)


@dataclass
class BacktestResult:
    """Full results from one backtest run."""
    config:             BacktestConfig
    trades:             List[Trade]        = field(default_factory=list)
    equity_curve:       List[float]        = field(default_factory=list)
    equity_timestamps:  List[str]          = field(default_factory=list)

    # --- Summary metrics (populated by compute_metrics) ---
    total_trades:       int   = 0
    winning_trades:     int   = 0
    losing_trades:      int   = 0
    win_rate:           float = 0.0
    avg_win_r:          float = 0.0
    avg_loss_r:         float = 0.0
    profit_factor:      float = 0.0
    expectancy_r:       float = 0.0
    total_pnl_pct:      float = 0.0
    max_drawdown_pct:   float = 0.0
    max_dd_duration:    int   = 0
    sharpe_ratio:       float = 0.0
    sortino_ratio:      float = 0.0
    cagr:               float = 0.0
    calmar_ratio:       float = 0.0
    max_consec_losses:  int   = 0
    final_equity:       float = 0.0
    long_trades:        int   = 0
    short_trades:       int   = 0
    long_win_rate:      float = 0.0
    short_win_rate:     float = 0.0
    avg_hold_candles:   float = 0.0
    signal_coverage:    float = 0.0


# =============================================================================
# CORE BACKTESTING ENGINE
# =============================================================================

class VectorisedBacktester:
    """
    Replays model signals against historical OHLCV data and simulates
    trade execution with ATR-based SL/TP, position sizing, and costs.

    Usage:
        engine = VectorisedBacktester(config)
        result = engine.run(df_features, signals, confidences)
        report = engine.generate_report(result)
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(
        self,
        df: pd.DataFrame,
        signals: np.ndarray,
        confidences: np.ndarray,
    ) -> BacktestResult:
        """
        Execute the backtest.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame with columns including ha_close, ha_high,
            ha_low, atr. Must be aligned with signals array.
        signals : np.ndarray
            Integer signal array (-1, 0, 1), same length as df.
        confidences : np.ndarray
            Model confidence scores (0-1), same length as df.

        Returns
        -------
        BacktestResult with all trades and equity curve populated.
        """
        cfg     = self.config
        result  = BacktestResult(config=cfg)
        equity  = cfg.initial_equity
        trade_id = 0

        closes    = df["ha_close"].values
        highs     = df["ha_high"].values
        lows      = df["ha_low"].values
        atrs      = df["atr"].values
        n         = len(df)
        timestamps= df.index.astype(str).tolist() if hasattr(df.index, 'astype') else [str(i) for i in range(n)]

        result.equity_curve.append(equity)
        result.equity_timestamps.append(timestamps[0] if timestamps else "0")

        i = 0
        while i < n - 1:
            sig  = signals[i]
            conf = confidences[i]
            atr  = atrs[i]

            # Skip no-trade signals and below-threshold confidence
            if sig == 0 or conf < cfg.confidence_threshold:
                result.equity_curve.append(equity)
                result.equity_timestamps.append(timestamps[i])
                i += 1
                continue

            # Skip if ATR is too low (flat/illiquid market)
            if atr < cfg.min_atr_filter:
                result.equity_curve.append(equity)
                result.equity_timestamps.append(timestamps[i])
                i += 1
                continue

            entry_price = closes[i]

            # Apply spread cost on entry
            if sig == 1:
                entry_price *= (1 + cfg.spread_pct / 2 / 100)
            else:
                entry_price *= (1 - cfg.spread_pct / 2 / 100)

            # SL and TP based on ATR multiples
            sl_dist = atr * cfg.sl_atr_mult
            tp_dist = atr * cfg.tp_atr_mult

            if sig == 1:   # LONG
                sl_price = entry_price - sl_dist
                tp_price = entry_price + tp_dist
            else:          # SHORT
                sl_price = entry_price + sl_dist
                tp_price = entry_price - tp_dist

            # Position sizing: risk a fixed % of current equity
            # risk_amount = equity * risk_pct / 100
            # position_size = risk_amount / sl_dist (in price units)
            risk_amount   = equity * cfg.risk_per_trade_pct / 100
            position_size = risk_amount / sl_dist if sl_dist > 0 else 0

            if position_size <= 0:
                i += 1
                continue

            # --- Simulate trade forward ---
            exit_price  = None
            exit_reason = "EXPIRED"
            exit_idx    = min(i + cfg.max_hold_candles, n - 1)

            for j in range(i + 1, min(i + cfg.max_hold_candles + 1, n)):
                high_j = highs[j]
                low_j  = lows[j]

                if sig == 1:   # LONG: check SL (low touches sl) then TP (high touches tp)
                    if low_j <= sl_price:
                        exit_price  = sl_price
                        exit_reason = "SL"
                        exit_idx    = j
                        break
                    if high_j >= tp_price:
                        exit_price  = tp_price
                        exit_reason = "TP"
                        exit_idx    = j
                        break

                else:          # SHORT: check SL (high touches sl) then TP (low touches tp)
                    if high_j >= sl_price:
                        exit_price  = sl_price
                        exit_reason = "SL"
                        exit_idx    = j
                        break
                    if low_j <= tp_price:
                        exit_price  = tp_price
                        exit_reason = "TP"
                        exit_idx    = j
                        break

            if exit_price is None:
                exit_price  = closes[exit_idx]
                exit_reason = "EXPIRED"

            # Apply spread cost on exit
            if sig == 1:
                exit_price *= (1 - cfg.spread_pct / 2 / 100)
            else:
                exit_price *= (1 + cfg.spread_pct / 2 / 100)

            # Calculate PnL
            if sig == 1:
                raw_pnl = (exit_price - entry_price) * position_size
            else:
                raw_pnl = (entry_price - exit_price) * position_size

            # Commission
            commission = equity * cfg.commission_pct / 100
            raw_pnl   -= commission

            # PnL in R multiples
            pnl_r   = raw_pnl / risk_amount if risk_amount > 0 else 0
            pnl_pct = raw_pnl / equity * 100

            equity_before = equity
            equity       += raw_pnl
            equity        = max(equity, 0.01)   # Prevent negative equity

            trade = Trade(
                trade_id       = trade_id,
                asset          = cfg.asset_name,
                signal         = sig,
                entry_price    = entry_price,
                sl_price       = sl_price,
                tp_price       = tp_price,
                entry_idx      = i,
                exit_idx       = exit_idx,
                exit_price     = exit_price,
                exit_reason    = exit_reason,
                pnl_r          = pnl_r,
                pnl_pct        = pnl_pct,
                equity_at_entry= equity_before,
                equity_at_exit = equity,
                confidence     = conf,
                atr_at_entry   = atr,
                candle_time    = timestamps[i]
            )
            result.trades.append(trade)
            trade_id += 1

            # Advance past the trade
            for k in range(i, exit_idx + 1):
                result.equity_curve.append(equity)
                result.equity_timestamps.append(timestamps[min(k, n-1)])

            i = exit_idx + 1

        # Pad equity curve to full length
        while len(result.equity_curve) < n:
            result.equity_curve.append(equity)
            result.equity_timestamps.append(timestamps[min(len(result.equity_curve)-1, n-1)])

        result.final_equity = equity
        result = self.compute_metrics(result)
        return result

    # -------------------------------------------------------------------------
    # METRICS COMPUTATION
    # -------------------------------------------------------------------------

    def compute_metrics(self, result: BacktestResult) -> BacktestResult:
        """Compute all summary statistics from the trade list and equity curve."""
        cfg    = self.config
        trades = result.trades

        if not trades:
            logger.warning(f"[{cfg.asset_name}] No trades generated.")
            return result

        # Basic counts
        result.total_trades  = len(trades)
        winners = [t for t in trades if t.pnl_r > 0]
        losers  = [t for t in trades if t.pnl_r <= 0]

        result.winning_trades = len(winners)
        result.losing_trades  = len(losers)
        result.win_rate       = len(winners) / len(trades) if trades else 0.0

        # Direction breakdown
        longs  = [t for t in trades if t.signal == 1]
        shorts = [t for t in trades if t.signal == -1]
        result.long_trades   = len(longs)
        result.short_trades  = len(shorts)
        result.long_win_rate  = (
            sum(1 for t in longs  if t.pnl_r > 0) / len(longs)  if longs  else 0.0
        )
        result.short_win_rate = (
            sum(1 for t in shorts if t.pnl_r > 0) / len(shorts) if shorts else 0.0
        )

        # Win/loss averages in R
        result.avg_win_r  = np.mean([t.pnl_r for t in winners]) if winners else 0.0
        result.avg_loss_r = np.mean([t.pnl_r for t in losers])  if losers  else 0.0

        # Profit factor: gross wins / gross losses
        gross_wins  = sum(t.pnl_r for t in winners)
        gross_losses= abs(sum(t.pnl_r for t in losers))
        result.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        # Expectancy: expected R per trade
        result.expectancy_r = np.mean([t.pnl_r for t in trades])

        # Total PnL %
        result.total_pnl_pct = (
            (result.final_equity - cfg.initial_equity) / cfg.initial_equity * 100
        )

        # Hold time
        result.avg_hold_candles = np.mean([t.exit_idx - t.entry_idx for t in trades])

        # Signal coverage
        n_candles = len(result.equity_curve)
        result.signal_coverage = len(trades) / n_candles if n_candles > 0 else 0.0

        # --- Equity curve metrics ---
        equity_arr = np.array(result.equity_curve)

        # Returns per candle
        returns = np.diff(equity_arr) / equity_arr[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) > 1:
            # Sharpe Ratio (annualised, risk-free rate = 0)
            mean_r  = np.mean(returns)
            std_r   = np.std(returns)
            result.sharpe_ratio = (
                (mean_r / std_r) * np.sqrt(cfg.candles_per_year)
                if std_r > 0 else 0.0
            )

            # Sortino Ratio (only penalises downside volatility)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 0.0
            result.sortino_ratio = (
                (mean_r / downside_std) * np.sqrt(cfg.candles_per_year)
                if downside_std > 0 else 0.0
            )

        # Max Drawdown
        peak      = equity_arr[0]
        max_dd    = 0.0
        dd_start  = 0
        max_dd_dur = 0
        current_dd_dur = 0

        for k, eq in enumerate(equity_arr):
            if eq > peak:
                peak = eq
                current_dd_dur = 0
            else:
                current_dd_dur += 1
                max_dd_dur = max(max_dd_dur, current_dd_dur)

            dd = (peak - eq) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        result.max_drawdown_pct = max_dd * 100
        result.max_dd_duration  = max_dd_dur

        # CAGR
        n_years = len(equity_arr) / cfg.candles_per_year
        if n_years > 0 and cfg.initial_equity > 0:
            result.cagr = (
                (result.final_equity / cfg.initial_equity) ** (1 / n_years) - 1
            ) * 100
        else:
            result.cagr = 0.0

        # Calmar Ratio
        result.calmar_ratio = (
            result.cagr / result.max_drawdown_pct
            if result.max_drawdown_pct > 0 else 0.0
        )

        # Max consecutive losses
        consec = 0
        max_consec = 0
        for t in trades:
            if t.pnl_r <= 0:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0
        result.max_consec_losses = max_consec

        return result


# =============================================================================
# WALK-FORWARD BACKTESTER
# =============================================================================

class WalkForwardBacktester:
    """
    Runs the backtest across all walk-forward folds to produce honest
    out-of-sample performance estimates.

    For each fold, the model was trained on data BEFORE the validation window.
    This backtester runs signal generation only on the validation windows,
    so every trade in the report was generated by a model that never saw
    that data during training.

    This is the gold standard for evaluating trading systems. Any strategy
    can be curve-fitted to look good in-sample. What matters is OOS performance.
    """

    def __init__(self, base_config: BacktestConfig):
        self.base_config = base_config

    def run_folds(
        self,
        df: pd.DataFrame,
        all_signals: np.ndarray,
        all_confidences: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict:
        """
        Run backtests on validation windows only (out-of-sample).

        Parameters
        ----------
        df : full feature DataFrame
        all_signals : signal array aligned with df
        all_confidences : confidence array aligned with df
        folds : list of (train_idx, val_idx) from WalkForwardSplitter

        Returns
        -------
        dict with per-fold results and aggregated OOS metrics
        """
        fold_results  = []
        all_oos_trades = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            logger.info(f"Backtesting fold {fold_idx+1}/{len(folds)} "
                        f"(OOS window: {len(val_idx)} candles)...")

            df_fold      = df.iloc[val_idx].reset_index(drop=False)
            sig_fold     = all_signals[val_idx]
            conf_fold    = all_confidences[val_idx]

            # Each fold starts fresh with initial equity for fair comparison
            fold_config = BacktestConfig(**{**asdict(self.base_config),
                                            "asset_name": f"{self.base_config.asset_name}_fold{fold_idx+1}"})
            engine      = VectorisedBacktester(fold_config)
            result      = engine.run(df_fold, sig_fold, conf_fold)

            fold_results.append(result)
            all_oos_trades.extend(result.trades)
            logger.info(
                f"  Fold {fold_idx+1}: {result.total_trades} trades, "
                f"WR={result.win_rate:.2%}, Sharpe={result.sharpe_ratio:.2f}, "
                f"MaxDD={result.max_drawdown_pct:.2f}%"
            )

        # Aggregate OOS metrics
        if fold_results:
            avg_wr      = np.mean([r.win_rate       for r in fold_results])
            avg_sharpe  = np.mean([r.sharpe_ratio   for r in fold_results])
            avg_sortino = np.mean([r.sortino_ratio  for r in fold_results])
            avg_pf      = np.mean([r.profit_factor  for r in fold_results
                                   if r.profit_factor != float("inf")])
            avg_maxdd   = np.mean([r.max_drawdown_pct for r in fold_results])
            total_tr    = sum([r.total_trades       for r in fold_results])

            logger.info(
                f"\n{'='*55}\n"
                f"Walk-Forward OOS Summary ({len(fold_results)} folds):\n"
                f"  Total OOS trades:     {total_tr}\n"
                f"  Avg Win Rate:         {avg_wr:.2%}\n"
                f"  Avg Sharpe Ratio:     {avg_sharpe:.3f}\n"
                f"  Avg Sortino Ratio:    {avg_sortino:.3f}\n"
                f"  Avg Profit Factor:    {avg_pf:.3f}\n"
                f"  Avg Max Drawdown:     {avg_maxdd:.2f}%\n"
                f"{'='*55}"
            )

        return {
            "fold_results":    fold_results,
            "all_oos_trades":  all_oos_trades,
            "avg_win_rate":    avg_wr    if fold_results else 0.0,
            "avg_sharpe":      avg_sharpe if fold_results else 0.0,
            "avg_sortino":     avg_sortino if fold_results else 0.0,
            "avg_profit_factor": avg_pf  if fold_results else 0.0,
            "avg_max_drawdown":avg_maxdd  if fold_results else 0.0,
            "total_oos_trades":total_tr   if fold_results else 0,
        }


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_report(result: BacktestResult, save: bool = True) -> str:
    """
    Generate a human-readable backtest report and optionally save to disk.

    Returns the report as a formatted string.
    """
    cfg = result.config
    sep = "=" * 60

    lines = [
        sep,
        f"  SynthTrade Backtest Report",
        f"  Asset: {cfg.asset_name}",
        f"  Initial Equity: ${cfg.initial_equity:,.2f}",
        f"  Final Equity:   ${result.final_equity:,.2f}",
        sep,
        "",
        "--- TRADE STATISTICS ---",
        f"  Total Trades:          {result.total_trades}",
        f"  Winning Trades:        {result.winning_trades}  ({result.win_rate:.1%})",
        f"  Losing Trades:         {result.losing_trades}",
        f"  Long Trades:           {result.long_trades}  (WR: {result.long_win_rate:.1%})",
        f"  Short Trades:          {result.short_trades}  (WR: {result.short_win_rate:.1%})",
        f"  Avg Hold (candles):    {result.avg_hold_candles:.1f}",
        f"  Signal Coverage:       {result.signal_coverage:.2%}",
        "",
        "--- PROFITABILITY ---",
        f"  Total PnL:             {result.total_pnl_pct:+.2f}%",
        f"  Profit Factor:         {result.profit_factor:.3f}",
        f"  Expectancy:            {result.expectancy_r:+.3f} R per trade",
        f"  Avg Win:               {result.avg_win_r:+.3f} R",
        f"  Avg Loss:              {result.avg_loss_r:+.3f} R",
        f"  CAGR:                  {result.cagr:+.2f}%",
        "",
        "--- RISK METRICS ---",
        f"  Max Drawdown:          {result.max_drawdown_pct:.2f}%",
        f"  Max DD Duration:       {result.max_dd_duration} candles",
        f"  Max Consec. Losses:    {result.max_consec_losses}",
        f"  Sharpe Ratio:          {result.sharpe_ratio:.3f}",
        f"  Sortino Ratio:         {result.sortino_ratio:.3f}",
        f"  Calmar Ratio:          {result.calmar_ratio:.3f}",
        "",
        "--- TRADE LOG (last 10) ---",
    ]

    for t in result.trades[-10:]:
        lines.append(
            f"  #{t.trade_id:04d} "
            f"{'LONG ' if t.signal==1 else 'SHORT'} "
            f"@ {t.entry_price:.5f} -> {t.exit_price:.5f} "
            f"[{t.exit_reason}] "
            f"PnL: {t.pnl_r:+.2f}R  conf={t.confidence:.3f}"
        )

    lines.append(sep)
    report = "\n".join(lines)

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        report_path = os.path.join(RESULTS_DIR, f"report_{cfg.asset_name}.txt")
        with open(report_path, "w") as f:
            f.write(report)

        # Also save trade log as CSV for further analysis
        if result.trades:
            trades_df = pd.DataFrame([asdict(t) for t in result.trades])
            csv_path  = os.path.join(RESULTS_DIR, f"trades_{cfg.asset_name}.csv")
            trades_df.to_csv(csv_path, index=False)
            logger.info(f"Trade log saved to {csv_path}")

        # Save equity curve
        eq_df = pd.DataFrame({
            "timestamp": result.equity_timestamps,
            "equity":    result.equity_curve
        })
        eq_path = os.path.join(RESULTS_DIR, f"equity_{cfg.asset_name}.csv")
        eq_df.to_csv(eq_path, index=False)

        logger.info(f"Report saved to {report_path}")

    print(report)
    return report


def save_result_json(result: BacktestResult, asset_name: str):
    """Save key backtest metrics as JSON for the dashboard to read."""
    summary = {
        "asset":           asset_name,
        "total_trades":    result.total_trades,
        "win_rate":        result.win_rate,
        "profit_factor":   result.profit_factor,
        "expectancy_r":    result.expectancy_r,
        "total_pnl_pct":   result.total_pnl_pct,
        "max_drawdown_pct":result.max_drawdown_pct,
        "sharpe_ratio":    result.sharpe_ratio,
        "sortino_ratio":   result.sortino_ratio,
        "calmar_ratio":    result.calmar_ratio,
        "cagr":            result.cagr,
        "final_equity":    result.final_equity,
        "max_consec_losses":result.max_consec_losses,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"summary_{asset_name}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary JSON saved to {path}")
    return summary


# =============================================================================
# REGIME PERFORMANCE ANALYSER
# Breaks down performance by market regime (trending vs ranging)
# and by time-of-day to identify where the edge is strongest.
# This feeds directly into the self-improvement loop.
# =============================================================================

def analyse_regime_performance(
    result: BacktestResult,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Break down trade performance by market regime.

    Uses bb_regime (1=trending, 0=ranging) and trend_score from the
    feature DataFrame to categorise each trade's market condition.

    Returns a DataFrame with win rate and expectancy per regime.
    """
    if not result.trades:
        return pd.DataFrame()

    records = []
    for t in result.trades:
        idx = t.entry_idx
        if idx >= len(df):
            continue

        regime     = int(df["bb_regime"].iloc[idx])   if "bb_regime"   in df.columns else -1
        trend_score= int(df["trend_score"].iloc[idx]) if "trend_score" in df.columns else 0

        records.append({
            "trade_id":   t.trade_id,
            "signal":     t.signal,
            "regime":     "trending" if regime == 1 else "ranging",
            "trend_score":trend_score,
            "pnl_r":      t.pnl_r,
            "win":        1 if t.pnl_r > 0 else 0,
            "confidence": t.confidence,
        })

    df_trades = pd.DataFrame(records)
    if df_trades.empty:
        return df_trades

    breakdown = df_trades.groupby("regime").agg(
        n_trades   =("pnl_r", "count"),
        win_rate   =("win", "mean"),
        avg_pnl_r  =("pnl_r", "mean"),
        total_pnl_r=("pnl_r", "sum"),
    ).round(4)

    logger.info(f"\nRegime performance breakdown:\n{breakdown}")
    return breakdown


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import pandas as pd
    from features.pipeline import run_pipeline
    from features.dataset import prepare_dataset
    from models.cnn_lstm import build_cnn_lstm, train_model, batch_predict

    print("\n=== SynthTrade Backtester Test ===\n")

    # Generate synthetic trending data
    np.random.seed(99)
    n = 1200
    # Simulate a trending market with some mean reversion
    trend = np.cumsum(np.random.randn(n) * 15 + 0.5)
    price = 19000 + trend

    df_raw = pd.DataFrame({
        "open":   price + np.random.randn(n) * 8,
        "close":  price + np.random.randn(n) * 8,
        "volume": np.abs(np.random.randn(n) * 1000 + 3000),
    }, index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"))
    df_raw["high"] = df_raw[["open","close"]].max(axis=1) + np.abs(np.random.randn(n))*12
    df_raw["low"]  = df_raw[["open","close"]].min(axis=1) - np.abs(np.random.randn(n))*12

    # Run pipeline and train a quick model
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    pipe    = run_pipeline(df_raw, asset_name="BACKTEST_TEST", save_normaliser=False)
    dataset = prepare_dataset(pipe["X"], pipe["y"], asset_name="BACKTEST_TEST",
                               resample="moderate", n_folds=3, save_to_disk=False)

    model = build_cnn_lstm(input_shape=dataset["shape"], dropout_rate=0.25)
    train_model(
        model,
        dataset["X_train"], dataset["y_train_encoded"],
        dataset["X_val"],   dataset["y_val_encoded"],
        class_weights=dataset["class_weights"],
        asset_name="BACKTEST_TEST",
        epochs=8, batch_size=32, save_best=False
    )

    # Generate signals for the full dataset
    from models.cnn_lstm import batch_predict
    import tensorflow as tf

    probs = model.predict(pipe["X"], verbose=0)
    sigs  = batch_predict(model, pipe["X"], threshold=0.55)
    confs = probs.max(axis=1)

    print(f"\nSignals: LONG={(sigs==1).sum()}, "
          f"SHORT={(sigs==-1).sum()}, NO_TRADE={(sigs==0).sum()}")

    # Align df_features with signals
    df_feat = pipe["df_features"].iloc[30:].copy()   # trim for sequence offset
    df_feat = df_feat.iloc[:len(sigs)].copy()

    # Run backtest
    config = BacktestConfig(
        asset_name="BACKTEST_TEST",
        initial_equity=10_000,
        risk_per_trade_pct=1.0,
        sl_atr_mult=1.0,
        tp_atr_mult=1.5,
        max_hold_candles=15,
        confidence_threshold=0.55,
        spread_pct=0.01,
    )
    engine = VectorisedBacktester(config)
    result = engine.run(df_feat, sigs, confs)

    print(f"\nBacktest complete:")
    print(f"  Total trades: {result.total_trades}")
    print(f"  Win rate:     {result.win_rate:.2%}")
    print(f"  Sharpe:       {result.sharpe_ratio:.3f}")
    print(f"  Sortino:      {result.sortino_ratio:.3f}")
    print(f"  Max DD:       {result.max_drawdown_pct:.2f}%")
    print(f"  Profit Factor:{result.profit_factor:.3f}")
    print(f"  Expectancy:   {result.expectancy_r:+.3f} R")
    print(f"  Total PnL:    {result.total_pnl_pct:+.2f}%")
    print(f"  Final Equity: ${result.final_equity:,.2f}")

    generate_report(result, save=False)

    regime_df = analyse_regime_performance(result, df_feat)
    if not regime_df.empty:
        print(f"\nRegime breakdown:\n{regime_df}")

    print("\n=== Phase 5 Backtesting Engine: ALL SYSTEMS OK ===")
