# =============================================================================
# SynthTrade - Main Entry Point
# =============================================================================
# Usage:
#   python main.py --mode train   --asset US100    # Train model for one asset
#   python main.py --mode train   --asset ALL      # Train all active assets
#   python main.py --mode backtest --asset US100   # Backtest trained model
#   python main.py --mode signal  --asset US100    # Generate live signals (once)
#   python main.py --mode dashboard                # Launch Streamlit dashboard
#
# Self-improvement check runs automatically after every backtest.

import os, sys, argparse, asyncio, logging
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from config import (
    ACTIVE_ASSETS, REAL_ASSETS, SYNTHETIC_ASSETS,
    PRIMARY_TF, MODELS_DIR, RESULTS_DIR,
    SL_ATR_MULTIPLE, TP_ATR_MULTIPLE, SIGNAL_CONFIDENCE_THRESHOLD
)
from data.ingestion.market_data  import fetch_ohlcv, fetch_all_real_assets
from data.ingestion.deriv_ws     import fetch_synthetic_candles
from features.pipeline           import run_pipeline
from features.dataset            import prepare_dataset
from models.cnn_lstm             import (
    build_cnn_lstm, train_model, batch_predict,
    save_model, load_model, evaluate_model
)
from backtest.engine             import (
    VectorisedBacktester, BacktestConfig,
    generate_report, save_result_json
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Main")


def is_synthetic(asset: str) -> bool:
    return asset in SYNTHETIC_ASSETS


async def get_data(asset: str, interval: str = PRIMARY_TF,
                   n_candles: int = 2000) -> pd.DataFrame:
    """Fetch data for any asset, routing to correct source."""
    if is_synthetic(asset):
        df = await fetch_synthetic_candles(asset, interval=interval, count=n_candles)
    else:
        df = fetch_ohlcv(asset, interval=interval, n_candles=n_candles)
    return df


def train_asset(asset: str, epochs: int = 80, resample: str = "moderate"):
    """Full train pipeline for one asset."""
    logger.info(f"\n{'='*55}\nTraining: {asset}\n{'='*55}")

    # Fetch data
    df_raw = asyncio.run(get_data(asset))
    if df_raw is None or len(df_raw) < 200:
        logger.error(f"Insufficient data for {asset}. Skipping.")
        return None

    # Feature pipeline
    pipe = run_pipeline(df_raw, asset_name=asset, save_normaliser=True)
    if pipe["X"] is None or len(pipe["X"]) < 100:
        logger.error(f"Insufficient sequences for {asset}. Skipping.")
        return None

    # Dataset preparation
    dataset = prepare_dataset(
        pipe["X"], pipe["y"],
        asset_name=asset,
        resample=resample,
        n_folds=5,
        save_to_disk=True
    )

    # Build and train model
    model = build_cnn_lstm(
        input_shape=dataset["shape"],
        dropout_rate=0.3,
        learning_rate=1e-3
    )
    train_model(
        model,
        dataset["X_train"], dataset["y_train_encoded"],
        dataset["X_val"],   dataset["y_val_encoded"],
        class_weights=dataset["class_weights"],
        asset_name=asset,
        epochs=epochs,
        batch_size=32,
        save_best=True
    )

    # Evaluate on held-out test set
    eval_r = evaluate_model(
        model, dataset["X_test"], dataset["y_test_encoded"],
        asset_name=asset
    )

    # Save final model
    save_model(model, asset)
    logger.info(f"[{asset}] Training complete. "
                f"Signal accuracy: {eval_r['signal_accuracy']:.2%}, "
                f"Coverage: {eval_r['coverage']:.2%}")
    return model


def backtest_asset(asset: str):
    """Run backtest for a trained asset model."""
    logger.info(f"\n{'='*55}\nBacktesting: {asset}\n{'='*55}")

    # Load model
    model = load_model(asset)
    if model is None:
        logger.error(f"No trained model found for {asset}. Run --mode train first.")
        return

    # Fetch fresh data for backtest
    df_raw = asyncio.run(get_data(asset, n_candles=3000))
    if df_raw is None:
        logger.error(f"Could not fetch data for {asset}.")
        return

    # Run pipeline (use saved normaliser, no refit)
    import joblib
    norm_path = os.path.join(MODELS_DIR, f"normaliser_{asset}.pkl")
    from features.pipeline import FeatureNormaliser
    normaliser = FeatureNormaliser()
    if os.path.exists(norm_path):
        normaliser.load(norm_path)
        pipe = run_pipeline(df_raw, asset_name=asset,
                            normaliser=normaliser,
                            fit_normaliser=False,
                            save_normaliser=False)
    else:
        pipe = run_pipeline(df_raw, asset_name=asset, save_normaliser=False)

    if pipe["X"] is None:
        logger.error(f"Pipeline produced no sequences for {asset}.")
        return

    # Generate signals
    probs  = model.predict(pipe["X"], verbose=0)
    sigs   = batch_predict(model, pipe["X"], threshold=SIGNAL_CONFIDENCE_THRESHOLD)
    confs  = probs.max(axis=1)

    logger.info(f"[{asset}] Signals: LONG={(sigs==1).sum()}, "
                f"SHORT={(sigs==-1).sum()}, NO_TRADE={(sigs==0).sum()}")

    # Align feature df with sequences
    seq_offset = 30   # SEQUENCE_LENGTH
    df_feat = pipe["df_features"].iloc[seq_offset:].copy()
    df_feat = df_feat.iloc[:len(sigs)].copy()

    # Spread config per asset type
    spread = 0.0 if is_synthetic(asset) else 0.02

    config = BacktestConfig(
        asset_name          = asset,
        initial_equity      = 10_000,
        risk_per_trade_pct  = 1.0,
        sl_atr_mult         = SL_ATR_MULTIPLE,
        tp_atr_mult         = TP_ATR_MULTIPLE,
        max_hold_candles    = 20,
        spread_pct          = spread,
        confidence_threshold= SIGNAL_CONFIDENCE_THRESHOLD,
    )
    engine = VectorisedBacktester(config)
    result = engine.run(df_feat, sigs, confs)

    generate_report(result, save=True)
    save_result_json(result, asset)

    return result


def main():
    parser = argparse.ArgumentParser(description="SynthTrade - Trading Signal System")
    parser.add_argument("--mode",   choices=["train", "backtest", "signal", "dashboard"],
                        default="dashboard")
    parser.add_argument("--asset",  default="ALL",
                        help="Asset name or ALL for all active assets")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    if args.mode == "dashboard":
        logger.info("Launching Streamlit dashboard...")
        os.system("streamlit run dashboard/app.py")
        return

    assets = ACTIVE_ASSETS if args.asset == "ALL" else [args.asset]

    if args.mode == "train":
        for asset in assets:
            train_asset(asset, epochs=args.epochs)

    elif args.mode == "backtest":
        for asset in assets:
            backtest_asset(asset)

    elif args.mode == "signal":
        for asset in assets:
            model = load_model(asset)
            if model is None:
                logger.warning(f"No model for {asset}. Train first.")
                continue
            df_raw = asyncio.run(get_data(asset, n_candles=200))
            if df_raw is None:
                continue
            pipe = run_pipeline(df_raw, asset_name=asset,
                                generate_label=False, save_normaliser=False)
            if pipe["X"] is None:
                continue
            from models.cnn_lstm import predict_signal
            result = predict_signal(model, pipe["X"][-1])
            logger.info(
                f"[{asset}] Signal: {result['signal_name']} | "
                f"Confidence: {result['confidence']:.3f} | "
                f"Above threshold: {result['above_threshold']}"
            )


if __name__ == "__main__":
    main()
