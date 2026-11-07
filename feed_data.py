#!/usr/bin/env python3
"""
Interactive Data Feeder and Adaptive Model Weight Updater.
Allows feeding live market data, custom CSV files, or simulating streaming data to adjust model weights.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from src.data.loader import DataLoader
from src.data.universe import AssetUniverse
from src.features.engineer import FeatureEngineer
from src.features.labeling import TargetLabeler
from src.models.trainer import ModelTrainer


def feed_and_update(
    config_path: str = "config/default_config.yaml",
    model_path: str = "models/ko_model_up_to_2023.joblib",
    csv_file: str = None,
    live_update: bool = False,
    rolling_window: int = 500,
    save_updated_model: str = "models/ko_model_updated.joblib",
):
    print("=" * 70)
    print("  FEEDING NEW DATA & ADAPTING MACHINE LEARNING WEIGHTS")
    print("=" * 70)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    universe = AssetUniverse.from_config(config)
    loader = DataLoader(
        universe=universe,
        cache_dir=config.get("data", {}).get("cache_dir", "data/cache"),
        start_date=config.get("data", {}).get("start_date", "2019-01-01"),
        end_date=None,
    )

    # 1. Ingest Data (either from CSV, live delta update, or cache)
    if csv_file:
        print(f"[*] Feeding custom data from CSV file: {csv_file}")
        prices = loader.load_from_csv(csv_file)
        print(f"    Loaded custom data up to {prices.index[-1].date()} ({len(prices)} total bars)")
    elif live_update:
        print("[*] Fetching live market updates from online sources...")
        prices, new_bars = loader.update_latest()
        print(f"    Ingested {new_bars} new daily market bars. Latest date: {prices.index[-1].date()}")
    else:
        print("[*] Loading existing market data from cache...")
        prices = loader.fetch_data(use_cache=True)
        print(f"    Current dataset date span: {prices.index[0].date()} to {prices.index[-1].date()}")

    # 2. Compute Updated Feature Matrix
    print("[*] Step 2: Computing updated cross-asset features and spreads...")
    feat_cfg = config.get("features", {})
    engineer = FeatureEngineer(
        universe=universe,
        rolling_windows=feat_cfg.get("rolling_windows", [5, 10, 21, 63]),
        lead_lag_lags=feat_cfg.get("lead_lag_lags", [1, 2, 3]),
        cointegration_pairs=feat_cfg.get("cointegration_pairs", [["KO", "PEP"], ["KO", "XLP"], ["KO", "COKE"], ["KO", "TLT"]]),
        pca_components=feat_cfg.get("pca", {}).get("n_components", 3),
    )
    X = engineer.compute_features(prices)

    # 3. Compute Supervisory Target Labels
    print("[*] Step 3: Computing supervisory relative alpha targets...")
    labeler = TargetLabeler(
        forward_horizon=config.get("labeling", {}).get("forward_horizon", 5),
        neutral_threshold=config.get("labeling", {}).get("neutral_threshold", 0.0015),
    )
    y, _ = labeler.create_labels(prices, target_ticker="KO", benchmark_ticker="XLP")

    # Align training samples (ignoring the last `forward_horizon` unlabelled bars)
    valid_idx = X.index.intersection(y.dropna().index)
    X_train = X.loc[valid_idx]
    y_train = y.loc[valid_idx]

    # Optional: Apply rolling window (e.g. trailing 500 days ~ 2 years) to adapt to recent regimes
    if rolling_window and len(X_train) > rolling_window:
        print(f"[*] Using trailing {rolling_window} trading days for adaptive regime learning...")
        X_train = X_train.iloc[-rolling_window:]
        y_train = y_train.iloc[-rolling_window:]

    # 4. Load Previous Model to Compare Weight Adjustments
    previous_weights = None
    trainer = ModelTrainer(config)
    if Path(model_path).exists():
        try:
            trainer.load_model(model_path)
            if hasattr(trainer.fitted_model, "feature_importances_"):
                previous_weights = dict(zip(trainer.feature_names, trainer.fitted_model.feature_importances_))
            elif hasattr(trainer.fitted_model, "coef_"):
                previous_weights = dict(zip(trainer.feature_names, np.abs(trainer.fitted_model.coef_[0])))
        except Exception:
            pass

    # 5. Fit Model & Adjust Weights on Newly Fed Data
    print(f"[*] Step 5: Updating model weights on {len(X_train)} historical samples...")
    new_importance_df = trainer.fit_production(X_train, y_train)

    # 6. Display How Weights Were Adjusted
    print("\n" + "=" * 70)
    print("  HOW MODEL WEIGHTS ADJUSTED WITH THE NEW DATA")
    print("=" * 70)
    print(f"  {'Feature Name':40s} | {'New Weight':10s} | {'Previous':10s} | {'Shift':8s}")
    print("  " + "-" * 68)

    for idx, row in new_importance_df.head(10).iterrows():
        feat = row["feature"]
        new_w = row["importance"]
        old_w = previous_weights.get(feat, 0.0) if previous_weights else 0.0
        delta = new_w - old_w
        shift_str = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
        print(f"  {feat:40s} | {new_w:10.2f} | {old_w:10.2f} | {shift_str:8s}")

    # 7. Save Updated Model Checkpoint
    trainer.save_model(save_updated_model)

    # 8. Produce Live Trade Signal for Tomorrow
    latest_bar = X.iloc[[-1]]
    decision = trainer.predict_signal(latest_bar)

    print("\n" + "=" * 70)
    print(f"  LATEST LIVE TRADE RECOMMENDATION (FOR {decision['date']})")
    print("=" * 70)
    print(f"  Target Asset              : Coca-Cola ($KO)")
    print(f"  Model Probability (Alpha) : {decision['probability_outperform']:.4f}")
    print(f"  Signal Confidence         : {decision['confidence'] * 100:.1f}%")
    print(f"  Recommended Action        : {decision['action']}")
    print(f"  Suggested Position Weight : {decision['recommended_weight'] * 100:.1f}%")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Feed new data and adjust machine learning model weights.")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default="models/ko_model_up_to_2023.joblib", help="Path to base model")
    parser.add_argument("--csv", type=str, default=None, help="Path to custom CSV price file")
    parser.add_argument("--live", action="store_true", help="Fetch newest daily bars online")
    parser.add_argument("--rolling-window", type=int, default=500, help="Trailing window of days for retraining")
    parser.add_argument("--output", type=str, default="models/ko_model_updated.joblib", help="Output model path")
    args = parser.parse_args()

    feed_and_update(
        config_path=args.config,
        model_path=args.model,
        csv_file=args.csv,
        live_update=args.live,
        rolling_window=args.rolling_window,
        save_updated_model=args.output,
    )


if __name__ == "__main__":
    main()
