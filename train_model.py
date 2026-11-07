#!/usr/bin/env python3
"""
Model Training Script for Coca-Cola ($KO) Cross-Asset Strategy.
Strictly trains on data up to 2023-12-31, reserving 2024+ for out-of-sample testing.
"""

import argparse
from pathlib import Path
import pandas as pd
import yaml

from src.data.loader import DataLoader
from src.data.universe import AssetUniverse
from src.features.engineer import FeatureEngineer
from src.features.labeling import TargetLabeler
from src.models.trainer import ModelTrainer


def train(
    config_path: str = "config/default_config.yaml",
    cutoff_date: str = "2023-12-31",
    model_output_path: str = "models/ko_model_up_to_2023.joblib",
):
    print("=" * 70)
    print("  COCA-COLA ($KO) MODEL TRAINING (DATA <= 2023-12-31)")
    print("=" * 70)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Force strict end date to cutoff_date
    config["data"]["end_date"] = cutoff_date
    universe = AssetUniverse.from_config(config)

    print(f"[*] Step 1: Ingesting training data up to {cutoff_date}...")
    loader = DataLoader(
        universe=universe,
        cache_dir=config.get("data", {}).get("cache_dir", "data/cache"),
        start_date=config.get("data", {}).get("start_date", "2019-01-01"),
        end_date=cutoff_date,
    )

    # Fetch training slice
    prices = loader.fetch_data()
    # Hard clamp to ensure zero data past cutoff
    prices = prices[prices.index <= pd.to_datetime(cutoff_date)]

    print(f"    Training dataset span: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"    Total trading days in training set: {len(prices)} bars")
    assert prices.index.max() <= pd.to_datetime(cutoff_date), f"Data leak! Found date {prices.index.max()} > {cutoff_date}"

    print("[*] Step 2: Engineering cross-asset competitor & cointegration features...")
    feat_cfg = config.get("features", {})
    engineer = FeatureEngineer(
        universe=universe,
        rolling_windows=feat_cfg.get("rolling_windows", [5, 10, 21, 63]),
        lead_lag_lags=feat_cfg.get("lead_lag_lags", [1, 2, 3]),
        cointegration_pairs=feat_cfg.get("cointegration_pairs", [["KO", "PEP"], ["KO", "XLP"]]),
        pca_components=feat_cfg.get("pca", {}).get("n_components", 3),
    )
    X = engineer.compute_features(prices)

    print("[*] Step 3: Computing supervisory relative alpha labels (KO vs XLP)...")
    label_cfg = config.get("labeling", {})
    labeler = TargetLabeler(
        forward_horizon=label_cfg.get("forward_horizon", 5),
        neutral_threshold=label_cfg.get("neutral_threshold", 0.0015),
        target_type=label_cfg.get("target_type", "relative_alpha"),
    )
    y, _ = labeler.create_labels(prices, target_ticker="KO", benchmark_ticker="XLP")

    # Align valid training rows
    common_idx = X.index.intersection(y.dropna().index)
    X_train = X.loc[common_idx]
    y_train = y.loc[common_idx]

    print(f"    Aligned training samples: {len(X_train)} rows x {X_train.shape[1]} features")

    print("[*] Step 4: Fitting model and optimizing decision weights...")
    trainer = ModelTrainer(config)
    importance_df = trainer.fit_production(X_train, y_train)

    print("\n" + "=" * 70)
    print("  TOP 12 LEARNED FEATURE WEIGHTS / IMPORTANCES (2019 - 2023)")
    print("=" * 70)
    for idx, row in importance_df.head(12).iterrows():
        print(f"  #{idx+1:<2d} {row['feature']:40s} | Weight / Importance: {row['importance']:.4f}")

    print("\n[*] Step 5: Serializing learned model checkpoint to disk...")
    trainer.save_model(model_output_path)
    print(f"    Checkpoint saved to {model_output_path}")
    print("=" * 70)
    print("  TRAINING COMPLETE! 2024+ DATA HAS BEEN FULLY PRESERVED FOR TESTING.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train Coca-Cola ML Model with strict cutoff date.")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config file")
    parser.add_argument("--cutoff-date", type=str, default="2023-12-31", help="Cutoff date for training")
    parser.add_argument("--output", type=str, default="models/ko_model_up_to_2023.joblib", help="Output model path")
    args = parser.parse_args()

    train(config_path=args.config, cutoff_date=args.cutoff_date, model_output_path=args.output)


if __name__ == "__main__":
    main()
