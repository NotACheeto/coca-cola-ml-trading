#!/usr/bin/env python3
"""
Out-of-Sample Testing Script for Coca-Cola ML Strategy (2024 - Present).
Evaluates models trained exclusively on data <= 2023 on unseen future market periods.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.backtest.analytics import PerformanceAnalytics
from src.backtest.engine import BacktestEngine
from src.data.loader import DataLoader
from src.data.universe import AssetUniverse
from src.features.engineer import FeatureEngineer
from src.models.trainer import ModelTrainer


def test_out_of_sample(
    config_path: str = "config/default_config.yaml",
    model_path: str = "models/ko_model_up_to_2023.joblib",
    test_start_date: str = "2024-01-01",
    output_plot: str = "results/out_of_sample_2024_2026.png",
):
    print("=" * 70)
    print(f"  OUT-OF-SAMPLE TESTING ON RESERVED DATA ({test_start_date} TO PRESENT)")
    print("=" * 70)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    universe = AssetUniverse.from_config(config)

    # 1. Load trained model
    print(f"[*] Step 1: Loading pre-trained model checkpoint from {model_path}...")
    trainer = ModelTrainer(config)
    trainer.load_model(model_path)

    # 2. Ingest data including 2024+ (using 2023 for feature warm-up)
    print(f"[*] Step 2: Ingesting reserved test data ({test_start_date} onwards)...")
    loader = DataLoader(
        universe=universe,
        cache_dir=config.get("data", {}).get("cache_dir", "data/cache"),
        start_date="2023-01-01",  # Warm-up period for 63-day rolling indicators
        end_date=None,             # Fetch up to current date
    )
    prices = loader.fetch_data(use_cache=False)

    print(f"    Total data available: {prices.index[0].date()} to {prices.index[-1].date()} ({len(prices)} bars)")

    # 3. Engineer features
    print("[*] Step 3: Generating features on unseen data...")
    feat_cfg = config.get("features", {})
    engineer = FeatureEngineer(
        universe=universe,
        rolling_windows=feat_cfg.get("rolling_windows", [5, 10, 21, 63]),
        lead_lag_lags=feat_cfg.get("lead_lag_lags", [1, 2, 3]),
        cointegration_pairs=feat_cfg.get("cointegration_pairs", [["KO", "PEP"], ["KO", "XLP"], ["KO", "COKE"], ["KO", "TLT"]]),
        pca_components=feat_cfg.get("pca", {}).get("n_components", 3),
    )
    X = engineer.compute_features(prices)

    # Filter strictly to the test period (>= 2024-01-01)
    test_mask = X.index >= pd.to_datetime(test_start_date)
    X_test = X[test_mask]

    print(f"    Out-of-sample evaluation dataset: {X_test.index[0].date()} to {X_test.index[-1].date()}")
    print(f"    Test sample count: {len(X_test)} trading days")

    # 4. Generate Predictions using the model trained <= 2023
    print("[*] Step 4: Generating out-of-sample predictions and trade signals...")
    X_test_scaled = pd.DataFrame(
        trainer.scaler.transform(X_test[trainer.feature_names]),
        columns=trainer.feature_names,
        index=X_test.index,
    )
    test_probs = pd.Series(
        trainer.fitted_model.predict_proba(X_test_scaled)[:, 1],
        index=X_test.index,
    )

    # 5. Run Backtest on the 2024+ unseen test data
    print("[*] Step 5: Simulating out-of-sample trading execution with costs...")
    bt_cfg = config.get("backtest", {})
    engine = BacktestEngine(
        initial_capital=bt_cfg.get("initial_capital", 100000.0),
        transaction_cost_bps=bt_cfg.get("transaction_cost_bps", 5.0),
        position_sizing=bt_cfg.get("position_sizing", "confidence_weighted"),
    )
    backtest_results = engine.run(
        prices=prices,
        signals=test_probs,
        target_ticker="KO",
        benchmark_ticker="XLP",
    )

    # 6. Performance Comparison
    print("\n" + "=" * 70)
    print(f"  OUT-OF-SAMPLE PERFORMANCE COMPARISON ({test_start_date} TO {prices.index[-1].date()})")
    print("=" * 70)
    comparison = PerformanceAnalytics.generate_comparison_table(backtest_results)
    map_func = getattr(comparison, "map", getattr(comparison, "applymap", None))
    pd_table = map_func(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else str(x))
    print(pd_table.to_string())

    # 7. Save Tear Sheet
    PerformanceAnalytics.plot_tear_sheet(backtest_results, output_path=output_plot)
    print(f"\n[*] Out-of-sample performance tear sheet saved to {output_plot}")
    print("=" * 70)

    # 8. Show Latest Live Signal (for today/tomorrow)
    latest_bar = X.iloc[[-1]]
    latest_decision = trainer.predict_signal(latest_bar)
    print("\n" + "=" * 70)
    print("  LATEST LIVE TRADING DECISION FOR TOMORROW")
    print("=" * 70)
    for k, v in latest_decision.items():
        print(f"  {k:25s}: {v}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test Coca-Cola Model on reserved 2024+ data.")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default="models/ko_model_up_to_2023.joblib", help="Path to trained model")
    parser.add_argument("--test-start", type=str, default="2024-01-01", help="Start date of reserved test period")
    parser.add_argument("--output-plot", type=str, default="results/out_of_sample_2024_2026.png", help="Output plot path")
    args = parser.parse_args()

    test_out_of_sample(
        config_path=args.config,
        model_path=args.model,
        test_start_date=args.test_start,
        output_plot=args.output_plot,
    )


if __name__ == "__main__":
    main()
