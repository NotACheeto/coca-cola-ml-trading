#!/usr/bin/env python3
"""
CLI entry point to execute the Coca-Cola ML quantitative trading pipeline.
"""

import argparse
import sys
from src.pipeline import TradingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run Machine Learning Trading Strategy for Coca-Cola ($KO) based on competitor dynamics."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="results/tear_sheet.png",
        help="Destination path for performance tear sheet image",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  COCA-COLA ($KO) CROSS-ASSET MACHINE LEARNING TRADING SYSTEM")
    print("=" * 70)

    pipeline = TradingPipeline(config_path=args.config)
    results = pipeline.run(save_plot_path=args.output_plot)

    print("\n" + "=" * 70)
    print("  1. MODEL EVALUATION METRICS (OUT-OF-FOLD / PURGED)")
    print("=" * 70)
    for k, v in results["evaluation_metrics"].items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:10.4f}")
        else:
            print(f"  {k:30s}: {v}")

    print("\n" + "=" * 70)
    print("  2. TOP 10 PREDICTIVE FEATURES (CROSS-ASSET DYNAMICS)")
    print("=" * 70)
    feat_df = results["feature_importance"].head(10)
    for idx, row in feat_df.iterrows():
        print(f"  #{idx+1:<2d} {row['feature']:40s} | Importance: {row['importance']:.4f}")

    print("\n" + "=" * 70)
    print("  3. STRATEGY VS BENCHMARKS PERFORMANCE COMPARISON")
    print("=" * 70)
    comparison = results["comparison_table"]
    # Nicely formatted table
    map_func = getattr(comparison, "map", getattr(comparison, "applymap", None))
    pd_table = map_func(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else str(x))
    print(pd_table.to_string())

    print("\n" + "=" * 70)
    print(f"  Backtest completed successfully. Tear sheet saved to {args.output_plot}")
    print("=" * 70)


if __name__ == "__main__":
    main()
