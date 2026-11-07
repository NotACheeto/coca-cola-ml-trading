"""
End-to-End Quantitative Trading Pipeline for Coca-Cola ML Strategy.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import yaml

from src.backtest.analytics import PerformanceAnalytics
from src.backtest.engine import BacktestEngine
from src.data.loader import DataLoader
from src.data.universe import AssetUniverse
from src.features.engineer import FeatureEngineer
from src.features.labeling import TargetLabeler
from src.models.evaluator import ModelEvaluator
from src.models.trainer import ModelTrainer


class TradingPipeline:
    """Orchestrates data ingestion, feature generation, ML training, and backtesting."""

    def __init__(self, config_path: str = "config/default_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.universe = AssetUniverse.from_config(self.config)
        self.data_loader = DataLoader(
            universe=self.universe,
            cache_dir=self.config.get("data", {}).get("cache_dir", "data/cache"),
            start_date=self.config.get("data", {}).get("start_date", "2019-01-01"),
            end_date=self.config.get("data", {}).get("end_date", None),
            bars_per_day=self.config.get("data", {}).get("bars_per_day", 3),
        )

        feat_cfg = self.config.get("features", {})
        self.feature_engineer = FeatureEngineer(
            universe=self.universe,
            rolling_windows=feat_cfg.get("rolling_windows", [15, 30, 63, 189]),
            lead_lag_lags=feat_cfg.get("lead_lag_lags", [1, 2, 3]),
            cointegration_pairs=feat_cfg.get("cointegration_pairs", [["KO", "PEP"], ["KO", "XLP"], ["KO", "COKE"], ["KO", "TLT"]]),
            pca_components=feat_cfg.get("pca", {}).get("n_components", 3),
        )

        label_cfg = self.config.get("labeling", {})
        self.labeler = TargetLabeler(
            forward_horizon=label_cfg.get("forward_horizon", 15),
            neutral_threshold=label_cfg.get("neutral_threshold", 0.0015),
            target_type=label_cfg.get("target_type", "relative_alpha"),
        )

        self.trainer = ModelTrainer(self.config)

        bt_cfg = self.config.get("backtest", {})
        self.backtester = BacktestEngine(
            initial_capital=bt_cfg.get("initial_capital", 100000.0),
            transaction_cost_bps=bt_cfg.get("transaction_cost_bps", 5.0),
            position_sizing=bt_cfg.get("position_sizing", "confidence_weighted"),
            target_volatility=bt_cfg.get("target_volatility", 0.12),
        )

    def run(self, save_plot_path: Optional[str] = "results/tear_sheet.png") -> Dict[str, any]:
        """
        Executes complete pipeline: Data -> Features -> Model -> Backtest -> Analytics.
        """
        print(f"[*] Step 1: Ingesting asset data for universe: {self.universe.all_tickers}")
        prices = self.data_loader.fetch_data()
        target_ohlcv = self.data_loader.fetch_target_ohlcv()
        print(f"    Loaded {len(prices)} rows across {len(prices.columns)} assets from {prices.index[0].date()} to {prices.index[-1].date()}")

        print("[*] Step 2: Engineering 200+ cross-asset, commodity, rate, and microstructure features")
        X = self.feature_engineer.compute_features(prices, ohlcv=target_ohlcv)
        print(f"    Created {X.shape[1]} engineered features")

        print("[*] Step 3: Generating forward relative alpha supervisory targets")
        y, fwd_ret = self.labeler.create_labels(prices, target_ticker=self.universe.target, benchmark_ticker="XLP")

        # Intersect indices
        valid_idx = X.index.intersection(y.dropna().index)
        X_clean = X.loc[valid_idx]
        y_clean = y.loc[valid_idx]
        fwd_ret_clean = fwd_ret.loc[valid_idx]

        print(f"[*] Step 4: Training {self.config.get('model', {}).get('type')} with Purged Walk-Forward CV")
        oof_probs, oof_targets, feature_importance = self.trainer.train_walk_forward(X_clean, y_clean)

        print("[*] Step 5: Evaluating model predictability")
        eval_metrics = ModelEvaluator.evaluate(
            y_true=oof_targets,
            y_prob=oof_probs,
            forward_returns=fwd_ret_clean,
        )

        print("[*] Step 6: Simulating strategy execution & backtest")
        backtest_df = self.backtester.run(
            prices=prices,
            signals=oof_probs,
            target_ticker=self.universe.target,
            benchmark_ticker="XLP",
        )

        print("[*] Step 7: Computing portfolio analytics & tear-sheet")
        comparison_table = PerformanceAnalytics.generate_comparison_table(backtest_df)

        if save_plot_path:
            PerformanceAnalytics.plot_tear_sheet(backtest_df, output_path=save_plot_path)
            print(f"    Saved visual performance tear sheet to {save_plot_path}")

        return {
            "evaluation_metrics": eval_metrics,
            "feature_importance": feature_importance,
            "comparison_table": comparison_table,
            "backtest_results": backtest_df,
        }
