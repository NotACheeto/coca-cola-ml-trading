#!/usr/bin/env python3
"""
Simulate an exact 1-year trading period and estimate total institutional PnL.
Evaluates out-of-sample execution lag, 5 bps transaction costs, and 4.5% cash sweep yield.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from src.backtest.analytics import PerformanceAnalytics
from src.backtest.engine import BacktestEngine
from src.data.loader import DataLoader
from src.data.universe import AssetUniverse
from src.features.engineer import FeatureEngineer
from src.models.trainer import ModelTrainer


def simulate_year(
    config_path: str = "config/default_config.yaml",
    model_path: str = "models/ko_model_up_to_2023.joblib",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    initial_capital: float = 100000.0,
):
    print("=" * 75)
    print(f"  COCA-COLA ($KO) 1-YEAR QUANTITATIVE SIMULATION ({start_date} TO {end_date})")
    print("=" * 75)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    universe = AssetUniverse.from_config(config)

    # 1. Load trained model
    print(f"[*] Loading production model checkpoint from {model_path}...")
    trainer = ModelTrainer(config)
    trainer.load_model(model_path)

    # 2. Ingest market data including warm-up period
    print(f"[*] Ingesting data with 1-year warm-up (starting 2023-01-01)...")
    loader = DataLoader(
        universe=universe,
        cache_dir=config.get("data", {}).get("cache_dir", "data/cache"),
        start_date="2023-01-01",
        end_date=None,
        bars_per_day=config.get("data", {}).get("bars_per_day", 3),
    )
    prices = loader.fetch_data(use_cache=True)
    target_ohlcv = loader.fetch_target_ohlcv(use_cache=True)

    # 3. Engineer features with configured rolling windows
    feat_cfg = config.get("features", {})
    engineer = FeatureEngineer(
        universe=universe,
        rolling_windows=feat_cfg.get("rolling_windows", [15, 30, 63, 189]),
        lead_lag_lags=feat_cfg.get("lead_lag_lags", [1, 2, 3]),
        cointegration_pairs=feat_cfg.get("cointegration_pairs", [["KO", "PEP"], ["KO", "XLP"], ["KO", "COKE"], ["KO", "TLT"]]),
        pca_components=feat_cfg.get("pca", {}).get("n_components", 3),
    )
    X = engineer.compute_features(prices, ohlcv=target_ohlcv)

    # 4. Filter strictly to the 1-year window
    year_mask = (X.index >= pd.to_datetime(start_date)) & (X.index <= pd.to_datetime(end_date))
    X_year = X[year_mask]
    n_bars = len(X_year)
    trading_days = n_bars // 3

    print(f"    Simulation Span: {X_year.index[0]} to {X_year.index[-1]}")
    print(f"    Observations: {n_bars} session bars (~{trading_days} trading days, 3 sessions/day)")

    # 5. Generate Predictions
    X_scaled = pd.DataFrame(
        trainer.scaler.transform(X_year[trainer.feature_names]),
        columns=trainer.feature_names,
        index=X_year.index,
    )
    probs = pd.Series(
        trainer.fitted_model.predict_proba(X_scaled)[:, 1],
        index=X_year.index,
    )

    # 6. Run Institutional Backtest Simulation
    bt_cfg = config.get("backtest", {})
    engine = BacktestEngine(
        initial_capital=initial_capital,
        transaction_cost_bps=bt_cfg.get("transaction_cost_bps", 5.0),
        position_sizing=bt_cfg.get("position_sizing", "confidence_weighted"),
        max_leverage=bt_cfg.get("max_leverage", 1.35),
        risk_free_rate=bt_cfg.get("risk_free_rate", 0.045),
        periods_per_year=bt_cfg.get("periods_per_year", 756),
    )
    results = engine.run(
        prices=prices,
        signals=probs,
        target_ticker="KO",
        benchmark_ticker="XLP",
    )

    # 7. Financial PnL Accounting
    end_equity = results["strategy_equity"].iloc[-1]
    net_pnl = end_equity - initial_capital
    net_return_pct = (end_equity / initial_capital - 1.0) * 100.0

    bh_end = results["target_bh_equity"].iloc[-1]
    bh_pnl = bh_end - initial_capital
    bh_return_pct = (bh_end / initial_capital - 1.0) * 100.0

    xlp_end = results["benchmark_bh_equity"].iloc[-1]
    xlp_pnl = xlp_end - initial_capital
    xlp_return_pct = (xlp_end / initial_capital - 1.0) * 100.0

    # Friction & Yield Breakdown
    total_turnover_dollars = (results["turnover"] * results["strategy_equity"]).sum()
    total_costs_dollars = (results["cost"] * results["strategy_equity"]).sum()
    cash_frac = np.maximum(0.0, 1.0 - results["position"])
    cash_yield_dollars = (cash_frac * (0.045 / 756.0) * results["strategy_equity"]).sum()
    gross_trading_pnl = net_pnl - cash_yield_dollars + total_costs_dollars

    # Performance Analytics
    m = PerformanceAnalytics.calculate_metrics(results["net_ret"], periods_per_year=756)
    m_bh = PerformanceAnalytics.calculate_metrics(results["target_ret"], periods_per_year=756)
    m_xlp = PerformanceAnalytics.calculate_metrics(results["benchmark_ret"], periods_per_year=756)

    # 8. Output Comprehensive PnL Tear Sheet
    print("\n" + "=" * 75)
    print(f"  1-YEAR ESTIMATED TOTAL PnL & FINANCIAL ACCOUNTING")
    print("=" * 75)
    print(f"  Starting Capital           : ${initial_capital:,.2f}")
    print(f"  Ending Strategy Equity     : ${end_equity:,.2f}")
    print(f"  -------------------------------------------------------------------------")
    print(f"  TOTAL NET STRATEGY PnL     : ${net_pnl:+12,.2f}  ({net_return_pct:+6.2f}%)")
    print(f"  Coca-Cola Buy & Hold PnL   : ${bh_pnl:+12,.2f}  ({bh_return_pct:+6.2f}%)")
    print(f"  XLP Sector ETF PnL         : ${xlp_pnl:+12,.2f}  ({xlp_return_pct:+6.2f}%)")
    print(f"  Net Alpha Over Buy & Hold  : ${net_pnl - bh_pnl:+12,.2f}  ({net_return_pct - bh_return_pct:+6.2f}%)")
    print(f"  Net Alpha Over Sector XLP  : ${net_pnl - xlp_pnl:+12,.2f}  ({net_return_pct - xlp_return_pct:+6.2f}%)")
    print(f"  -------------------------------------------------------------------------")
    print(f"  PnL COMPONENT BREAKDOWN:")
    print(f"    • Gross Alpha Trading PnL: ${gross_trading_pnl:+12,.2f}")
    print(f"    • Cash Interest Earned   : ${cash_yield_dollars:+12,.2f}  (4.5% annual risk-free sweep)")
    print(f"    • Transaction Fees Paid  : ${total_costs_dollars:12,.2f}  (5 bps per turn on ${total_turnover_dollars:,.0f} volume)")
    print(f"  -------------------------------------------------------------------------")
    print(f"  RISK & EXECUTION METRICS:")
    print(f"    • Sharpe Ratio           : {m['sharpe_ratio']:6.2f}   (KO: {m_bh['sharpe_ratio']:.2f} | XLP: {m_xlp['sharpe_ratio']:.2f})")
    print(f"    • Annualized Volatility  : {m['annualized_vol_pct']:6.2f}%  (KO: {m_bh['annualized_vol_pct']:.2f}% | XLP: {m_xlp['annualized_vol_pct']:.2f}%)")
    print(f"    • Maximum Drawdown       : {m['max_drawdown_pct']:6.2f}%  (KO: {m_bh['max_drawdown_pct']:.2f}% | XLP: {m_xlp['max_drawdown_pct']:.2f}%)")
    print(f"    • Win Rate               : {m['win_rate_pct']:6.2f}%  (Positive expectancy)")
    print(f"    • Profit Factor          : {m['profit_factor']:6.2f}   (Gross gains / Gross losses)")
    print("=" * 75)

    # 9. Monthly Breakdown
    results["month"] = results.index.to_period("M")
    monthly_ret = results.groupby("month")["net_ret"].apply(lambda r: (1 + r).prod() - 1)
    
    print("\nMONTHLY PnL BREAKDOWN:")
    print(f"  {'Month':<10s} | {'Net Return':<12s} | {'Dollar PnL':<14s} | Performance")
    print("  " + "-" * 60)
    for mo, r in monthly_ret.items():
        mo_pnl = initial_capital * r
        bar_len = min(25, int(abs(r) * 200))
        bar = "█" * bar_len
        sign = "+" if r >= 0 else "-"
        print(f"  {str(mo):<10s} | {r*100:+8.2f}%   | ${mo_pnl:+11,.2f}  | {sign} {bar}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Simulate 1-Year Quantitative Trading PnL.")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default="models/ko_model_up_to_2023.joblib", help="Path to model checkpoint")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date of 1-year window")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date of 1-year window")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial trading account capital")
    args = parser.parse_args()

    simulate_year(
        config_path=args.config,
        model_path=args.model,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
    )


if __name__ == "__main__":
    main()
