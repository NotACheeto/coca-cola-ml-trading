"""
Unit tests for BacktestEngine and PerformanceAnalytics.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.analytics import PerformanceAnalytics
from src.backtest.engine import BacktestEngine


def test_backtest_engine_execution_lag_and_costs():
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    prices = pd.DataFrame(
        {
            "KO": 60.0 * np.exp(np.cumsum(np.repeat(0.001, 100))),
            "XLP": 70.0 * np.exp(np.cumsum(np.repeat(0.0005, 100))),
        },
        index=dates,
    )
    # Binary signal alternating every 10 days
    signals = pd.Series([1.0 if (i // 10) % 2 == 0 else 0.0 for i in range(100)], index=dates)

    engine = BacktestEngine(
        initial_capital=100000.0,
        transaction_cost_bps=10.0,
        position_sizing="binary",
    )

    results = engine.run(prices, signals, target_ticker="KO", benchmark_ticker="XLP")

    # Position at t=0 must be 0 due to execution shift(1)
    assert results["position"].iloc[0] == 0.0
    # Position at t=1 should reflect signal at t=0
    assert results["position"].iloc[1] == signals.iloc[0]
    # Costs should be non-negative
    assert (results["cost"] >= 0).all()
    # Net equity should be computed
    assert results["strategy_equity"].iloc[-1] > 0


def test_performance_analytics_metrics():
    # 252 days of 10 bps daily return with small vol
    rng = np.random.default_rng(42)
    daily_returns = pd.Series(rng.normal(0.0008, 0.008, 252))
    metrics = PerformanceAnalytics.calculate_metrics(daily_returns, risk_free_rate=0.02)

    assert "sharpe_ratio" in metrics
    assert "cagr_pct" in metrics
    assert "max_drawdown_pct" in metrics
    assert metrics["max_drawdown_pct"] >= 0.0
    assert metrics["annualized_vol_pct"] > 0.0
