"""
Unit tests for institutional quantitative strategies:
  1. KalmanStatArbStrategy (Kalman Filter Cointegration Pairs)
  2. TimeSeriesMomentumStrategy (Multi-Scale Trend Following)
  3. CrossAssetLeadLagStrategy (Commodity Input Cost & Bottler Lead)
  4. MacroRegimeStrategy (Rates, FX, and Flight to Safety)
  5. TripleBarrierMetaLabeler (López de Prado Triple Barrier)
  6. MultiStrategyAlphaCombiner (Composite Fusion)
"""

import numpy as np
import pandas as pd
import pytest

from src.strategies.stat_arb import KalmanStatArbStrategy
from src.strategies.momentum import TimeSeriesMomentumStrategy
from src.strategies.cross_asset_lead_lag import CrossAssetLeadLagStrategy
from src.strategies.macro_regime import MacroRegimeStrategy
from src.strategies.meta_labeling import TripleBarrierMetaLabeler
from src.strategies.combiner import MultiStrategyAlphaCombiner


@pytest.fixture
def sample_market_data():
    """Generates synthetic multi-asset price history over 200 bars."""
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    rng = np.random.default_rng(42)

    # Cointegrated random walk for KO and PEP
    drift = 0.0005
    ko_shocks = rng.normal(drift, 0.008, 200)
    pep_shocks = ko_shocks * 0.85 + rng.normal(0, 0.004, 200)

    p_ko = 60.0 * np.exp(np.cumsum(ko_shocks))
    p_pep = 170.0 * np.exp(np.cumsum(pep_shocks))
    p_xlp = 75.0 * np.exp(np.cumsum(ko_shocks * 0.7 + rng.normal(0, 0.003, 200)))
    p_coke = 650.0 * np.exp(np.cumsum(ko_shocks * 1.1 + rng.normal(0, 0.005, 200)))
    p_cane = 12.0 * np.exp(np.cumsum(rng.normal(0, 0.015, 200)))
    p_dbb = 20.0 * np.exp(np.cumsum(rng.normal(0, 0.012, 200)))
    p_uso = 70.0 * np.exp(np.cumsum(rng.normal(0, 0.018, 200)))
    p_tlt = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.008, 200)))
    p_uup = 28.0 * np.exp(np.cumsum(rng.normal(0, 0.003, 200)))
    p_spy = 450.0 * np.exp(np.cumsum(rng.normal(0.0008, 0.01, 200)))

    df = pd.DataFrame(
        {
            "KO": p_ko,
            "PEP": p_pep,
            "XLP": p_xlp,
            "COKE": p_coke,
            "CANE": p_cane,
            "DBB": p_dbb,
            "USO": p_uso,
            "TLT": p_tlt,
            "UUP": p_uup,
            "SPY": p_spy,
        },
        index=dates,
    )
    return df


def test_kalman_stat_arb_strategy(sample_market_data):
    strat = KalmanStatArbStrategy()
    signals = strat.generate_signals(sample_market_data, target_ticker="KO")

    assert not signals.empty
    assert "composite_zscore" in signals.columns
    assert "stat_arb_alpha" in signals.columns
    assert "stat_arb_position" in signals.columns
    # Check bounds
    assert (signals["stat_arb_alpha"].dropna() >= -1.0).all()
    assert (signals["stat_arb_alpha"].dropna() <= 1.0).all()


def test_time_series_momentum_strategy(sample_market_data):
    strat = TimeSeriesMomentumStrategy()
    signals = strat.generate_signals(sample_market_data, target_ticker="KO")

    assert not signals.empty
    assert "mom_score" in signals.columns
    assert "vol_scaler" in signals.columns
    assert "tsmom_weight" in signals.columns
    assert (signals["tsmom_weight"].dropna() >= 0.0).all()
    assert (signals["tsmom_weight"].dropna() <= 1.35).all()


def test_cross_asset_lead_lag_strategy(sample_market_data):
    strat = CrossAssetLeadLagStrategy(lookback_window=30, fast_window=10)
    signals = strat.generate_signals(sample_market_data, target_ticker="KO")

    assert not signals.empty
    assert "lead_lag_alpha" in signals.columns
    assert "lead_lag_weight" in signals.columns
    assert (signals["lead_lag_alpha"].dropna() >= -1.0).all()
    assert (signals["lead_lag_alpha"].dropna() <= 1.0).all()


def test_macro_regime_strategy(sample_market_data):
    strat = MacroRegimeStrategy(lookback_window=30)
    signals = strat.generate_signals(sample_market_data, target_ticker="KO")

    assert not signals.empty
    assert "macro_alpha" in signals.columns
    assert "macro_regime" in signals.columns
    assert set(signals["macro_regime"].unique()).issubset(
        {"NEUTRAL_EQUILIBRIUM", "DEFENSIVE_FLIGHT_TO_SAFETY", "STAGFLATION_RATES_SHOCK", "RISK_ON_GROWTH_EXPANSION"}
    )


def test_triple_barrier_meta_labeler(sample_market_data):
    labeler = TripleBarrierMetaLabeler(max_holding_bars=10)
    barriers = labeler.generate_triple_barriers(sample_market_data["KO"])

    assert not barriers.empty
    assert "meta_label" in barriers.columns
    assert "touch_type" in barriers.columns
    valid = barriers["meta_label"].dropna()
    assert set(valid.unique()).issubset({0.0, 1.0})


def test_multi_strategy_alpha_combiner(sample_market_data):
    combiner = MultiStrategyAlphaCombiner()
    # Mock ML probabilities
    ml_probs = pd.Series(0.53, index=sample_market_data.index)

    composite = combiner.generate_composite_signals(sample_market_data, ml_probs, target_ticker="KO")

    assert not composite.empty
    assert "composite_alpha" in composite.columns
    assert "composite_weight" in composite.columns
    assert "action" in composite.columns
    assert (composite["composite_weight"] >= 0.0).all()
    assert (composite["composite_weight"] <= 1.35).all()
