"""
Unit tests for feature engineering and target labeling.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.universe import AssetUniverse
from src.features.engineer import FeatureEngineer
from src.features.labeling import TargetLabeler


@pytest.fixture
def sample_price_data():
    """Generates synthetic price data for testing."""
    dates = pd.date_range("2022-01-01", periods=150, freq="B")
    tickers = ["KO", "PEP", "KDP", "MNST", "CELH", "COKE", "KOF", "BALL", "CCK", "XLP", "SPY", "UUP"]
    rng = np.random.default_rng(123)
    
    data = {}
    for t in tickers:
        drift = 0.0004
        vol = 0.012
        ret = rng.normal(drift, vol, len(dates))
        data[t] = 50.0 * np.exp(np.cumsum(ret))
        
    return pd.DataFrame(data, index=dates)


def test_feature_engineering_no_nans_and_dimensions(sample_price_data):
    universe = AssetUniverse()
    fe = FeatureEngineer(universe=universe, rolling_windows=[5, 10, 21], lead_lag_lags=[1, 2])
    
    features = fe.compute_features(sample_price_data)
    
    assert not features.empty
    assert not features.isna().any().any(), "Features must not contain NaNs after warm-up drop"
    assert "rel_mom_PEP_w5" in features.columns
    assert "rel_mom_XLP_w10" in features.columns
    assert "lead_lag_PEP_lag1" in features.columns
    assert "spread_zscore_KO_PEP_w21" in features.columns


def test_target_labeling(sample_price_data):
    labeler = TargetLabeler(forward_horizon=5, neutral_threshold=0.001)
    y, fwd_ret = labeler.create_labels(sample_price_data, target_ticker="KO", benchmark_ticker="XLP")
    
    assert len(y) == len(sample_price_data)
    assert set(y.dropna().unique()).issubset({0, 1})
    assert len(fwd_ret.dropna()) == len(sample_price_data) - 5
