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


@pytest.fixture
def sample_ohlcv_data(sample_price_data):
    """Generates synthetic OHLCV data for target asset."""
    dates = sample_price_data.index
    c = sample_price_data["KO"]
    rng = np.random.default_rng(42)
    o = c * (1.0 + rng.normal(0, 0.002, len(dates)))
    h = np.maximum(o, c) * (1.0 + np.abs(rng.normal(0, 0.004, len(dates))))
    l = np.minimum(o, c) * (1.0 - np.abs(rng.normal(0, 0.004, len(dates))))
    v = rng.lognormal(mean=16.7, sigma=0.4, size=len(dates))
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=dates)


def test_feature_engineering_no_nans_and_dimensions(sample_price_data, sample_ohlcv_data):
    universe = AssetUniverse()
    fe = FeatureEngineer(universe=universe, rolling_windows=[5, 10, 21], lead_lag_lags=[1, 2])
    
    features = fe.compute_features(sample_price_data, ohlcv=sample_ohlcv_data)
    
    assert not features.empty
    assert not features.isna().any().any(), "Features must not contain NaNs after warm-up drop"
    assert "rel_mom_PEP_w5" in features.columns
    assert "vol_garman_klass_w5" in features.columns
    assert "rsi_KO_w14" in features.columns
    assert "macd_line" in features.columns
    assert "frac_diff_KO_d40" in features.columns
    assert len(features.columns) >= 70


def test_target_labeling(sample_price_data):
    labeler = TargetLabeler(forward_horizon=5, neutral_threshold=0.001)
    y, fwd_ret = labeler.create_labels(sample_price_data, target_ticker="KO", benchmark_ticker="XLP")
    
    assert len(y) == len(sample_price_data)
    assert set(y.dropna().unique()).issubset({0, 1})
    assert len(fwd_ret.dropna()) == len(sample_price_data) - 5


def test_intraday_3sessions_features(sample_price_data, sample_ohlcv_data):
    """Verifies that 3-session intraday index correctly generates session regime indicators."""
    universe = AssetUniverse()
    fe = FeatureEngineer(universe=universe, rolling_windows=[5, 10, 21], lead_lag_lags=[1, 2])

    # Convert sample dates to 3 intraday session timestamps
    session_ts = []
    for d in sample_price_data.index:
        d_str = d.strftime("%Y-%m-%d")
        session_ts.extend([
            pd.Timestamp(f"{d_str} 11:30:00"),
            pd.Timestamp(f"{d_str} 13:30:00"),
            pd.Timestamp(f"{d_str} 16:00:00"),
        ])
    
    tiled_prices = pd.DataFrame(
        np.repeat(sample_price_data.values, 3, axis=0),
        index=pd.DatetimeIndex(session_ts),
        columns=sample_price_data.columns,
    )
    tiled_ohlcv = pd.DataFrame(
        np.repeat(sample_ohlcv_data.values, 3, axis=0),
        index=pd.DatetimeIndex(session_ts),
        columns=sample_ohlcv_data.columns,
    )

    features = fe.compute_features(tiled_prices, ohlcv=tiled_ohlcv)

    assert not features.empty
    assert "session_id" in features.columns
    assert "session_sin" in features.columns
    assert "session_cos" in features.columns
    assert "is_morning_session" in features.columns
    assert "is_midday_session" in features.columns
    assert "is_close_session" in features.columns
    assert "session_return" in features.columns
    assert "session_volume_ratio" in features.columns
    assert set(features["session_id"].unique()).issubset({0, 1, 2})

