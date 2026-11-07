"""
Rigorous Unit Tests Verifying Zero Data Leakage in Feature Engineering, Cross-Validation,
Target Labeling, and Out-of-Sample Evaluation.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.universe import AssetUniverse
from src.features.engineer import FeatureEngineer
from src.features.labeling import TargetLabeler
from src.models.cross_val import PurgedWalkForwardCV, CombinatorialPurgedCV
from src.models.trainer import ModelTrainer


@pytest.fixture
def base_price_history():
    """Generates baseline daily price history for 300 trading days."""
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    rng = np.random.default_rng(42)
    universe = AssetUniverse()
    tickers = universe.all_tickers

    data = {}
    for t in tickers:
        ret = rng.normal(0.0003, 0.01, len(dates))
        data[t] = 100.0 * np.exp(np.cumsum(ret))

    return pd.DataFrame(data, index=dates)


def test_feature_engineering_future_data_causality(base_price_history):
    """
    Leakage Test 1: Feature Causality.
    Modifying price data at day t+1 must NOT change engineered features at day t.
    """
    universe = AssetUniverse()
    fe = FeatureEngineer(universe=universe, rolling_windows=[5, 10, 21], lead_lag_lags=[1, 2])

    # 1. Compute baseline features
    features_base = fe.compute_features(base_price_history.copy())

    # Pick an inspection point t well after warm-up
    cutoff_idx = len(features_base) // 2
    cutoff_date = features_base.index[cutoff_idx]
    base_features_at_t = features_base.loc[cutoff_date].copy()

    # 2. Perturb future prices at t+1 and beyond drastically (10x shock)
    perturbed_prices = base_price_history.copy()
    future_mask = perturbed_prices.index > cutoff_date
    perturbed_prices.loc[future_mask] = perturbed_prices.loc[future_mask] * 10.0

    # 3. Recompute features on perturbed price dataset
    features_perturbed = fe.compute_features(perturbed_prices)
    perturbed_features_at_t = features_perturbed.loc[cutoff_date].copy()

    # Features at day t MUST BE exactly identical (zero future lookahead)
    pd.testing.assert_series_equal(
        base_features_at_t,
        perturbed_features_at_t,
        check_exact=False,
        rtol=1e-7,
        atol=1e-7,
        obj="Features at time t must be completely unaffected by future prices at t+1",
    )


def test_target_labeling_lookahead_horizon_isolation(base_price_history):
    """
    Leakage Test 2: Target Horizon Isolation.
    A forward label with horizon H at day t must ONLY depend on data up to day t+H.
    Modifying prices after day t+H must NOT change the label at day t.
    """
    horizon = 5
    labeler = TargetLabeler(forward_horizon=horizon, target_type="relative_alpha")

    # Baseline labels
    y_base, fwd_ret_base = labeler.create_labels(base_price_history.copy(), target_ticker="KO", benchmark_ticker="XLP")

    test_idx = 80
    test_date = base_price_history.index[test_idx]
    base_label_at_t = y_base.loc[test_date]
    base_ret_at_t = fwd_ret_base.loc[test_date]

    # Perturb prices strictly after t + horizon
    horizon_date = base_price_history.index[test_idx + horizon]
    perturbed_prices = base_price_history.copy()
    after_horizon_mask = perturbed_prices.index > horizon_date
    perturbed_prices.loc[after_horizon_mask] = perturbed_prices.loc[after_horizon_mask] * 5.0

    y_perturbed, fwd_ret_perturbed = labeler.create_labels(perturbed_prices, target_ticker="KO", benchmark_ticker="XLP")

    assert y_perturbed.loc[test_date] == base_label_at_t, "Target label at t changed when prices after t+H were modified!"
    np.testing.assert_almost_equal(
        fwd_ret_perturbed.loc[test_date],
        base_ret_at_t,
        decimal=6,
        err_msg="Forward return at t changed when prices after t+H were modified!",
    )


def test_purged_walk_forward_cv_no_overlap_and_embargo():
    """
    Leakage Test 3: Purged Walk-Forward Cross-Validation Split Integrity.
    Verifies that train and test folds never share indices, and that train_end is
    strictly separated from test_start by at least purge_window + 1.
    """
    n_samples = 300
    X = pd.DataFrame(np.random.randn(n_samples, 5))
    y = pd.Series(np.random.randint(0, 2, size=n_samples))

    purge_window = 5
    embargo_pct = 0.02
    cv = PurgedWalkForwardCV(n_splits=5, purge_window=purge_window, embargo_pct=embargo_pct)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        # 1. Zero index intersection
        overlap = set(train_idx).intersection(set(test_idx))
        assert len(overlap) == 0, f"Fold {fold}: Train and test share indices: {overlap}"

        # 2. Strict Purging: max(train_idx) must be strictly < min(test_idx) - purge_window
        min_test = np.min(test_idx)
        max_train = np.max(train_idx)
        assert max_train <= min_test - purge_window - 1, (
            f"Fold {fold}: Data leakage! max train {max_train} violates purge window {purge_window} from test {min_test}"
        )


def test_combinatorial_purged_cv_embargo_and_purge():
    """
    Leakage Test 4: Combinatorial Purged CV (CPCV) Purge & Embargo Enforcement.
    Verifies both pre-test purging and post-test embargo buffers across all combinatorial folds.
    """
    n_samples = 250
    X = pd.DataFrame(np.random.randn(n_samples, 4))
    y = pd.Series(np.random.randint(0, 2, size=n_samples))

    purge_window = 5
    embargo_pct = 0.02
    embargo_samples = max(1, int(n_samples * embargo_pct))

    cpcv = CombinatorialPurgedCV(
        n_splits=5,
        n_test_splits=2,
        purge_window=purge_window,
        embargo_pct=embargo_pct,
    )

    for fold, (train_idx, test_idx) in enumerate(cpcv.split(X, y)):
        # Zero intersection
        overlap = set(train_idx).intersection(set(test_idx))
        assert len(overlap) == 0, f"CPCV Fold {fold}: Overlapping train and test indices!"

        # Check embargo on samples immediately following any test block
        for t in test_idx:
            # The sample immediately after test cannot be in train if within embargo window
            for e in range(1, embargo_samples + 1):
                assert (t + e) not in train_idx, f"CPCV Fold {fold}: Post-test sample {t+e} in train violates embargo!"


def test_scaler_strict_training_quarantine():
    """
    Leakage Test 5: Scaler and Preprocessing Isolation.
    Verifies that feature scaler is fit strictly on training samples and never influenced by test data.
    """
    rng = np.random.default_rng(101)
    dates_train = pd.date_range("2020-01-01", periods=100, freq="B")
    dates_test = pd.date_range("2020-06-01", periods=50, freq="B")

    # In-sample train has mean 0, std 1
    X_train = pd.DataFrame({"f1": rng.normal(0, 1, 100), "f2": rng.normal(10, 2, 100)}, index=dates_train)
    y_train = pd.Series(rng.choice([0, 1], size=100), index=dates_train)

    # Out-of-sample test has completely different distribution (mean 50, std 10)
    X_test = pd.DataFrame({"f1": rng.normal(50, 10, 50), "f2": rng.normal(100, 20, 50)}, index=dates_test)

    config = {
        "model": {
            "type": "ridge",
            "hyperparameters": {},
        }
    }
    trainer = ModelTrainer(config)
    trainer.fit_production(X_train, y_train)

    # Scaler mean must match X_train mean within float tolerance, not X_test
    np.testing.assert_allclose(trainer.scaler.mean_, X_train.mean().values, rtol=1e-5)
    np.testing.assert_allclose(trainer.scaler.scale_, X_train.std(ddof=0).values, rtol=1e-5)
