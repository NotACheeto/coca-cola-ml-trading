"""
Unit test for model fitting, serialization, and out-of-sample prediction.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.trainer import ModelTrainer


def test_model_production_fit_save_and_predict(tmp_path):
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    rng = np.random.default_rng(42)

    # Synthetic features
    X = pd.DataFrame(
        {
            "feat_1": rng.normal(0, 1, 120),
            "feat_2": rng.normal(0, 1, 120),
            "feat_3": rng.normal(0, 1, 120),
        },
        index=dates,
    )
    y = pd.Series(rng.choice([0, 1], size=120), index=dates)

    config = {
        "model": {
            "type": "lightgbm",
            "hyperparameters": {
                "n_estimators": 10,
                "learning_rate": 0.05,
                "random_state": 42,
            },
        }
    }

    trainer = ModelTrainer(config)
    imp_df = trainer.fit_production(X, y)

    assert not imp_df.empty
    assert len(imp_df) == 3

    # Test serialization
    model_file = tmp_path / "test_model.joblib"
    trainer.save_model(str(model_file))
    assert model_file.exists()

    # Test loading and predicting on a new unseen bar
    new_trainer = ModelTrainer(config)
    new_trainer.load_model(str(model_file))

    latest_bar = X.iloc[[-1]]
    decision = new_trainer.predict_signal(latest_bar)

    assert "probability_outperform" in decision
    assert 0.0 <= decision["probability_outperform"] <= 1.0
    assert decision["action"] in ["BUY / LONG", "NEUTRAL / CASH", "HOLD / NEUTRAL"]


def test_neural_network_fit_save_and_predict(tmp_path):
    """Verifies that the DeepQuantNet neural network (1400+ neurons) fits, serializes, and predicts."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        {
            "feat_1": rng.normal(0, 1, 100),
            "feat_2": rng.normal(0, 1, 100),
            "feat_3": rng.normal(0, 1, 100),
            "feat_4": rng.normal(0, 1, 100),
        },
        index=dates,
    )
    y = pd.Series(rng.choice([0, 1], size=100), index=dates)

    config = {
        "model": {
            "type": "neural_network",
            "hyperparameters": {
                "hidden_dims": [128, 64, 32],
                "dropout": 0.2,
                "epochs": 10,
                "batch_size": 32,
                "random_state": 42,
            },
        }
    }

    trainer = ModelTrainer(config)
    imp_df = trainer.fit_production(X, y)

    assert not imp_df.empty
    assert len(imp_df) == 4

    # Save and reload
    model_file = tmp_path / "test_nn_model.joblib"
    trainer.save_model(str(model_file))
    assert model_file.exists()

    new_trainer = ModelTrainer(config)
    new_trainer.load_model(str(model_file))
    decision = new_trainer.predict_signal(X.iloc[[-1]])

    assert "probability_outperform" in decision
    assert 0.0 <= decision["probability_outperform"] <= 1.0


def test_ensemble_model_fit_save_and_predict(tmp_path):
    """Verifies that the Ensemble model (Deep NN + LightGBM + Ridge) fits, serializes, and predicts."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(99)

    X = pd.DataFrame(
        {
            "f1": rng.normal(0, 1, 100),
            "f2": rng.normal(0, 1, 100),
            "f3": rng.normal(0, 1, 100),
        },
        index=dates,
    )
    y = pd.Series(rng.choice([0, 1], size=100), index=dates)

    config = {
        "model": {
            "type": "ensemble",
            "ensemble": {
                "lgbm_weight": 0.30,
                "catboost_weight": 0.25,
                "xgboost_weight": 0.20,
                "neural_weight": 0.15,
                "ridge_weight": 0.10,
            },
            "neural_network": {
                "hidden_dims": [64, 32],
                "epochs": 5,
                "batch_size": 32,
                "random_state": 42,
            },
            "lightgbm": {
                "n_estimators": 10,
                "random_state": 42,
            },
            "catboost": {
                "iterations": 10,
                "random_seed": 42,
                "verbose": 0,
            },
            "xgboost": {
                "n_estimators": 10,
                "random_state": 42,
                "eval_metric": "logloss",
            },
        }
    }

    trainer = ModelTrainer(config)
    imp_df = trainer.fit_production(X, y)

    assert not imp_df.empty
    assert len(imp_df) == 3

    model_file = tmp_path / "test_ensemble_model.joblib"
    trainer.save_model(str(model_file))
    assert model_file.exists()

    new_trainer = ModelTrainer(config)
    new_trainer.load_model(str(model_file))
    decision = new_trainer.predict_signal(X.iloc[[-1]])

    assert "probability_outperform" in decision
    assert 0.0 <= decision["probability_outperform"] <= 1.0
