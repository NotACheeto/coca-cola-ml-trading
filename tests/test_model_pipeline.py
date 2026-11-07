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
