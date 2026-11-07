"""
Model training pipeline with strict out-of-sample generation and feature importance.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.cross_val import PurgedWalkForwardCV


class ModelTrainer:
    """Trains predictive models with walk-forward cross validation."""

    def __init__(self, config: dict):
        self.config = config
        self.model_type = config.get("model", {}).get("type", "lightgbm")
        self.cv_cfg = config.get("model", {}).get("cv", {})
        self.hyperparams = config.get("model", {}).get("hyperparameters", {})

    def _get_base_estimator(self):
        """Constructs base model based on configuration."""
        if self.model_type in ["neural_network", "deep_quant_net", "mlp"]:
            from src.models.neural_net import PyTorchNeuralNetClassifier
            return PyTorchNeuralNetClassifier(
                hidden_dims=self.hyperparams.get("hidden_dims", [512, 512, 256, 128, 64]),
                dropout=self.hyperparams.get("dropout", 0.30),
                learning_rate=self.hyperparams.get("learning_rate", 0.001),
                weight_decay=self.hyperparams.get("weight_decay", 0.0001),
                epochs=self.hyperparams.get("epochs", 80),
                batch_size=self.hyperparams.get("batch_size", 64),
                early_stopping_patience=self.hyperparams.get("early_stopping_patience", 12),
                random_state=self.hyperparams.get("random_state", 42),
            )
        elif self.model_type == "ensemble":
            from src.models.neural_net import EnsembleQuantClassifier
            ensemble_cfg = self.config.get("model", {}).get("ensemble", {})
            return EnsembleQuantClassifier(
                lgbm_weight=ensemble_cfg.get("lgbm_weight", 0.30),
                catboost_weight=ensemble_cfg.get("catboost_weight", 0.25),
                xgboost_weight=ensemble_cfg.get("xgboost_weight", 0.20),
                neural_weight=ensemble_cfg.get("neural_weight", 0.15),
                ridge_weight=ensemble_cfg.get("ridge_weight", 0.10),
                nn_params=self.config.get("model", {}).get("neural_network", {}),
                lgbm_params=self.config.get("model", {}).get("lightgbm", {}),
                catboost_params=self.config.get("model", {}).get("catboost", {}),
                xgboost_params=self.config.get("model", {}).get("xgboost", {}),
            )
        elif self.model_type == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=self.hyperparams.get("n_estimators", 150),
                learning_rate=self.hyperparams.get("learning_rate", 0.03),
                max_depth=self.hyperparams.get("max_depth", 4),
                num_leaves=self.hyperparams.get("num_leaves", 15),
                subsample=self.hyperparams.get("subsample", 0.8),
                colsample_bytree=self.hyperparams.get("colsample_bytree", 0.8),
                reg_alpha=self.hyperparams.get("reg_alpha", 0.1),
                reg_lambda=self.hyperparams.get("reg_lambda", 1.0),
                random_state=self.hyperparams.get("random_state", 42),
                verbose=-1,
            )
        elif self.model_type == "ridge":
            return LogisticRegression(
                C=0.01,
                max_iter=1000,
                random_state=42,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train_walk_forward(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Executes purged walk-forward cross-validation.
        
        Returns:
            oof_predictions (pd.Series): Out-of-fold predicted probabilities
            oof_targets (pd.Series): Ground truth target for test periods
            feature_importance (pd.DataFrame): Aggregated feature importance across folds
        """
        cv = PurgedWalkForwardCV(
            n_splits=self.cv_cfg.get("n_splits", 5),
            purge_window=self.cv_cfg.get("purge_window", 5),
            embargo_pct=self.cv_cfg.get("embargo_pct", 0.01),
        )

        oof_probs = pd.Series(index=X.index, dtype=float)
        fold_importances = []

        feature_names = list(X.columns)

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            # Fit scaler strictly on in-sample training fold
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

            model = self._get_base_estimator()
            model.fit(X_train_scaled, y_train)

            # Predict probabilities for class 1 (Outperform)
            probs = model.predict_proba(X_test_scaled)[:, 1]
            oof_probs.iloc[test_idx] = probs

            # Extract feature importance
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
            elif hasattr(model, "coef_"):
                imp = np.abs(model.coef_[0])
            else:
                imp = np.zeros(len(feature_names))

            fold_importances.append(imp)

        # Drop periods prior to first walk-forward test fold
        valid_mask = ~oof_probs.isna()
        clean_oof_probs = oof_probs[valid_mask]
        clean_oof_targets = y[valid_mask]

        # Aggregate feature importances
        avg_importance = np.mean(fold_importances, axis=0) if fold_importances else np.zeros(len(feature_names))
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": avg_importance
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return clean_oof_probs, clean_oof_targets, importance_df

    def fit_production(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fits final model on all provided training data (e.g. data <= 2023-12-31).
        Stores fitted model, scaler, and calculates final feature importances.
        """
        self.feature_names = list(X.columns)
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=self.feature_names, index=X.index)

        self.fitted_model = self._get_base_estimator()
        self.fitted_model.fit(X_scaled, y)

        if hasattr(self.fitted_model, "feature_importances_"):
            importances = self.fitted_model.feature_importances_
        elif hasattr(self.fitted_model, "coef_"):
            importances = np.abs(self.fitted_model.coef_[0])
        else:
            importances = np.zeros(len(self.feature_names))

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return importance_df

    def predict_signal(self, X_latest: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates prediction and sizing recommendation for the latest observation bar.
        """
        if not hasattr(self, "fitted_model") or self.fitted_model is None:
            raise RuntimeError("Model has not been fitted. Call fit_production or load_model first.")

        # Ensure correct column ordering
        X_sub = X_latest[self.feature_names]
        X_scaled = pd.DataFrame(self.scaler.transform(X_sub), columns=self.feature_names, index=X_sub.index)

        # Format timestamp and session
        ts = X_latest.index[-1]
        session_name = {
            11: "Morning Session (11:30 ET)",
            13: "Midday Session (13:30 ET)",
            16: "Afternoon / Close Session (16:00 ET)",
        }.get(ts.hour, "Daily Close" if ts.hour == 0 else f"Session ({ts.strftime('%H:%M')} ET)")
        date_str = str(ts.date()) if ts.hour == 0 else str(ts)

        # If model has dedicated quant multi-task decision engine, use it
        if hasattr(self.fitted_model, "predict_signal"):
            res = self.fitted_model.predict_signal(X_scaled)
            res["date"] = date_str
            res["session"] = session_name
            return res

        # Predict probability of class 1 (Outperform)
        prob = float(self.fitted_model.predict_proba(X_scaled)[-1, 1])

        # Sizing and action with realistic financial calibration
        confidence = float(np.clip((abs(prob - 0.5) / 0.12) * 0.50, 0.05, 0.65))

        # Continuous active tilt allocation around neutral 50% benchmark weight
        tilt = (prob - 0.50) / 0.08
        target_weight = float(np.clip(0.50 + tilt * 0.50, 0.0, 1.0))

        if prob >= 0.52:
            action = "BUY / LONG"
        elif prob <= 0.48:
            action = "NEUTRAL / CASH"
        else:
            action = "HOLD / NEUTRAL"

        return {
            "date": date_str,
            "session": session_name,
            "target": "Coca-Cola ($KO)",
            "probability_outperform": round(prob, 4),
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100.0, 1),
            "action": action,
            "recommended_weight": round(target_weight, 4),
            "recommended_weight_pct": round(target_weight * 100.0, 1),
        }

    def save_model(self, output_path: str = "models/production_model.joblib"):
        """Serializes the trained model bundle to disk."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": self.fitted_model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "config": self.config,
            "saved_at": pd.Timestamp.now().isoformat(),
        }
        joblib.dump(bundle, out)
        print(f"[*] Saved trained model checkpoint to {output_path}")

    def load_model(self, model_path: str = "models/production_model.joblib"):
        """Loads a pre-trained model bundle from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        bundle = joblib.load(path)
        self.fitted_model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_names = bundle["feature_names"]
        print(f"[*] Loaded trained model checkpoint from {model_path} (trained with {len(self.feature_names)} features)")

