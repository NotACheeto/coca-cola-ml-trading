"""
Model training pipeline with strict out-of-sample generation and feature importance.
"""

from typing import Any, Dict, List, Optional, Tuple
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
        if self.model_type == "lightgbm":
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
                penalty="l2",
                C=1.0,
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
