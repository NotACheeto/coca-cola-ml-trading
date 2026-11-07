"""
Model evaluation metrics tailored for statistical finance and alpha modeling.
"""

from typing import Dict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score


class ModelEvaluator:
    """Evaluates ML model performance on financial time series."""

    @staticmethod
    def evaluate(
        y_true: pd.Series,
        y_prob: pd.Series,
        forward_returns: pd.Series = None
    ) -> Dict[str, float]:
        """
        Computes financial ML metrics:
        - ROC AUC
        - Directional Accuracy
        - Brier Score (probability calibration error)
        - Information Coefficient (Spearman Rank Correlation)
        """
        # Ensure alignment
        common_idx = y_true.dropna().index.intersection(y_prob.dropna().index)
        y_t = y_true.loc[common_idx]
        y_p = y_prob.loc[common_idx]

        auc = roc_auc_score(y_t, y_p) if len(np.unique(y_t)) > 1 else 0.5
        acc = accuracy_score(y_t, (y_p >= 0.5).astype(int))
        brier = brier_score_loss(y_t, y_p)

        metrics = {
            "roc_auc": float(auc),
            "directional_accuracy": float(acc),
            "brier_score": float(brier),
            "sample_count": int(len(common_idx)),
        }

        # Calculate Information Coefficient (IC) if forward returns provided
        if forward_returns is not None:
            fwd_aligned = forward_returns.loc[common_idx].dropna()
            valid_sub_idx = fwd_aligned.index.intersection(y_p.index)
            if len(valid_sub_idx) > 10:
                ic, p_val = spearmanr(y_p.loc[valid_sub_idx], fwd_aligned.loc[valid_sub_idx])
                metrics["information_coefficient"] = float(ic)
                metrics["ic_p_value"] = float(p_val)

        return metrics
