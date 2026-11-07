"""
Institutional Multi-Task Quant Neural Network Machine for Trading Coca-Cola ($KO).
Features Gated Feature Attention, 3,520 Neurons across Deep Tabular ResNet layers, LayerNorm, Swish,
and Multi-Task Heads for Directional Probability, Expected Alpha Return, and Predictive Uncertainty Sizing.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset


class GatedFeatureAttention(nn.Module):
    """
    Feature Tokenizer and Regime-Aware Gating Mechanism.
    Dynamically attends to relevant factors (e.g. commodities vs rates vs momentum) based on market state.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate(x)
        return x * g


class ResidualDenseBlockLN(nn.Module):
    """Residual dense block with LayerNorm, Swish (SiLU) activation, and dropout."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.ln1 = nn.LayerNorm(out_features)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(out_features, out_features)
        self.ln2 = nn.LayerNorm(out_features)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout)

        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.dropout1(self.act1(self.ln1(self.linear1(x))))
        out = self.dropout2(self.act2(self.ln2(self.linear2(out))))
        return out + res


class CocaColaQuantNeuralMachine(nn.Module):
    """
    Dedicated Multi-Task Quantitative Neural Machine for Coca-Cola ($KO).
    Contains 3,520 neurons across deep tabular residual blocks.
    
    Predicts:
    1. Directional Alpha Probability P(KO Outperforms XLP)
    2. Continuous Expected Alpha Return
    3. Predictive Uncertainty / Volatility for dynamic Kelly position sizing
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.30,
    ):
        super().__init__()
        # Default architecture: 1024 -> 1024 -> 512 -> 512 -> 256 -> 128 -> 64 = 3,520 neurons
        if hidden_dims is None:
            hidden_dims = [1024, 1024, 512, 512, 256, 128, 64]

        self.feature_gating = GatedFeatureAttention(input_dim)

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        blocks = []
        for i in range(len(hidden_dims) - 1):
            stage_dropout = max(0.15, dropout - i * 0.03)
            blocks.append(
                ResidualDenseBlockLN(
                    in_features=hidden_dims[i],
                    out_features=hidden_dims[i + 1],
                    dropout=stage_dropout,
                )
            )
        self.res_backbone = nn.Sequential(*blocks)

        # Multi-Task Predictive Heads
        bottleneck_dim = hidden_dims[-1]

        # Head 1: Directional classification (2 logits: underperform vs outperform)
        self.head_direction = nn.Linear(bottleneck_dim, 2)

        # Head 2: Continuous expected alpha return
        self.head_expected_return = nn.Linear(bottleneck_dim, 1)

        # Head 3: Predictive uncertainty / return variance (Softplus ensures positive variance)
        self.head_variance = nn.Sequential(
            nn.Linear(bottleneck_dim, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (batch_size, 2) directional classification logits
            expected_ret: (batch_size, 1) continuous predicted relative return
            variance: (batch_size, 1) predictive return variance for Kelly sizing
        """
        x_gated = self.feature_gating(x)
        h = self.input_layer(x_gated)
        h = self.res_backbone(h)

        logits = self.head_direction(h)
        expected_ret = self.head_expected_return(h)
        variance = self.head_variance(h) + 1e-4

        return logits, expected_ret, variance


# Alias DeepQuantNet for backwards compatibility
DeepQuantNet = CocaColaQuantNeuralMachine


class PyTorchNeuralNetClassifier:
    """
    Production Scikit-learn Compatible Interface for the Quant Neural Machine.
    Executes multi-task training with AdamW, Cosine Annealing, early stopping,
    and calculates risk-adjusted Kelly position sizing.
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.30,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        epochs: int = 80,
        batch_size: int = 64,
        early_stopping_patience: int = 12,
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims or [1024, 1024, 512, 512, 256, 128, 64]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[CocaColaQuantNeuralMachine] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.classes_ = np.array([0, 1])

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        fwd_returns: Optional[Union[pd.Series, np.ndarray]] = None,
        eval_set: Optional[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]] = None,
    ) -> "PyTorchNeuralNetClassifier":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float32).copy())
        y_arr = np.ascontiguousarray(np.asarray(y, dtype=np.int64).copy())

        if fwd_returns is not None:
            r_arr = np.ascontiguousarray(np.asarray(fwd_returns, dtype=np.float32).copy())
        else:
            # Construct synthetic proxy from directional labels if continuous returns not passed
            r_arr = np.where(y_arr == 1, 0.015, -0.015).astype(np.float32)

        n_samples, n_features = X_arr.shape
        self.model = CocaColaQuantNeuralMachine(
            input_dim=n_features,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        criterion_ce = nn.CrossEntropyLoss()
        criterion_huber = nn.HuberLoss(delta=0.01)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )

        dataset = TensorDataset(
            torch.from_numpy(X_arr),
            torch.from_numpy(y_arr),
            torch.from_numpy(r_arr),
        )
        loader = TorchDataLoader(
            dataset, batch_size=min(self.batch_size, n_samples), shuffle=True
        )

        # Validation set for early stopping if provided
        X_val_t, y_val_t = None, None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_t = torch.from_numpy(np.ascontiguousarray(np.asarray(X_val, dtype=np.float32).copy())).to(self.device)
            y_val_t = torch.from_numpy(np.ascontiguousarray(np.asarray(y_val, dtype=np.int64).copy())).to(self.device)

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y, batch_r in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_r = batch_r.to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                logits, pred_ret, pred_var = self.model(batch_x)

                loss_ce = criterion_ce(logits, batch_y)
                loss_huber = criterion_huber(pred_ret, batch_r)
                # Gaussian negative log likelihood for predictive uncertainty
                loss_nll = 0.5 * (torch.log(pred_var) + ((batch_r - pred_ret) ** 2) / pred_var).mean()

                loss = loss_ce + 0.3 * loss_huber + 0.1 * loss_nll
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            # Early stopping check
            if X_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits, _, _ = self.model(X_val_t)
                    val_loss = criterion_ce(val_logits, y_val_t).item()
                self.model.train()

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        self.model.eval()
        self._compute_feature_importance(X_arr, y_arr)
        return self

    def _compute_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """Computes permutation feature importance across the factor library."""
        self.model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(np.ascontiguousarray(X.copy())).to(self.device)
            base_logits, _, _ = self.model(x_t)
            base_probs = torch.softmax(base_logits, dim=-1)[:, 1].cpu().numpy()
            base_acc = np.mean((base_probs >= 0.5) == y)

        importances = []
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]

        sub_idx = rng.choice(len(X), size=min(len(X), 250), replace=False)
        X_sub = X[sub_idx].copy()
        y_sub = y[sub_idx]

        with torch.no_grad():
            for f in range(n_features):
                saved_col = X_sub[:, f].copy()
                X_sub[:, f] = rng.permutation(saved_col)
                perm_t = torch.from_numpy(np.ascontiguousarray(X_sub.copy())).to(self.device)
                perm_logits, _, _ = self.model(perm_t)
                perm_probs = torch.softmax(perm_logits, dim=-1)[:, 1].cpu().numpy()
                perm_acc = np.mean((perm_probs >= 0.5) == y_sub)
                importances.append(max(0.0, base_acc - perm_acc + 0.005))
                X_sub[:, f] = saved_col

        imp_arr = np.array(importances, dtype=float)
        total = imp_arr.sum()
        if total > 0:
            imp_arr = imp_arr / total
        self.feature_importances_ = imp_arr

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")

        self.model.eval()
        X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float32).copy())
        with torch.no_grad():
            x_t = torch.from_numpy(X_arr).to(self.device)
            logits, _, _ = self.model(x_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def predict_signal(self, X_latest: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes multi-task quantitative prediction for Coca-Cola ($KO).
        Computes directional probability, continuous expected return, predictive uncertainty,
        and optimal risk-adjusted Kelly sizing.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted.")

        self.model.eval()
        X_arr = np.ascontiguousarray(np.asarray(X_latest, dtype=np.float32).copy())
        with torch.no_grad():
            x_t = torch.from_numpy(X_arr).to(self.device)
            logits, expected_ret_t, var_t = self.model(x_t)
            prob_outperform = float(torch.softmax(logits, dim=-1)[-1, 1].item())
            expected_alpha = float(expected_ret_t[-1, 0].item())
            pred_vol = float(torch.sqrt(var_t[-1, 0]).item())

        confidence = abs(prob_outperform - 0.5) * 2.0

        # Mean-Variance / Kelly Position Sizing:
        # Scale continuous expected alpha by uncertainty and directional confidence
        risk_aversion = 2.0
        raw_kelly = (expected_alpha / (risk_aversion * (pred_vol ** 2) + 1e-6)) * confidence
        target_weight = float(np.clip(raw_kelly, -1.0, 1.0))

        if prob_outperform >= 0.55 and expected_alpha > 0:
            action = "BUY / LONG"
            target_weight = max(0.15, min(1.0, abs(target_weight)))
        elif prob_outperform <= 0.45 or expected_alpha < -0.005:
            action = "NEUTRAL / CASH"
            target_weight = 0.0
        else:
            action = "HOLD / NEUTRAL"
            target_weight = 0.0

        return {
            "date": str(X_latest.index[-1].date()),
            "target": "Coca-Cola ($KO)",
            "probability_outperform": round(prob_outperform, 4),
            "expected_alpha_5d_pct": round(expected_alpha * 100.0, 3),
            "predictive_volatility_pct": round(pred_vol * 100.0, 3),
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100.0, 1),
            "recommended_weight": round(target_weight, 4),
            "recommended_weight_pct": round(target_weight * 100.0, 1),
            "action": action,
        }


class EnsembleQuantClassifier:
    """
    Institutional Ensemble Machine for Coca-Cola Trading.
    Blends the Deep Multi-Task Quant Neural Machine (3,520 neurons), LightGBM, and Ridge.
    """

    def __init__(
        self,
        neural_weight: float = 0.50,
        lgbm_weight: float = 0.35,
        ridge_weight: float = 0.15,
        nn_params: Optional[dict] = None,
        lgbm_params: Optional[dict] = None,
    ):
        self.neural_weight = neural_weight
        self.lgbm_weight = lgbm_weight
        self.ridge_weight = ridge_weight

        tot = self.neural_weight + self.lgbm_weight + self.ridge_weight
        self.neural_weight /= tot
        self.lgbm_weight /= tot
        self.ridge_weight /= tot

        self.nn_params = nn_params or {}
        self.lgbm_params = lgbm_params or {}

        self.nn_model: Optional[PyTorchNeuralNetClassifier] = None
        self.lgbm_model = None
        self.ridge_model = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.classes_ = np.array([0, 1])

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> "EnsembleQuantClassifier":
        import lightgbm as lgb
        from sklearn.linear_model import LogisticRegression

        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_ser = pd.Series(y) if not isinstance(y, pd.Series) else y

        # 1. Fit Multi-Task Quant Neural Machine (3,520 neurons)
        self.nn_model = PyTorchNeuralNetClassifier(**self.nn_params)
        self.nn_model.fit(X_df, y_ser)

        # 2. Fit LightGBM
        default_lgb = {
            "n_estimators": 150,
            "learning_rate": 0.03,
            "max_depth": 4,
            "num_leaves": 15,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
        }
        default_lgb.update(self.lgbm_params)
        self.lgbm_model = lgb.LGBMClassifier(**default_lgb)
        self.lgbm_model.fit(X_df, y_ser)

        # 3. Fit Ridge Logistic Regression
        self.ridge_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.ridge_model.fit(X_df, y_ser)

        # Blend feature importances
        nn_imp = self.nn_model.feature_importances_
        lgb_imp = self.lgbm_model.feature_importances_.astype(float)
        if lgb_imp.sum() > 0:
            lgb_imp /= lgb_imp.sum()
        ridge_imp = np.abs(self.ridge_model.coef_[0])
        if ridge_imp.sum() > 0:
            ridge_imp /= ridge_imp.sum()

        self.feature_importances_ = (
            self.neural_weight * nn_imp
            + self.lgbm_weight * lgb_imp
            + self.ridge_weight * ridge_imp
        )
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        p_nn = self.nn_model.predict_proba(X)
        p_lgb = self.lgbm_model.predict_proba(X)
        p_ridge = self.ridge_model.predict_proba(X)

        blended = (
            self.neural_weight * p_nn
            + self.lgbm_weight * p_lgb
            + self.ridge_weight * p_ridge
        )
        return blended

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def predict_signal(self, X_latest: pd.DataFrame) -> Dict[str, Any]:
        """Delegates to the Quant Neural Machine's multi-task prediction engine."""
        return self.nn_model.predict_signal(X_latest)
