"""
Institutional Deep Neural Network & Ensemble Model Architectures for Quantitative Trading.
Implements Wide & Deep Tabular ResNet with over 1,400 neurons, BatchNorm, GELU, and Dropout.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset


class ResidualDenseBlock(nn.Module):
    """Residual dense block with batch normalization, GELU activation, and dropout."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.dropout1(self.act1(self.bn1(self.linear1(x))))
        out = self.dropout2(self.act2(self.bn2(self.linear2(out))))
        return out + res


class DeepQuantNet(nn.Module):
    """
    Multi-layer Wide & Deep Tabular Neural Network for financial market prediction.
    Features 1,472+ neurons across deep residual layers to capture complex non-linear patterns.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        # Default: 512 -> 512 -> 256 -> 128 -> 64 = 1,472 neurons
        if hidden_dims is None:
            hidden_dims = [512, 512, 256, 128, 64]

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(
                ResidualDenseBlock(
                    in_features=hidden_dims[i],
                    out_features=hidden_dims[i + 1],
                    dropout=max(0.15, dropout - i * 0.05),
                )
            )
        self.res_blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.res_blocks(h)
        return self.head(h)


class PyTorchNeuralNetClassifier:
    """
    Scikit-learn compatible classifier wrapper for DeepQuantNet.
    Features AdamW, Cosine Annealing, early stopping, and permutation importance.
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        epochs: int = 80,
        batch_size: int = 64,
        early_stopping_patience: int = 12,
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims or [512, 512, 256, 128, 64]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[DeepQuantNet] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.classes_ = np.array([0, 1])

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]] = None,
    ) -> "PyTorchNeuralNetClassifier":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)

        n_samples, n_features = X_arr.shape
        self.model = DeepQuantNet(
            input_dim=n_features,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            num_classes=2,
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )

        dataset = TensorDataset(
            torch.from_numpy(np.ascontiguousarray(X_arr.copy())),
            torch.from_numpy(np.ascontiguousarray(y_arr.copy())),
        )
        loader = TorchDataLoader(
            dataset, batch_size=min(self.batch_size, n_samples), shuffle=True
        )

        # Validation set for early stopping if provided
        X_val_t, y_val_t = None, None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_t = torch.from_numpy(np.asarray(X_val, dtype=np.float32)).to(self.device)
            y_val_t = torch.from_numpy(np.asarray(y_val, dtype=np.int64)).to(self.device)

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            # Early stopping check
            if X_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()
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

        # Compute permutation feature importance on training set
        self._compute_feature_importance(X_arr, y_arr)
        return self

    def _compute_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """Computes feature importance via permutation test on out-of-bag or in-sample data."""
        self.model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X).to(self.device)
            base_logits = self.model(x_t)
            base_probs = torch.softmax(base_logits, dim=-1)[:, 1].cpu().numpy()
            base_acc = np.mean((base_probs >= 0.5) == y)

        importances = []
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]

        # Sample subset for speed if n_samples > 300
        sub_idx = rng.choice(len(X), size=min(len(X), 300), replace=False)
        X_sub = X[sub_idx].copy()
        y_sub = y[sub_idx]

        with torch.no_grad():
            for f in range(n_features):
                saved_col = X_sub[:, f].copy()
                X_sub[:, f] = rng.permutation(saved_col)
                perm_t = torch.from_numpy(X_sub).to(self.device)
                perm_probs = torch.softmax(self.model(perm_t), dim=-1)[:, 1].cpu().numpy()
                perm_acc = np.mean((perm_probs >= 0.5) == y_sub)
                importances.append(max(0.0, base_acc - perm_acc + 0.01))
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
        X_arr = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            x_t = torch.from_numpy(X_arr).to(self.device)
            logits = self.model(x_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


class EnsembleQuantClassifier:
    """
    Ensemble blending DeepQuantNet (1,400+ neurons), LightGBM, and Ridge.
    Combines deep non-linear neural representations with gradient boosted decision boundaries.
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

        # Normalize weights
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

        # 1. Fit Deep Neural Network (1,400+ neurons)
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
        self.ridge_model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
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
