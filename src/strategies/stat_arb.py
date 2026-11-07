"""
Statistical Arbitrage & Dynamic Cointegration Pairs Strategy for Coca-Cola ($KO).
Modeled after Citadel / Millennium equity stat-arb desks.
Uses recursive Kalman Filtering to track dynamic hedge ratios and mean-reverting residuals
against economic peer PepsiCo ($PEP$), Consumer Staples ($XLP$), and Bottler ($COKE$).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DynamicKalmanFilter:
    """
    Online recursive Kalman Filter estimating time-varying hedge ratio and spread intercept.
    State vector: theta_t = [alpha_t, beta_t]^T
    Observation: y_t = alpha_t + beta_t * x_t + epsilon_t
    """

    def __init__(self, delta: float = 1e-4, measurement_noise: float = 1e-3):
        self.delta = delta
        self.measurement_noise = measurement_noise
        self.theta = np.zeros(2)  # [alpha, beta]
        self.P = np.eye(2) * 1.0  # Initial state covariance
        self.R = measurement_noise

    def update(self, y: float, x: float) -> Tuple[float, float, float]:
        """
        Processes single sequential observation (zero lookahead).
        Returns:
            beta: Updated dynamic cointegrating hedge ratio
            residual: Innovation spread error (e_t)
            z_score: Standardized residual z-score (e_t / sqrt(Q_t))
        """
        # 1. State transition: P_{t|t-1} = P_{t-1} + W
        # Random walk model on parameters
        W = self.delta / (1.0 - self.delta) * np.eye(2)
        P_prior = self.P + W

        # 2. Measurement prediction: H = [1, x]
        H = np.array([1.0, x])
        y_pred = np.dot(H, self.theta)

        # 3. Innovation error & variance
        e = y - y_pred
        Q = np.dot(H, np.dot(P_prior, H)) + self.R

        # 4. Kalman Gain
        K = np.dot(P_prior, H) / Q

        # 5. State & Covariance Update
        self.theta = self.theta + K * e
        self.P = P_prior - np.outer(K, np.dot(H, P_prior))

        z_score = e / np.sqrt(max(Q, 1e-8))
        return float(self.theta[1]), float(e), float(z_score)


class KalmanStatArbStrategy:
    """
    Multi-Pair Statistical Arbitrage Strategy for Coca-Cola ($KO).
    Generates mean-reverting signals across a basket of cointegrated anchors.
    """

    def __init__(
        self,
        entry_zscore: float = 1.75,
        exit_zscore: float = 0.50,
        pairs: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.entry_z = entry_zscore
        self.exit_z = exit_zscore
        self.pairs = pairs or ["PEP", "XLP", "COKE"]
        self.weights = weights or {"PEP": 0.50, "XLP": 0.35, "COKE": 0.15}

    def generate_signals(
        self,
        prices: pd.DataFrame,
        target_ticker: str = "KO",
    ) -> pd.DataFrame:
        """
        Iterates chronologically through price history, updating Kalman filters online
        and outputting continuous StatArb signals and hedge ratios.
        """
        common_idx = prices.index
        log_p = np.log(prices)

        z_scores = {p: pd.Series(index=common_idx, dtype=float) for p in self.pairs if p in log_p.columns}
        hedge_ratios = {p: pd.Series(index=common_idx, dtype=float) for p in self.pairs if p in log_p.columns}

        filters = {p: DynamicKalmanFilter(delta=1e-4, measurement_noise=1e-3) for p in z_scores}

        for t in common_idx:
            y_val = log_p.loc[t, target_ticker]
            for p, kf in filters.items():
                x_val = log_p.loc[t, p]
                beta, _, z = kf.update(y_val, x_val)
                hedge_ratios[p].loc[t] = beta
                z_scores[p].loc[t] = z

        z_df = pd.DataFrame(z_scores)

        # Composite Z-score weighted across pairs
        avail_pairs = [p for p in self.pairs if p in z_df.columns]
        w_sum = sum(self.weights.get(p, 1.0) for p in avail_pairs)
        norm_weights = {p: self.weights.get(p, 1.0) / w_sum for p in avail_pairs}

        composite_z = sum(z_df[p] * norm_weights[p] for p in avail_pairs)

        # Continuous mean-reversion alpha signal in [-1.0, 1.0]
        # When composite_z is deeply negative -> KO is underpriced relative to peers -> Strong Long
        # When composite_z is deeply positive -> KO is overpriced relative to peers -> Defensive / Short
        raw_alpha = -composite_z / 2.0
        stat_arb_signal = raw_alpha.clip(lower=-1.0, upper=1.0)

        # Discrete trade action flags
        state = 0.0
        positions = pd.Series(index=common_idx, dtype=float)

        for i, z in enumerate(composite_z):
            if z <= -self.entry_z:
                state = 1.0  # Long KO (underpriced)
            elif z >= self.entry_z:
                state = -0.5  # Defensive / Underweight KO (overpriced)
            elif abs(z) <= self.exit_z:
                state = 0.0  # Equilibrium reached
            positions.iloc[i] = state

        res = pd.DataFrame(
            {
                "composite_zscore": composite_z,
                "stat_arb_alpha": stat_arb_signal,
                "stat_arb_position": positions,
            },
            index=common_idx,
        )

        for p in avail_pairs:
            res[f"zscore_{p}"] = z_df[p]
            res[f"hedge_ratio_{p}"] = hedge_ratios[p]

        return res
