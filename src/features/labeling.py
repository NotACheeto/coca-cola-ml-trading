"""
Labeling methods for financial ML targets (Relative Alpha & Triple Barrier).
"""

from typing import Tuple
import numpy as np
import pandas as pd


class TargetLabeler:
    """Generates stationary, forward-looking supervisory signals for model training."""

    def __init__(
        self,
        forward_horizon: int = 5,
        neutral_threshold: float = 0.0015,
        target_type: str = "relative_alpha",
    ):
        self.forward_horizon = forward_horizon
        self.neutral_threshold = neutral_threshold
        self.target_type = target_type

    def create_labels(
        self, prices: pd.DataFrame, target_ticker: str = "KO", benchmark_ticker: str = "XLP"
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Creates target supervisory signal and forward continuous return.
        
        Returns:
            y (pd.Series): Binary or multi-class target (+1 for outperform/long, -1 or 0 for underperform/short)
            forward_ret (pd.Series): Actual forward continuous return of target asset
        """
        # Forward return of target
        target_fwd_ret = prices[target_ticker].pct_change(self.forward_horizon).shift(-self.forward_horizon)

        if benchmark_ticker in prices.columns and self.target_type == "relative_alpha":
            # Forward return of sector benchmark
            bm_fwd_ret = prices[benchmark_ticker].pct_change(self.forward_horizon).shift(-self.forward_horizon)
            alpha = target_fwd_ret - bm_fwd_ret
        else:
            alpha = target_fwd_ret

        # Binary classification target: 1 = Long (positive relative alpha), 0 = Short / Neutral
        # If neutral_threshold > 0, we can also generate a strong signal indicator
        binary_label = (alpha > 0).astype(int)

        # Retain raw forward return for backtesting evaluation
        return binary_label, target_fwd_ret

    def compute_triple_barrier_labels(
        self,
        prices: pd.Series,
        upper_barrier_std: float = 2.0,
        lower_barrier_std: float = 2.0,
        vol_window: int = 20,
    ) -> pd.Series:
        """
        Implements Marcos López de Prado's Triple Barrier Method:
        - Upper profit-taking barrier
        - Lower stop-loss barrier
        - Vertical time limit barrier (forward_horizon)
        """
        daily_ret = prices.pct_change()
        roll_vol = daily_ret.rolling(vol_window).std()

        labels = pd.Series(index=prices.index, dtype=float)

        for i in range(len(prices) - self.forward_horizon):
            current_p = prices.iloc[i]
            vol = roll_vol.iloc[i]
            if np.isnan(vol) or vol == 0:
                continue

            upper_p = current_p * (1.0 + upper_barrier_std * vol * np.sqrt(self.forward_horizon))
            lower_p = current_p * (1.0 - lower_barrier_std * vol * np.sqrt(self.forward_horizon))

            path = prices.iloc[i + 1 : i + self.forward_horizon + 1]

            # Check barrier breaches
            hit_upper = (path >= upper_p).any()
            hit_lower = (path <= lower_p).any()

            if hit_upper and not hit_lower:
                labels.iloc[i] = 1.0  # Profit take hit first
            elif hit_lower and not hit_upper:
                labels.iloc[i] = -1.0  # Stop loss hit first
            elif hit_upper and hit_lower:
                # Find which occurred first
                idx_upper = (path >= upper_p).idxmax()
                idx_lower = (path <= lower_p).idxmax()
                labels.iloc[i] = 1.0 if idx_upper < idx_lower else -1.0
            else:
                # Vertical barrier: sign of return at horizon
                ret = (path.iloc[-1] - current_p) / current_p
                labels.iloc[i] = 1.0 if ret > 0 else -1.0

        return labels
