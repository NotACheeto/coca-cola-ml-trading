"""
Marcos López de Prado's Triple Barrier Method & Secondary Meta-Labeling Engine.
Separates trade direction (Primary Model) from trade quality and sizing (Secondary Meta-Model).
Reference: 'Advances in Financial Machine Learning' (Marcos López de Prado, 2018).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class TripleBarrierMetaLabeler:
    """
    Implements dynamic volatility-adjusted Triple Barrier labeling and secondary meta-model sizing.
    Barriers:
      - Upper Barrier: +pt_multiplier * Daily_Vol (Profit Taking)
      - Lower Barrier: -sl_multiplier * Daily_Vol (Stop Loss)
      - Vertical Barrier: max_holding_bars (Time expiration)
    """

    def __init__(
        self,
        pt_multiplier: float = 2.0,
        sl_multiplier: float = 1.5,
        max_holding_bars: int = 15,  # ~5 trading days
        vol_lookback: int = 30,
    ):
        self.pt_mult = pt_multiplier
        self.sl_mult = sl_multiplier
        self.max_holding = max_holding_bars
        self.vol_lookback = vol_lookback

    def generate_triple_barriers(
        self,
        prices: pd.Series,
    ) -> pd.DataFrame:
        """
        Calculates the first barrier touched for every observation bar.
        Returns:
            label: 1 if upper barrier (profit) hit first, 0 if lower barrier (loss) hit first or time exit negative
            ret: Realized trade return over the barrier window
            bars_held: Number of bars until barrier touch
            touch_type: 'PROFIT_TAKE', 'STOP_LOSS', or 'TIME_EXIT'
        """
        daily_ret = prices.pct_change().fillna(0.0)
        daily_vol = daily_ret.rolling(self.vol_lookback, min_periods=10).std().bfill()

        labels = pd.Series(index=prices.index, dtype=float)
        realized_rets = pd.Series(index=prices.index, dtype=float)
        touch_types = pd.Series(index=prices.index, dtype=object)

        n = len(prices)
        p_arr = prices.values
        vol_arr = daily_vol.values

        for i in range(n - self.max_holding):
            p0 = p_arr[i]
            vol0 = vol_arr[i]
            upper = p0 * (1.0 + self.pt_mult * vol0)
            lower = p0 * (1.0 - self.sl_mult * vol0)

            touched = False
            for j in range(1, self.max_holding + 1):
                pj = p_arr[i + j]
                if pj >= upper:
                    labels.iloc[i] = 1.0
                    realized_rets.iloc[i] = pj / p0 - 1.0
                    touch_types.iloc[i] = "PROFIT_TAKE"
                    touched = True
                    break
                elif pj <= lower:
                    labels.iloc[i] = 0.0
                    realized_rets.iloc[i] = pj / p0 - 1.0
                    touch_types.iloc[i] = "STOP_LOSS"
                    touched = True
                    break

            if not touched:
                # Vertical time barrier expired
                p_end = p_arr[i + self.max_holding]
                final_ret = p_end / p0 - 1.0
                labels.iloc[i] = 1.0 if final_ret > 0 else 0.0
                realized_rets.iloc[i] = final_ret
                touch_types.iloc[i] = "TIME_EXIT"

        return pd.DataFrame(
            {
                "meta_label": labels,
                "realized_return": realized_rets,
                "touch_type": touch_types,
                "volatility_barrier": daily_vol,
            },
            index=prices.index,
        )

    def filter_and_size_trades(
        self,
        primary_signals: pd.Series,
        meta_probabilities: pd.Series,
        threshold: float = 0.50,
        max_leverage: float = 1.35,
    ) -> pd.Series:
        """
        Applies meta-model filtering:
        If meta_probability < threshold, cancel trade (set size to 0.0).
        If meta_probability >= threshold, scale position by confidence 2 * (p_meta - 0.50).
        """
        clean_idx = primary_signals.index.intersection(meta_probabilities.index)
        p_prim = primary_signals.loc[clean_idx]
        p_meta = meta_probabilities.loc[clean_idx]

        # Active sizing formula
        confidence = (p_meta - 0.50).clip(lower=0.0) * 2.0
        active_weights = np.where(
            (p_prim >= 0.52) & (p_meta >= threshold),
            (0.60 + confidence * 0.75).clip(upper=max_leverage),
            0.0,
        )

        return pd.Series(active_weights, index=clean_idx)
