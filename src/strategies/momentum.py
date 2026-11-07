"""
Time-Series Momentum & Volatility-Targeted Trend Following Strategy for Coca-Cola ($KO).
Modeled after AQR / Man AHL CTA and systematic trend desks.
Implements multi-timeframe trend signals (Fast, Medium, Slow) scaled inversely by realized volatility.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class TimeSeriesMomentumStrategy:
    """
    Multi-Scale Time-Series Momentum (TSMOM) with Volatility Targeting.
    Filters trends across multiple horizons (5-day, 21-day, 63-day) and targets steady risk exposure.
    """

    def __init__(
        self,
        fast_window: int = 15,    # ~5 trading days
        med_window: int = 63,     # ~21 trading days
        slow_window: int = 189,   # ~63 trading days
        target_vol: float = 0.11, # 11% annualized target volatility
        max_leverage: float = 1.35,
        periods_per_year: int = 756,
    ):
        self.fast_w = fast_window
        self.med_w = med_window
        self.slow_w = slow_window
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.periods_per_year = periods_per_year

    def generate_signals(
        self,
        prices: pd.DataFrame,
        target_ticker: str = "KO",
    ) -> pd.DataFrame:
        """
        Generates continuous time-series momentum signals and volatility-scaled position weights.
        """
        target_p = prices[target_ticker]
        ret = target_p.pct_change().fillna(0.0)

        # 1. Multi-scale Exponential Moving Averages
        ema_fast = target_p.ewm(span=self.fast_w, adjust=False).mean()
        ema_med = target_p.ewm(span=self.med_w, adjust=False).mean()
        ema_slow = target_p.ewm(span=self.slow_w, adjust=False).mean()

        # Trend direction z-scores / distance from moving average
        trend_fast = (target_p / ema_fast - 1.0)
        trend_med = (target_p / ema_med - 1.0)
        trend_slow = (target_p / ema_slow - 1.0)

        # 2. Donchian Channel Breakouts (20-day high / low)
        donchian_high = target_p.rolling(self.med_w, min_periods=10).max()
        donchian_low = target_p.rolling(self.med_w, min_periods=10).min()
        channel_pos = (target_p - donchian_low) / (donchian_high - donchian_low + 1e-8)

        # 3. Normalized Combined Momentum Score in [-1.0, 1.0]
        # 40% Medium + 35% Fast + 25% Slow
        raw_mom_score = 0.35 * np.sign(trend_fast) + 0.40 * np.sign(trend_med) + 0.25 * np.sign(trend_slow)
        smooth_mom = (0.35 * (trend_fast / 0.02) + 0.40 * (trend_med / 0.04) + 0.25 * (trend_slow / 0.06)).clip(-1.0, 1.0)

        # 4. Volatility Scaling (Risk Parity / Vol Targeting)
        roll_vol = ret.rolling(self.med_w, min_periods=15).std() * np.sqrt(self.periods_per_year)
        roll_vol = roll_vol.replace(0, np.nan).bfill().clip(lower=0.04, upper=0.30)
        vol_scaler = (self.target_vol / roll_vol)

        # 5. Position Sizing
        # For long-only bias, shift baseline around 0.60
        base_tilt = (smooth_mom + 1.0) / 2.0  # Maps [-1, 1] -> [0, 1]
        mom_weights = (base_tilt * vol_scaler).clip(lower=0.0, upper=self.max_leverage)

        return pd.DataFrame(
            {
                "mom_score": smooth_mom,
                "raw_mom_direction": raw_mom_score,
                "channel_position": channel_pos,
                "realized_vol": roll_vol,
                "vol_scaler": vol_scaler,
                "tsmom_weight": mom_weights,
            },
            index=prices.index,
        )
