"""
Institutional Multi-Strategy Alpha Combiner & Risk Parity Allocator for Coca-Cola ($KO).
Integrates all 5 quantitative alpha pillars:
  1. Statistical Arbitrage (Kalman Filter Cointegration Pairs)
  2. Time-Series Momentum (TSMOM Trend Following)
  3. Cross-Asset Supply Chain & Commodity Lead-Lag
  4. Macroeconomic Regime & Factor Allocation
  5. 5-Model Multi-Paradigm Machine Learning Ensemble
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.strategies.stat_arb import KalmanStatArbStrategy
from src.strategies.momentum import TimeSeriesMomentumStrategy
from src.strategies.cross_asset_lead_lag import CrossAssetLeadLagStrategy
from src.strategies.macro_regime import MacroRegimeStrategy


class MultiStrategyAlphaCombiner:
    """
    Fuses multiple non-correlated quantitative alpha signals into an optimal composite trade order.
    Uses inverse-volatility weighting, macro-regime conditioning, and dynamic cash management.
    """

    def __init__(
        self,
        strategy_weights: Optional[Dict[str, float]] = None,
        max_leverage: float = 1.35,
        target_volatility: float = 0.11,
    ):
        # Default institutional risk allocation:
        # 35% ML Multi-Paradigm Ensemble + 25% Stat-Arb + 20% TSMOM + 10% Cross-Asset + 10% Macro
        self.weights = strategy_weights or {
            "ml_ensemble": 0.35,
            "stat_arb": 0.25,
            "tsmom": 0.20,
            "cross_asset": 0.10,
            "macro_regime": 0.10,
        }
        tot = sum(self.weights.values())
        self.weights = {k: v / tot for k, v in self.weights.items()}
        self.max_leverage = max_leverage
        self.target_vol = target_volatility

        self.stat_arb = KalmanStatArbStrategy()
        self.tsmom = TimeSeriesMomentumStrategy(target_vol=self.target_vol, max_leverage=self.max_leverage)
        self.cross_asset = CrossAssetLeadLagStrategy()
        self.macro = MacroRegimeStrategy()

    def generate_composite_signals(
        self,
        prices: pd.DataFrame,
        ml_probabilities: pd.Series,
        target_ticker: str = "KO",
    ) -> pd.DataFrame:
        """
        Calculates all sub-strategy signals and synthesizes them into an institutional composite trade allocation.
        """
        # 1. Compute sub-strategy signals on full price history to support rolling warm-up
        sa_df = self.stat_arb.generate_signals(prices, target_ticker=target_ticker)
        mom_df = self.tsmom.generate_signals(prices, target_ticker=target_ticker)
        ca_df = self.cross_asset.generate_signals(prices, target_ticker=target_ticker)
        mac_df = self.macro.generate_signals(prices, target_ticker=target_ticker)

        # Align to evaluation period where ML predictions exist
        common_idx = prices.index.intersection(ml_probabilities.index).sort_values()
        sub_ml_probs = ml_probabilities.loc[common_idx]

        # 2. Extract normalized alpha scores in [-1.0, 1.0]
        # ML ensemble alpha centered at 0.50
        ml_alpha = ((sub_ml_probs - 0.50) / 0.08).clip(-1.0, 1.0)
        sa_alpha = sa_df.loc[common_idx, "stat_arb_alpha"]
        mom_alpha = mom_df.loc[common_idx, "mom_score"]
        ca_alpha = ca_df.loc[common_idx, "lead_lag_alpha"]
        mac_alpha = mac_df.loc[common_idx, "macro_alpha"]

        # 3. Macro Regime Conditioning
        # If in DEFENSIVE_FLIGHT_TO_SAFETY -> boost StatArb and ML Longs
        regimes = mac_df.loc[common_idx, "macro_regime"]
        boost_mask = regimes == "DEFENSIVE_FLIGHT_TO_SAFETY"
        trim_mask = regimes == "STAGFLATION_RATES_SHOCK"

        # 4. Composite Alpha Blending
        composite_alpha = (
            self.weights["ml_ensemble"] * ml_alpha
            + self.weights["stat_arb"] * sa_alpha
            + self.weights["tsmom"] * mom_alpha
            + self.weights["cross_asset"] * ca_alpha
            + self.weights["macro_regime"] * mac_alpha
        )

        # Apply regime multiplier
        regime_multiplier = pd.Series(1.0, index=common_idx)
        regime_multiplier[boost_mask] = 1.15
        regime_multiplier[trim_mask] = 0.85
        final_alpha = (composite_alpha * regime_multiplier).fillna(0.0).clip(-1.0, 1.0)

        # 5. Risk-Parity Active Conviction Sizing
        # 0.60 neutral base scaled by multi-strategy composite alpha
        composite_weights = (0.60 + final_alpha * 0.70).fillna(0.60).clip(lower=0.0, upper=self.max_leverage)

        # Action Classification
        actions = pd.Series("HOLD / NEUTRAL", index=common_idx)
        actions[composite_weights >= 0.85] = "OVERWEIGHT / HIGH CONVICTION LONG"
        actions[(composite_weights > 0.0) & (composite_weights < 0.40)] = "UNDERWEIGHT / DEFENSIVE CASH"
        actions[composite_weights == 0.0] = "ZERO EXPOSURE / 100% CASH HEDGE"

        return pd.DataFrame(
            {
                "composite_alpha": final_alpha,
                "composite_weight": composite_weights,
                "action": actions,
                "macro_regime": regimes,
                "ml_alpha": ml_alpha,
                "stat_arb_alpha": sa_alpha,
                "tsmom_alpha": mom_alpha,
                "cross_asset_alpha": ca_alpha,
                "macro_alpha": mac_alpha,
                "tsmom_realized_vol": mom_df["realized_vol"],
                "stat_arb_zscore": sa_df["composite_zscore"],
            },
            index=common_idx,
        )
