"""
Institutional Quantitative Strategies for Coca-Cola ($KO).
Implements canonical hedge fund strategy families:
  1. Statistical Arbitrage (Kalman Filter Cointegration Pairs)
  2. Time-Series Momentum (TSMOM & Volatility-Targeted Trend)
  3. Cross-Asset Supply Chain & Commodity Lead-Lag
  4. Macroeconomic Regime & Factor Sensitivity
  5. Marcos López de Prado Triple Barrier & Meta-Labeling
  6. Multi-Strategy Alpha Combiner & Risk Parity Allocator
"""

from src.strategies.stat_arb import KalmanStatArbStrategy
from src.strategies.momentum import TimeSeriesMomentumStrategy
from src.strategies.cross_asset_lead_lag import CrossAssetLeadLagStrategy
from src.strategies.macro_regime import MacroRegimeStrategy
from src.strategies.meta_labeling import TripleBarrierMetaLabeler
from src.strategies.combiner import MultiStrategyAlphaCombiner

__all__ = [
    "KalmanStatArbStrategy",
    "TimeSeriesMomentumStrategy",
    "CrossAssetLeadLagStrategy",
    "MacroRegimeStrategy",
    "TripleBarrierMetaLabeler",
    "MultiStrategyAlphaCombiner",
]
