"""
Macroeconomic Regime & Factor Sensitivity Strategy for Coca-Cola ($KO).
Modeled after Bridgewater / Renaissance macro factor models.
Evaluates discount rate bond-proxy sensitivity (TLT), FX currency drag (UUP), and equity market risk (SPY/VIXY).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class MacroRegimeStrategy:
    """
    Classifies macroeconomic regimes and calculates factor multipliers for Coca-Cola ($KO).
    KO behaves as a classic high-dividend defensive bond proxy with high international FX exposure.
    """

    def __init__(
        self,
        lookback_window: int = 63,
        vix_spike_threshold: float = 1.5,
    ):
        self.lookback = lookback_window
        self.vix_thresh = vix_spike_threshold

    def generate_signals(
        self,
        prices: pd.DataFrame,
        target_ticker: str = "KO",
    ) -> pd.DataFrame:
        """
        Determines current macro regime and outputs continuous macro risk weight multiplier.
        """
        cols = prices.columns
        common_idx = prices.index

        # 1. Bond-Proxy Discount Rate Impact ($TLT)
        if "TLT" in cols:
            tlt_ret = prices["TLT"].pct_change(self.lookback)
            # Falling TLT (rising yields) hurts dividend bond-proxies -> Negative
            rate_z = (tlt_ret - tlt_ret.rolling(189, min_periods=30).mean()) / (
                tlt_ret.rolling(189, min_periods=30).std() + 1e-8
            )
        else:
            rate_z = pd.Series(0.0, index=common_idx)

        # 2. Foreign Exchange Headwind ($UUP)
        if "UUP" in cols:
            uup_ret = prices["UUP"].pct_change(self.lookback)
            # Surging US Dollar hurts Coca-Cola's translated 65%+ foreign earnings -> Negative
            fx_drag = -uup_ret
            fx_z = (fx_drag - fx_drag.rolling(189, min_periods=30).mean()) / (
                fx_drag.rolling(189, min_periods=30).std() + 1e-8
            )
        else:
            fx_z = pd.Series(0.0, index=common_idx)

        # 3. Market Risk Appetite / Flight to Safety ($SPY, $VIXY)
        if "SPY" in cols:
            spy_ret = prices["SPY"].pct_change(self.lookback)
            # In a broad market sell-off, capital rotates into defensive consumer staples (Flight to Quality)
            flight_to_safety = -spy_ret
            risk_z = (flight_to_safety - flight_to_safety.rolling(189, min_periods=30).mean()) / (
                flight_to_safety.rolling(189, min_periods=30).std() + 1e-8
            )
        else:
            risk_z = pd.Series(0.0, index=common_idx)

        # 4. Composite Macro Factor Score in [-1.0, 1.0]
        # 35% Rates (TLT) + 35% FX Dollar (UUP) + 30% Flight to Safety (SPY)
        composite_macro = 0.35 * rate_z + 0.35 * fx_z + 0.30 * risk_z
        macro_alpha = (composite_macro / 2.0).clip(lower=-1.0, upper=1.0)

        # Regime Classification
        regimes = pd.Series("NEUTRAL_EQUILIBRIUM", index=common_idx)
        regimes[risk_z > 1.0] = "DEFENSIVE_FLIGHT_TO_SAFETY"
        regimes[(rate_z < -1.0) & (fx_z < -1.0)] = "STAGFLATION_RATES_SHOCK"
        regimes[(rate_z > 1.0) & (risk_z < -1.0)] = "RISK_ON_GROWTH_EXPANSION"

        macro_weight = (0.60 + macro_alpha * 0.50).clip(lower=0.0, upper=1.35)

        return pd.DataFrame(
            {
                "rate_sensitivity_z": rate_z,
                "fx_drag_z": fx_z,
                "flight_to_safety_z": risk_z,
                "macro_alpha": macro_alpha,
                "macro_regime": regimes,
                "macro_weight": macro_weight,
            },
            index=common_idx,
        )
