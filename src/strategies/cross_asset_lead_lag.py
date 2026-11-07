"""
Cross-Asset Supply Chain & Lead-Lag Spillover Strategy for Coca-Cola ($KO).
Modeled after Two Sigma / Point72 cross-asset equity quantitative desks.
Ingests commodity cost pressures (Sugar, Aluminum, Oil), downstream bottlers, and competitor earnings leads.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class CrossAssetLeadLagStrategy:
    """
    Evaluates upstream supply chain cost shocks, downstream bottling margin signals,
    and competitor momentum divergence to forecast Coca-Cola ($KO) outperformance.
    """

    def __init__(
        self,
        lookback_window: int = 63,  # ~21 trading days (1 month)
        fast_window: int = 15,      # ~5 trading days (1 week)
    ):
        self.lookback = lookback_window
        self.fast_w = fast_window

    def generate_signals(
        self,
        prices: pd.DataFrame,
        target_ticker: str = "KO",
    ) -> pd.DataFrame:
        """
        Calculates cross-asset lead-lag alpha score in [-1.0, 1.0].
        """
        cols = prices.columns
        target_ret = prices[target_ticker].pct_change(self.lookback)

        # 1. Commodity Input Cost Pressure (Sugar CANE, Aluminum DBB, Crude Oil USO)
        comm_signals = []
        if "CANE" in cols:
            sugar_mom = prices["CANE"].pct_change(self.lookback)
            # Surging sugar prices compress margins -> Negative for KO
            comm_signals.append(-sugar_mom)
        if "DBB" in cols:
            base_metal_mom = prices["DBB"].pct_change(self.lookback)
            # Surging aluminum packaging costs compress margins -> Negative for KO
            comm_signals.append(-base_metal_mom)
        if "USO" in cols:
            oil_mom = prices["USO"].pct_change(self.lookback)
            # Surging oil increases PET plastic bottle and transportation costs -> Negative for KO
            comm_signals.append(-oil_mom)

        if comm_signals:
            comm_pressure = pd.concat(comm_signals, axis=1).mean(axis=1)
            # Standardize rolling z-score of commodity shock
            comm_z = (comm_pressure - comm_pressure.rolling(189, min_periods=30).mean()) / (
                comm_pressure.rolling(189, min_periods=30).std() + 1e-8
            )
        else:
            comm_z = pd.Series(0.0, index=prices.index)

        # 2. Downstream Bottler Volume Signal ($COKE, $KOF)
        bottler_cols = [c for c in ["COKE", "KOF"] if c in cols]
        if bottler_cols:
            bottler_basket = prices[bottler_cols].pct_change(self.fast_w).mean(axis=1)
            ko_fast = prices[target_ticker].pct_change(self.fast_w)
            # When bottlers lead KO momentum, positive operational spillover
            bottler_spillover = (bottler_basket - ko_fast)
            bottler_z = (bottler_spillover - bottler_spillover.rolling(63, min_periods=15).mean()) / (
                bottler_spillover.rolling(63, min_periods=15).std() + 1e-8
            )
        else:
            bottler_z = pd.Series(0.0, index=prices.index)

        # 3. Direct Competitor Divergence ($PEP, $KDP, $MNST)
        peer_cols = [c for c in ["PEP", "KDP", "MNST"] if c in cols]
        if peer_cols:
            peer_basket = prices[peer_cols].pct_change(self.lookback).mean(axis=1)
            # Competitor spillover: if peers broke out and KO lagged without news, KO catches up
            peer_divergence = (peer_basket - target_ret)
            peer_z = (peer_divergence - peer_divergence.rolling(63, min_periods=15).mean()) / (
                peer_divergence.rolling(63, min_periods=15).std() + 1e-8
            )
        else:
            peer_z = pd.Series(0.0, index=prices.index)

        # 4. Composite Cross-Asset Alpha Score in [-1.0, 1.0]
        # 40% Commodity Cost Pressure + 35% Bottler Lead + 25% Competitor Divergence
        composite_lead_lag = 0.40 * comm_z + 0.35 * bottler_z + 0.25 * peer_z
        clamped_alpha = (composite_lead_lag / 2.0).clip(lower=-1.0, upper=1.0)

        # Active Weight
        target_weight = (0.60 + clamped_alpha * 0.50).clip(lower=0.0, upper=1.35)

        return pd.DataFrame(
            {
                "commodity_cost_z": comm_z,
                "bottler_lead_z": bottler_z,
                "peer_divergence_z": peer_z,
                "lead_lag_alpha": clamped_alpha,
                "lead_lag_weight": target_weight,
            },
            index=prices.index,
        )
