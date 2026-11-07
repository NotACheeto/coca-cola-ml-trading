"""
Feature engineering for cross-asset, peer-relative, and cointegration signals.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.data.universe import AssetUniverse


class FeatureEngineer:
    """Computes cross-asset features, relative momentum, spreads, and lead-lag indicators."""

    def __init__(
        self,
        universe: AssetUniverse,
        rolling_windows: Optional[List[int]] = None,
        lead_lag_lags: Optional[List[int]] = None,
        cointegration_pairs: Optional[List[Tuple[str, str]]] = None,
        pca_components: int = 3,
    ):
        self.universe = universe
        self.rolling_windows = rolling_windows or [5, 10, 21, 63]
        self.lead_lag_lags = lead_lag_lags or [1, 2, 3]
        self.cointegration_pairs = cointegration_pairs or [("KO", "PEP"), ("KO", "XLP")]
        self.pca_components = pca_components

    def compute_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw price data into an enriched predictive feature matrix.
        Strictly respects temporal order (no forward-looking leakage).
        """
        target = self.universe.target
        features = pd.DataFrame(index=prices.index)

        # 1. Base Log Returns across all universe assets
        log_prices = np.log(prices)
        daily_returns = prices.pct_change()
        daily_log_returns = log_prices.diff()

        # 2. Competitor & Sector Relative Momentum
        for w in self.rolling_windows:
            target_ret = prices[target].pct_change(w)
            
            # Primary competitor spread (KO vs PEP)
            if "PEP" in prices.columns:
                pep_ret = prices["PEP"].pct_change(w)
                features[f"rel_mom_PEP_w{w}"] = target_ret - pep_ret

            # Sector benchmark spread (KO vs XLP)
            if "XLP" in prices.columns:
                xlp_ret = prices["XLP"].pct_change(w)
                features[f"rel_mom_XLP_w{w}"] = target_ret - xlp_ret

            # Peer group average momentum
            peer_cols = [c for c in self.universe.direct_competitors if c in prices.columns]
            if peer_cols:
                peer_mean_ret = prices[peer_cols].pct_change(w).mean(axis=1)
                features[f"rel_mom_peer_mean_w{w}"] = target_ret - peer_mean_ret

            # Supplier momentum (Ball Corp / Crown Holdings) - Input cost indicator
            supplier_cols = [c for c in self.universe.suppliers if c in prices.columns]
            if supplier_cols:
                supplier_mean_ret = prices[supplier_cols].pct_change(w).mean(axis=1)
                features[f"rel_mom_suppliers_w{w}"] = target_ret - supplier_mean_ret

            # Bottler momentum (COKE / KOF) - Downstream distribution indicator
            bottler_cols = [c for c in self.universe.bottlers if c in prices.columns]
            if bottler_cols:
                bottler_mean_ret = prices[bottler_cols].pct_change(w).mean(axis=1)
                features[f"rel_mom_bottlers_w{w}"] = target_ret - bottler_mean_ret

        # 3. Cointegration Spreads & Rolling Z-Scores (Statistical Arbitrage)
        for pair in self.cointegration_pairs:
            s1, s2 = pair[0], pair[1]
            if s1 in log_prices.columns and s2 in log_prices.columns:
                for w in [21, 63]:
                    # Rolling regression beta without lookahead
                    roll_cov = daily_log_returns[s1].rolling(w).cov(daily_log_returns[s2])
                    roll_var = daily_log_returns[s2].rolling(w).var()
                    hedge_ratio = (roll_cov / (roll_var + 1e-8)).clip(lower=0.1, upper=3.0)

                    # Spread = log(P1) - hedge_ratio * log(P2)
                    spread = log_prices[s1] - hedge_ratio * log_prices[s2]
                    spread_mean = spread.rolling(w).mean()
                    spread_std = spread.rolling(w).std() + 1e-8
                    z_score = (spread - spread_mean) / spread_std
                    
                    features[f"spread_zscore_{s1}_{s2}_w{w}"] = z_score

        # 4. Lead-Lag Competitor Returns (Asymmetric information / propagation delay)
        for lag in self.lead_lag_lags:
            for comp in self.universe.direct_competitors:
                if comp in daily_returns.columns:
                    features[f"lead_lag_{comp}_lag{lag}"] = daily_returns[comp].shift(lag)

            if "XLP" in daily_returns.columns:
                features[f"lead_lag_XLP_lag{lag}"] = daily_returns["XLP"].shift(lag)

        # 5. Rolling Correlations & Volatility Ratios
        for w in [21, 63]:
            # Rolling volatility
            ko_vol = daily_returns[target].rolling(w).std() * np.sqrt(252)
            features[f"vol_KO_w{w}"] = ko_vol
            
            if "PEP" in daily_returns.columns:
                pep_vol = daily_returns["PEP"].rolling(w).std() * np.sqrt(252)
                features[f"vol_ratio_KO_PEP_w{w}"] = ko_vol / (pep_vol + 1e-8)
                features[f"corr_KO_PEP_w{w}"] = daily_returns[target].rolling(w).corr(daily_returns["PEP"])

            if "XLP" in daily_returns.columns:
                features[f"corr_KO_XLP_w{w}"] = daily_returns[target].rolling(w).corr(daily_returns["XLP"])

        # 6. Cross-Sectional Dispersion across Beverage Peers
        peer_cols = [c for c in self.universe.direct_competitors if c in daily_returns.columns]
        if peer_cols:
            features["peer_cross_sectional_dispersion"] = daily_returns[peer_cols].std(axis=1)

        # 7. Unsupervised Latent Factors via Rolling PCA on Competitor Returns
        if len(peer_cols) >= 3 and self.pca_components > 0:
            features = self._add_rolling_pca_factors(features, daily_returns[peer_cols], window=63)

        # Drop warm-up NaN rows caused by rolling windows
        features = features.dropna()
        return features

    def _add_rolling_pca_factors(
        self, features: pd.DataFrame, return_matrix: pd.DataFrame, window: int = 63
    ) -> pd.DataFrame:
        """Computes rolling PCA components on peer returns to avoid lookahead."""
        pca_cols = [f"peer_pca_factor_{i+1}" for i in range(self.pca_components)]
        pca_df = pd.DataFrame(index=return_matrix.index, columns=pca_cols, dtype=float)

        for i in range(window, len(return_matrix)):
            window_slice = return_matrix.iloc[i - window : i].dropna()
            if len(window_slice) >= window * 0.8:
                pca = PCA(n_components=self.pca_components)
                pca.fit(window_slice)
                current_row = return_matrix.iloc[[i]]
                if not current_row.isna().any().any():
                    transformed = pca.transform(current_row)[0]
                    pca_df.iloc[i] = transformed

        for col in pca_cols:
            features[col] = pca_df[col].astype(float)

        return features
