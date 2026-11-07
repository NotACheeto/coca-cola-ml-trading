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
        f_dict = {}

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
                f_dict[f"rel_mom_PEP_w{w}"] = target_ret - pep_ret

            # Sector benchmark spread (KO vs XLP)
            if "XLP" in prices.columns:
                xlp_ret = prices["XLP"].pct_change(w)
                f_dict[f"rel_mom_XLP_w{w}"] = target_ret - xlp_ret

            # Peer group average momentum
            peer_cols = [c for c in self.universe.direct_competitors if c in prices.columns]
            if peer_cols:
                peer_mean_ret = prices[peer_cols].pct_change(w).mean(axis=1)
                f_dict[f"rel_mom_peer_mean_w{w}"] = target_ret - peer_mean_ret

            # Staples peers basket momentum (PG, MDLZ, KMB, GIS, CL)
            staples_cols = [c for c in getattr(self.universe, "staples_peers", []) if c in prices.columns]
            if staples_cols:
                staples_mean_ret = prices[staples_cols].pct_change(w).mean(axis=1)
                f_dict[f"rel_mom_staples_peers_w{w}"] = target_ret - staples_mean_ret

            # Supplier momentum (Ball Corp / Crown Holdings) - Packaging cost indicator
            supplier_cols = [c for c in self.universe.suppliers if c in prices.columns]
            if supplier_cols:
                supplier_mean_ret = prices[supplier_cols].pct_change(w).mean(axis=1)
                f_dict[f"rel_mom_suppliers_w{w}"] = target_ret - supplier_mean_ret

            # Bottler momentum (COKE / KOF) - Downstream distribution indicator
            bottler_cols = [c for c in self.universe.bottlers if c in prices.columns]
            if bottler_cols:
                bottler_mean_ret = prices[bottler_cols].pct_change(w).mean(axis=1)
                f_dict[f"rel_mom_bottlers_w{w}"] = target_ret - bottler_mean_ret

            # Commodity Input Cost Pressures (Sugar CANE, Ag DBA, Oil USO for PET bottles, Aluminum DBB)
            for comm in getattr(self.universe, "commodities", []):
                if comm in prices.columns:
                    f_dict[f"rel_mom_input_{comm}_w{w}"] = target_ret - prices[comm].pct_change(w)

            # Rates & Duration Sensitivity (TLT, IEF - KO as dividend bond-proxy)
            if "TLT" in prices.columns:
                f_dict[f"rel_mom_TLT_w{w}"] = target_ret - prices["TLT"].pct_change(w)
            if "IEF" in prices.columns:
                f_dict[f"rel_mom_IEF_w{w}"] = target_ret - prices["IEF"].pct_change(w)

            # FX & Emerging Market Exposure (UUP for USD strength headwind, EEM for emerging growth)
            if "UUP" in prices.columns:
                f_dict[f"mom_UUP_w{w}"] = prices["UUP"].pct_change(w)
            if "EEM" in prices.columns and "SPY" in prices.columns:
                f_dict[f"rel_mom_EEM_SPY_w{w}"] = prices["EEM"].pct_change(w) - prices["SPY"].pct_change(w)

            # Corporate Credit Appetite Spread (HYG vs IEF)
            if "HYG" in prices.columns and "IEF" in prices.columns:
                f_dict[f"credit_appetite_spread_w{w}"] = prices["HYG"].pct_change(w) - prices["IEF"].pct_change(w)

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
                    
                    f_dict[f"spread_zscore_{s1}_{s2}_w{w}"] = z_score

        # 4. Lead-Lag Competitor Returns (Asymmetric information / propagation delay)
        for lag in self.lead_lag_lags:
            for comp in self.universe.direct_competitors:
                if comp in daily_returns.columns:
                    f_dict[f"lead_lag_{comp}_lag{lag}"] = daily_returns[comp].shift(lag)

            if "XLP" in daily_returns.columns:
                f_dict[f"lead_lag_XLP_lag{lag}"] = daily_returns["XLP"].shift(lag)

            if "UUP" in daily_returns.columns:
                f_dict[f"lead_lag_UUP_lag{lag}"] = daily_returns["UUP"].shift(lag)

            if "TLT" in daily_returns.columns:
                f_dict[f"lead_lag_TLT_lag{lag}"] = daily_returns["TLT"].shift(lag)

        # 5. Technical Momentum, Mean Reversion & Volatility Ratios
        # 14-day RSI for target KO and benchmark XLP
        f_dict["rsi_KO_14"] = self._compute_rsi(prices[target], period=14)
        if "XLP" in prices.columns:
            rsi_xlp = self._compute_rsi(prices["XLP"], period=14)
            f_dict["rsi_XLP_14"] = rsi_xlp
            f_dict["rsi_spread_KO_XLP"] = f_dict["rsi_KO_14"] - rsi_xlp

        for w in [21, 63]:
            # Rolling volatility
            ko_vol = daily_returns[target].rolling(w).std() * np.sqrt(252)
            f_dict[f"vol_KO_w{w}"] = ko_vol
            
            # EMA distance z-scores
            ema = prices[target].ewm(span=w, adjust=False).mean()
            roll_std = prices[target].rolling(w).std() + 1e-8
            f_dict[f"ema_dist_zscore_w{w}"] = (prices[target] - ema) / roll_std

            if "PEP" in daily_returns.columns:
                pep_vol = daily_returns["PEP"].rolling(w).std() * np.sqrt(252)
                f_dict[f"vol_ratio_KO_PEP_w{w}"] = ko_vol / (pep_vol + 1e-8)
                f_dict[f"corr_KO_PEP_w{w}"] = daily_returns[target].rolling(w).corr(daily_returns["PEP"])

            if "XLP" in daily_returns.columns:
                f_dict[f"corr_KO_XLP_w{w}"] = daily_returns[target].rolling(w).corr(daily_returns["XLP"])

            if "TLT" in daily_returns.columns:
                f_dict[f"corr_KO_TLT_w{w}"] = daily_returns[target].rolling(w).corr(daily_returns["TLT"])

            if "UUP" in daily_returns.columns:
                f_dict[f"corr_KO_UUP_w{w}"] = daily_returns[target].rolling(w).corr(daily_returns["UUP"])

        # 6. Cross-Sectional Dispersion across Beverage Peers
        peer_cols = [c for c in self.universe.direct_competitors if c in daily_returns.columns]
        if peer_cols:
            f_dict["peer_cross_sectional_dispersion"] = daily_returns[peer_cols].std(axis=1)

        # Assemble full features DataFrame in one pass to avoid fragmentation
        features = pd.DataFrame(f_dict, index=prices.index)

        # 7. Unsupervised Latent Factors via Rolling PCA on Competitor Returns
        if len(peer_cols) >= 3 and self.pca_components > 0:
            features = self._add_rolling_pca_factors(features, daily_returns[peer_cols], window=63)

        # Drop warm-up NaN rows caused by rolling windows
        features = features.dropna()
        return features

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Computes backward-looking Relative Strength Index (RSI)."""
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

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

        # Concat once
        features = pd.concat([features, pca_df.astype(float)], axis=1)
        return features
