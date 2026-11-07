"""
Institutional Quantitative Factor Engineering Machine for Coca-Cola ($KO).
Computes 200+ multi-scale alpha factors across microstructure, volatility estimators,
commodity cost pressures, yield curve dynamics, FX drag, technical momentum, and cross-asset stat-arb.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.data.universe import AssetUniverse


class FeatureEngineer:
    """Computes comprehensive quantitative alpha factors and cross-asset dynamics for trading Coca-Cola ($KO)."""

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
        self.cointegration_pairs = cointegration_pairs or [
            ("KO", "PEP"),
            ("KO", "XLP"),
            ("KO", "COKE"),
            ("KO", "KOF"),
            ("KO", "TLT"),
        ]
        self.pca_components = pca_components

    def compute_features(
        self,
        prices: pd.DataFrame,
        ohlcv: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Transforms raw price and OHLCV data into a massive institutional predictive factor matrix.
        Strictly respects temporal order (zero forward-looking leakage).
        """
        target = self.universe.target
        f_dict = {}

        # 1. Base Log Returns & Continuous Series
        log_prices = np.log(prices)
        daily_returns = prices.pct_change()
        daily_log_returns = log_prices.diff()
        target_series = prices[target]

        # 2. Target Microstructure, Range, & Advanced Volatility Estimators (if OHLCV provided)
        if ohlcv is not None and not ohlcv.empty and all(c in ohlcv.columns for c in ["Open", "High", "Low", "Close"]):
            o = ohlcv["Open"]
            h = ohlcv["High"]
            l = ohlcv["Low"]
            c = ohlcv["Close"]
            v = ohlcv["Volume"] if "Volume" in ohlcv.columns else pd.Series(1.0, index=ohlcv.index)

            # Intraday Price Range & Gaps
            f_dict["intraday_range_hl"] = np.log(h / (l + 1e-8))
            f_dict["overnight_gap"] = (o / c.shift(1) - 1.0).fillna(0.0)
            f_dict["intraday_ret"] = (c / o - 1.0).fillna(0.0)
            f_dict["close_location_value"] = ((c - l) / (h - l + 1e-8)).clip(0.0, 1.0)

            # Parkinson Volatility: sqrt(1/(4*ln2) * ln(H/L)^2)
            log_hl = np.log(h / (l + 1e-8))
            park_var_daily = (log_hl ** 2) / (4.0 * np.log(2.0))
            for w in [5, 10, 21, 63]:
                f_dict[f"vol_parkinson_w{w}"] = np.sqrt(park_var_daily.rolling(w).mean() * 252.0)

            # Garman-Klass Volatility: 0.5 * ln(H/L)^2 - (2*ln2 - 1) * ln(C/O)^2
            log_co = np.log(c / (o + 1e-8))
            gk_var_daily = np.maximum(0.0, 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2))
            for w in [5, 10, 21, 63]:
                f_dict[f"vol_garman_klass_w{w}"] = np.sqrt(gk_var_daily.rolling(w).mean() * 252.0)

            # Average True Range (ATR)
            tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
            for w in [7, 14, 21, 63]:
                atr = tr.rolling(w).mean()
                f_dict[f"atr_norm_w{w}"] = atr / (c + 1e-8)

            # Volume & Flow Dynamics
            if "Volume" in ohlcv.columns:
                for w in [10, 20, 63]:
                    v_mean = v.rolling(w).mean()
                    v_std = v.rolling(w).std() + 1e-8
                    f_dict[f"volume_zscore_w{w}"] = (v - v_mean) / v_std
                    f_dict[f"volume_ratio_median_w{w}"] = v / (v.rolling(w).median() + 1e-8)

                # On-Balance Volume (OBV)
                obv_dir = np.sign(c.diff().fillna(0.0))
                obv = (obv_dir * v).cumsum()
                f_dict["obv_roc_w5"] = obv.pct_change(5)
                f_dict["obv_roc_w21"] = obv.pct_change(21)

                # Chaikin Money Flow (CMF) 21-day
                clv = ((c - l) - (h - c)) / (h - l + 1e-8)
                f_dict["cmf_21"] = (clv * v).rolling(21).sum() / (v.rolling(21).sum() + 1e-8)

                # Amihud Illiquidity
                f_dict["amihud_illiquidity_21"] = (
                    (daily_returns[target].abs() / (v * c / 1e6 + 1e-8)).rolling(21).mean()
                )

                # Volume-Weighted Average Price (VWAP) 21-day distance
                cum_pv = (c * v).rolling(21).sum()
                cum_v = v.rolling(21).sum() + 1e-8
                vwap_21 = cum_pv / cum_v
                f_dict["vwap_dist_zscore_21"] = (c - vwap_21) / (tr.rolling(21).mean() + 1e-8)

        # 3. Multi-Scale Technical Oscillators & Trend Momentum
        # RSI at 7, 14, 21, 28 days
        for rsi_w in [7, 14, 21, 28]:
            f_dict[f"rsi_KO_w{rsi_w}"] = self._compute_rsi(target_series, period=rsi_w)

        if "XLP" in prices.columns:
            f_dict["rsi_XLP_14"] = self._compute_rsi(prices["XLP"], period=14)
            f_dict["rsi_spread_KO_XLP"] = f_dict["rsi_KO_w14"] - f_dict["rsi_XLP_14"]

        if "PEP" in prices.columns:
            f_dict["rsi_PEP_14"] = self._compute_rsi(prices["PEP"], period=14)
            f_dict["rsi_spread_KO_PEP"] = f_dict["rsi_KO_w14"] - f_dict["rsi_PEP_14"]

        # MACD (12, 26, 9)
        ema_12 = target_series.ewm(span=12, adjust=False).mean()
        ema_26 = target_series.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        f_dict["macd_line"] = macd_line / target_series
        f_dict["macd_signal"] = macd_signal / target_series
        f_dict["macd_hist"] = (macd_line - macd_signal) / target_series

        # Stochastic Oscillator %K (14) and %D (3)
        low_14 = target_series.rolling(14).min()
        high_14 = target_series.rolling(14).max()
        stoch_k = 100.0 * (target_series - low_14) / (high_14 - low_14 + 1e-8)
        f_dict["stoch_k_14"] = stoch_k
        f_dict["stoch_d_3"] = stoch_k.rolling(3).mean()

        # Bollinger Bands %B and Bandwidth over 20 and 63 days
        for bb_w in [20, 63]:
            bb_mid = target_series.rolling(bb_w).mean()
            bb_std = target_series.rolling(bb_w).std() + 1e-8
            bb_up = bb_mid + 2.0 * bb_std
            bb_low = bb_mid - 2.0 * bb_std
            f_dict[f"bb_pct_b_w{bb_w}"] = (target_series - bb_low) / (bb_up - bb_low + 1e-8)
            f_dict[f"bb_bandwidth_w{bb_w}"] = (bb_up - bb_low) / (bb_mid + 1e-8)

        # Rate of Change (ROC)
        for roc_w in [1, 2, 3, 5, 10, 21, 63]:
            f_dict[f"roc_w{roc_w}"] = target_series.pct_change(roc_w)

        # Moving Average Distance Z-Scores & Crossover Spreads
        for ma_w in [5, 10, 21, 50, 100, 200]:
            ma = target_series.ewm(span=ma_w, adjust=False).mean()
            m_std = target_series.rolling(ma_w, min_periods=min(ma_w, 20)).std() + 1e-8
            f_dict[f"ema_dist_zscore_w{ma_w}"] = (target_series - ma) / m_std

        ema_5 = target_series.ewm(span=5, adjust=False).mean()
        ema_20 = target_series.ewm(span=20, adjust=False).mean()
        ema_50 = target_series.ewm(span=50, adjust=False).mean()
        ema_200 = target_series.ewm(span=200, adjust=False).mean()
        f_dict["ema_cross_5_20"] = (ema_5 - ema_20) / target_series
        f_dict["ema_cross_20_50"] = (ema_20 - ema_50) / target_series
        f_dict["ema_cross_50_200"] = (ema_50 - ema_200) / target_series

        # 4. Cross-Sectional Ranking & Peer Return Dispersion
        all_peers = [c for c in self.universe.direct_competitors + getattr(self.universe, "staples_peers", []) if c in prices.columns]
        if all_peers:
            for rw in [1, 5, 21]:
                ret_sub = prices[[target] + all_peers].pct_change(rw)
                rank_df = ret_sub.rank(axis=1, pct=True)
                f_dict[f"cross_sectional_rank_KO_w{rw}"] = rank_df[target]
                f_dict[f"peer_dispersion_w{rw}"] = ret_sub[all_peers].std(axis=1)

        # 5. Competitor & Sector Relative Momentum
        for w in self.rolling_windows:
            target_ret = prices[target].pct_change(w)

            # Pairwise relative momentum against direct rivals
            for rival in self.universe.direct_competitors:
                if rival in prices.columns:
                    f_dict[f"rel_mom_{rival}_w{w}"] = target_ret - prices[rival].pct_change(w)

            # Benchmark alpha spread
            if "XLP" in prices.columns:
                f_dict[f"rel_mom_XLP_w{w}"] = target_ret - prices["XLP"].pct_change(w)

            # Staples peers basket average
            staples_cols = [c for c in getattr(self.universe, "staples_peers", []) if c in prices.columns]
            if staples_cols:
                f_dict[f"rel_mom_staples_basket_w{w}"] = target_ret - prices[staples_cols].pct_change(w).mean(axis=1)

            # Bottlers basket
            bottler_cols = [c for c in self.universe.bottlers if c in prices.columns]
            if bottler_cols:
                f_dict[f"rel_mom_bottlers_basket_w{w}"] = target_ret - prices[bottler_cols].pct_change(w).mean(axis=1)

            # Packaging suppliers basket
            supplier_cols = [c for c in self.universe.suppliers if c in prices.columns]
            if supplier_cols:
                f_dict[f"rel_mom_suppliers_basket_w{w}"] = target_ret - prices[supplier_cols].pct_change(w).mean(axis=1)

            # Commodity Input Cost Pressures (Sugar CANE, Ag DBA, Oil USO, Aluminum DBB)
            for comm in getattr(self.universe, "commodities", []):
                if comm in prices.columns:
                    f_dict[f"rel_mom_input_{comm}_w{w}"] = target_ret - prices[comm].pct_change(w)

            # Rates & Duration Sensitivity (TLT, IEF - KO as dividend bond-proxy)
            if "TLT" in prices.columns:
                f_dict[f"rel_mom_TLT_w{w}"] = target_ret - prices["TLT"].pct_change(w)
            if "IEF" in prices.columns:
                f_dict[f"rel_mom_IEF_w{w}"] = target_ret - prices["IEF"].pct_change(w)

            # Yield curve slope proxy (TLT - IEF)
            if "TLT" in prices.columns and "IEF" in prices.columns:
                f_dict[f"yield_curve_slope_w{w}"] = prices["TLT"].pct_change(w) - prices["IEF"].pct_change(w)

            # FX & Emerging Market Exposure
            if "UUP" in prices.columns:
                f_dict[f"mom_UUP_w{w}"] = prices["UUP"].pct_change(w)
            if "EEM" in prices.columns and "SPY" in prices.columns:
                f_dict[f"rel_mom_EEM_SPY_w{w}"] = prices["EEM"].pct_change(w) - prices["SPY"].pct_change(w)

            # Credit Risk Appetite Spread (HYG vs IEF)
            if "HYG" in prices.columns and "IEF" in prices.columns:
                f_dict[f"credit_appetite_spread_w{w}"] = prices["HYG"].pct_change(w) - prices["IEF"].pct_change(w)

        # 6. Dynamic Rolling Betas & Correlations
        for w in [21, 63]:
            # Rolling target volatility
            ko_vol = daily_returns[target].rolling(w).std() * np.sqrt(252.0)
            f_dict[f"vol_KO_w{w}"] = ko_vol

            # Rolling betas & correlations to key assets
            for ref_asset in ["PEP", "XLP", "SPY", "TLT", "UUP", "CANE", "USO"]:
                if ref_asset in daily_returns.columns:
                    cov = daily_returns[target].rolling(w).cov(daily_returns[ref_asset])
                    var_ref = daily_returns[ref_asset].rolling(w).var() + 1e-8
                    f_dict[f"beta_KO_{ref_asset}_w{w}"] = (cov / var_ref).clip(-4.0, 4.0)
                    f_dict[f"corr_KO_{ref_asset}_w{w}"] = daily_returns[target].rolling(w).corr(daily_returns[ref_asset])

        # 7. Statistical Arbitrage Cointegration Spreads & Rolling Z-Scores
        for pair in self.cointegration_pairs:
            s1, s2 = pair[0], pair[1]
            if s1 in log_prices.columns and s2 in log_prices.columns:
                for w in [21, 63]:
                    roll_cov = daily_log_returns[s1].rolling(w).cov(daily_log_returns[s2])
                    roll_var = daily_log_returns[s2].rolling(w).var()
                    hedge_ratio = (roll_cov / (roll_var + 1e-8)).clip(lower=0.1, upper=3.0)

                    spread = log_prices[s1] - hedge_ratio * log_prices[s2]
                    spread_mean = spread.rolling(w).mean()
                    spread_std = spread.rolling(w).std() + 1e-8
                    z_score = (spread - spread_mean) / spread_std
                    f_dict[f"spread_zscore_{s1}_{s2}_w{w}"] = z_score

        # 8. Lead-Lag Competitor Returns (Asymmetric information / propagation delay)
        for lag in self.lead_lag_lags:
            for lead_sym in ["PEP", "MNST", "CELH", "XLP", "UUP", "TLT", "CANE", "USO"]:
                if lead_sym in daily_returns.columns:
                    f_dict[f"lead_lag_{lead_sym}_lag{lag}"] = daily_returns[lead_sym].shift(lag)

        # 9. Fractional Differentiation (Preserving long-memory price stationarity)
        frac_diff = self._fractional_diff(target_series, d=0.40)
        f_dict["frac_diff_KO_d40"] = frac_diff

        # Assemble full features DataFrame in one pass to avoid fragmentation
        features = pd.DataFrame(f_dict, index=prices.index)

        # 10. Unsupervised Latent Factors via Rolling PCA on Competitor Returns
        peer_cols = [c for c in self.universe.direct_competitors if c in daily_returns.columns]
        if len(peer_cols) >= 3 and self.pca_components > 0:
            features = self._add_rolling_pca_factors(features, daily_returns[peer_cols], prefix="peer_pca", window=63)

        # Macro/Commodity PCA
        macro_cols = [c for c in ["TLT", "IEF", "HYG", "UUP", "CANE", "USO", "DBB", "XLP"] if c in daily_returns.columns]
        if len(macro_cols) >= 3 and self.pca_components > 0:
            features = self._add_rolling_pca_factors(features, daily_returns[macro_cols], prefix="macro_pca", window=63)

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

    @staticmethod
    def _fractional_diff(series: pd.Series, d: float = 0.40, thres: float = 1e-3, max_lag: int = 30) -> pd.Series:
        """Computes fractional differentiation with fixed window memory weights (López de Prado)."""
        weights = [1.0]
        max_k = min(len(series) // 2, max_lag)
        for k in range(1, max_k):
            w = -weights[-1] / k * (d - k + 1)
            if abs(w) < thres:
                break
            weights.append(w)

        w_arr = np.array(weights)
        diff_vals = pd.Series(index=series.index, dtype=float)
        vals = series.values
        w_len = len(weights)
        for i in range(w_len - 1, len(series)):
            window = vals[i - w_len + 1 : i + 1][::-1]
            diff_vals.iloc[i] = np.dot(w_arr, window)

        return diff_vals

    def _add_rolling_pca_factors(
        self, features: pd.DataFrame, return_matrix: pd.DataFrame, prefix: str = "pca", window: int = 63
    ) -> pd.DataFrame:
        """Computes rolling PCA components to avoid lookahead."""
        pca_cols = [f"{prefix}_factor_{i+1}" for i in range(self.pca_components)]
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

        features = pd.concat([features, pca_df.astype(float)], axis=1)
        return features
