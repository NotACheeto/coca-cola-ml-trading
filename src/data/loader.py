"""
Data ingestion, caching, and preprocessing for multi-asset universe.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from src.data.universe import AssetUniverse


class DataLoader:
    """Fetches, cleans, aligns, and caches historical price & volume data."""

    def __init__(
        self,
        universe: AssetUniverse,
        cache_dir: Optional[str] = "data/cache",
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None,
        bars_per_day: int = 3,
        session_checkpoints: Optional[List[str]] = None,
    ):
        self.universe = universe
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.start_date = start_date
        self.end_date = end_date
        self.bars_per_day = bars_per_day
        self.session_checkpoints = session_checkpoints or ["11:30", "13:30", "16:00"]

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_daily_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetches adjusted close prices and volumes for all tickers in universe on daily basis.
        Returns an aligned wide DataFrame of Adjusted Closes.
        """
        cache_file = self.cache_dir / "prices_cache.parquet" if self.cache_dir else None

        if use_cache and cache_file and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty and all(t in df.columns for t in self.universe.all_tickers):
                    if self.start_date:
                        df = df[df.index >= pd.to_datetime(self.start_date)]
                    if self.end_date:
                        df = df[df.index <= pd.to_datetime(self.end_date)]
                    return df
            except Exception:
                pass  # Fall back to live download if cache is corrupt

        tickers = self.universe.all_tickers
        try:
            raw_data = yf.download(
                tickers=tickers,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
                progress=False,
            )
            if raw_data.empty:
                raise ValueError("Yahoo Finance returned an empty dataset.")

            # Extract Adjusted Close
            if "Adj Close" in raw_data.columns:
                adj_close = raw_data["Adj Close"].copy()
            elif "Close" in raw_data.columns:
                adj_close = raw_data["Close"].copy()
            else:
                raise ValueError("Neither 'Adj Close' nor 'Close' found in data.")

            # Format columns
            if isinstance(adj_close, pd.Series):
                adj_close = adj_close.to_frame(name=tickers[0])

            # Ensure proper timezone-naive DatetimeIndex
            adj_close.index = pd.to_datetime(adj_close.index).tz_localize(None)

            # Extract full OHLCV for target asset (KO) for deep microstructure & volatility factor modeling
            target = self.universe.target
            target_ohlcv = pd.DataFrame(index=raw_data.index)
            for field in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                if field in raw_data.columns:
                    col_data = raw_data[field]
                    if isinstance(col_data, pd.DataFrame) and target in col_data.columns:
                        target_ohlcv[field] = col_data[target]
                    elif isinstance(col_data, pd.Series):
                        target_ohlcv[field] = col_data

            if not target_ohlcv.empty:
                target_ohlcv.index = pd.to_datetime(target_ohlcv.index).tz_localize(None)
                target_ohlcv = target_ohlcv.sort_index().ffill().bfill()
                ohlcv_cache_file = self.cache_dir / "target_ohlcv.parquet" if self.cache_dir else None
                if ohlcv_cache_file:
                    if ohlcv_cache_file.exists():
                        try:
                            cached_ohlcv = pd.read_parquet(ohlcv_cache_file)
                            target_ohlcv = target_ohlcv.combine_first(cached_ohlcv).sort_index()
                            target_ohlcv = target_ohlcv[~target_ohlcv.index.duplicated(keep="last")]
                        except Exception:
                            pass
                    target_ohlcv.to_parquet(ohlcv_cache_file)

            # Clean and align data
            adj_close = adj_close.sort_index()
            adj_close = adj_close.ffill().bfill()
            if self.universe.target in adj_close.columns:
                adj_close = adj_close.dropna(subset=[self.universe.target])

            # Preserve master cache: combine with existing historical data if present
            if cache_file:
                if cache_file.exists():
                    try:
                        cached_master = pd.read_parquet(cache_file)
                        combined_master = adj_close.combine_first(cached_master).sort_index()
                        combined_master = combined_master[~combined_master.index.duplicated(keep="last")]
                        combined_master.to_parquet(cache_file)
                    except Exception:
                        adj_close.to_parquet(cache_file)
                else:
                    adj_close.to_parquet(cache_file)

            # Return requested window
            out_df = adj_close.copy()
            if self.start_date:
                out_df = out_df[out_df.index >= pd.to_datetime(self.start_date)]
            if self.end_date:
                out_df = out_df[out_df.index <= pd.to_datetime(self.end_date)]

            return out_df

        except Exception as e:
            # Fallback for offline testing or rate-limits: generate realistic synthetic series
            print(f"[Warning] Failed to fetch online data ({e}). Generating synthetic data.")
            return self._generate_synthetic_data()

    def fetch_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetches adjusted close prices for all tickers in universe.
        If bars_per_day == 3, returns 3 intraday session checkpoints per day:
        - Morning (11:30), Midday (13:30), Close (16:00).
        Otherwise returns daily bars.
        """
        if self.bars_per_day == 3:
            return self.fetch_3session_data(use_cache=use_cache)
        return self.fetch_daily_data(use_cache=use_cache)

    def fetch_3session_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetches or constructs 3-session intraday prices per trading day.
        Checkpoints: 11:30 (Morning), 13:30 (Midday), 16:00 (Close).
        """
        cache_file = self.cache_dir / "prices_3sessions.parquet" if self.cache_dir else None
        if use_cache and cache_file and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty and all(t in df.columns for t in self.universe.all_tickers):
                    if self.start_date:
                        df = df[df.index >= pd.to_datetime(self.start_date)]
                    if self.end_date:
                        df = df[df.index <= pd.to_datetime(self.end_date)]
                    return df
            except Exception:
                pass

        daily_df = self.fetch_daily_data(use_cache=use_cache)
        session_df = self._construct_3session_prices(daily_df)

        if cache_file:
            session_df.to_parquet(cache_file)

        out_df = session_df.copy()
        if self.start_date:
            out_df = out_df[out_df.index >= pd.to_datetime(self.start_date)]
        if self.end_date:
            out_df = out_df[out_df.index <= pd.to_datetime(self.end_date)]
        return out_df

    def fetch_target_ohlcv(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetches full Open, High, Low, Close, Volume history for target asset (KO).
        If bars_per_day == 3, returns 3-session intraday OHLCV.
        """
        if self.bars_per_day == 3:
            return self.fetch_3session_target_ohlcv(use_cache=use_cache)
        return self.fetch_daily_target_ohlcv(use_cache=use_cache)

    def fetch_3session_target_ohlcv(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetches or constructs 3-session intraday OHLCV for KO.
        """
        cache_file = self.cache_dir / "target_ohlcv_3sessions.parquet" if self.cache_dir else None
        if use_cache and cache_file and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty and "Close" in df.columns:
                    if self.start_date:
                        df = df[df.index >= pd.to_datetime(self.start_date)]
                    if self.end_date:
                        df = df[df.index <= pd.to_datetime(self.end_date)]
                    return df
            except Exception:
                pass

        daily_prices = self.fetch_daily_data(use_cache=use_cache)
        daily_ohlcv = self.fetch_daily_target_ohlcv(use_cache=use_cache)
        session_ohlcv = self._construct_3session_target_ohlcv(daily_prices, daily_ohlcv)

        if cache_file:
            session_ohlcv.to_parquet(cache_file)

        out_df = session_ohlcv.copy()
        if self.start_date:
            out_df = out_df[out_df.index >= pd.to_datetime(self.start_date)]
        if self.end_date:
            out_df = out_df[out_df.index <= pd.to_datetime(self.end_date)]
        return out_df

    def fetch_daily_target_ohlcv(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetches full Open, High, Low, Close, Volume history for target asset (KO) on daily basis.
        """
        ohlcv_cache = self.cache_dir / "target_ohlcv.parquet" if self.cache_dir else None
        if use_cache and ohlcv_cache and ohlcv_cache.exists():
            try:
                df = pd.read_parquet(ohlcv_cache)
                if not df.empty and "Close" in df.columns:
                    if self.start_date:
                        df = df[df.index >= pd.to_datetime(self.start_date)]
                    if self.end_date:
                        df = df[df.index <= pd.to_datetime(self.end_date)]
                    return df
            except Exception:
                pass

        # If cache not populated, trigger fetch_daily_data
        self.fetch_daily_data(use_cache=use_cache)
        if ohlcv_cache and ohlcv_cache.exists():
            try:
                df = pd.read_parquet(ohlcv_cache)
                if self.start_date:
                    df = df[df.index >= pd.to_datetime(self.start_date)]
                if self.end_date:
                    df = df[df.index <= pd.to_datetime(self.end_date)]
                return df
            except Exception:
                pass

        synth_ohlcv = self._generate_synthetic_target_ohlcv()
        if ohlcv_cache:
            synth_ohlcv.to_parquet(ohlcv_cache)
        return synth_ohlcv

    def _construct_3session_prices(self, daily_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Decomposes daily price series into 3 intraday session checkpoints per day:
        - Session 0: 11:30:00 ET (Morning / Post-Open price discovery & European close)
        - Session 1: 13:30:00 ET (Midday / Post-Lunch consolidation & Treasury auctions)
        - Session 2: 16:00:00 ET (Afternoon / Market-on-Close official closing auction)
        Strictly guarantees exact mathematical closure at Session 2: P(16:00) == Daily Close.
        """
        dates = daily_prices.index
        cols = daily_prices.columns
        n_days = len(dates)
        n_assets = len(cols)
        rng = np.random.default_rng(42)

        timestamps = []
        for d in dates:
            d_str = d.strftime("%Y-%m-%d")
            timestamps.append(pd.Timestamp(f"{d_str} 11:30:00"))
            timestamps.append(pd.Timestamp(f"{d_str} 13:30:00"))
            timestamps.append(pd.Timestamp(f"{d_str} 16:00:00"))

        log_p = np.log(daily_prices.values)
        log_ret = np.zeros_like(log_p)
        log_ret[1:] = log_p[1:] - log_p[:-1]

        session_prices = np.zeros((n_days * 3, n_assets))

        # Initial day sessions
        session_prices[0] = daily_prices.values[0] * np.exp(-0.0015)
        session_prices[1] = daily_prices.values[0] * np.exp(-0.0007)
        session_prices[2] = daily_prices.values[0]

        for t in range(1, n_days):
            tot_r = log_ret[t]
            noise_1 = rng.normal(0, 0.0012, size=n_assets)
            noise_2 = rng.normal(0, 0.0008, size=n_assets)

            r_s1 = 0.45 * tot_r + noise_1
            r_s2 = 0.20 * tot_r + noise_2
            r_s3 = tot_r - r_s1 - r_s2  # Exact closure

            prev_close = session_prices[(t - 1) * 3 + 2]
            p_s1 = prev_close * np.exp(r_s1)
            p_s2 = p_s1 * np.exp(r_s2)
            p_s3 = p_s2 * np.exp(r_s3)

            session_prices[t * 3] = p_s1
            session_prices[t * 3 + 1] = p_s2
            session_prices[t * 3 + 2] = p_s3

        df_sessions = pd.DataFrame(session_prices, index=pd.DatetimeIndex(timestamps), columns=cols)
        return df_sessions

    def _construct_3session_target_ohlcv(
        self, daily_prices: pd.DataFrame, daily_target_ohlcv: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Constructs realistic Open, High, Low, Close, and Volume for each of the 3 daily sessions for KO.
        Models the U-shaped intraday volume profile: 40% Morning, 20% Midday, 40% Close.
        """
        target = self.universe.target
        session_prices = self._construct_3session_prices(daily_prices)
        ko_session_close = session_prices[target].values
        n_bars = len(ko_session_close)
        n_days = n_bars // 3
        rng = np.random.default_rng(42)

        if daily_target_ohlcv is not None and not daily_target_ohlcv.empty:
            aligned_ohlcv = daily_target_ohlcv.reindex(daily_prices.index).ffill().bfill()
            daily_open = aligned_ohlcv["Open"].values
            daily_high = aligned_ohlcv["High"].values
            daily_low = aligned_ohlcv["Low"].values
            daily_vol = aligned_ohlcv["Volume"].values
        else:
            daily_close = daily_prices[target].values
            daily_open = np.zeros(n_days)
            daily_open[0] = daily_close[0]
            daily_open[1:] = daily_close[:-1] * (1.0 + rng.normal(0, 0.002, n_days - 1))
            daily_high = np.maximum(daily_open, daily_close) * (1.0 + np.abs(rng.normal(0, 0.004, n_days)))
            daily_low = np.minimum(daily_open, daily_close) * (1.0 - np.abs(rng.normal(0, 0.004, n_days)))
            daily_vol = rng.lognormal(mean=16.7, sigma=0.4, size=n_days)

        s_open = np.zeros(n_bars)
        s_high = np.zeros(n_bars)
        s_low = np.zeros(n_bars)
        s_close = ko_session_close.copy()
        s_vol = np.zeros(n_bars)

        for t in range(n_days):
            idx0 = t * 3
            idx1 = t * 3 + 1
            idx2 = t * 3 + 2

            d_o = daily_open[t]
            d_h = daily_high[t]
            d_l = daily_low[t]
            d_v = daily_vol[t]

            # Session 0: Morning (11:30)
            c0 = s_close[idx0]
            o0 = d_o
            h0 = max(o0, c0) * (1.0 + np.abs(rng.normal(0, 0.0025)))
            l0 = min(o0, c0) * (1.0 - np.abs(rng.normal(0, 0.0025)))
            v0 = 0.40 * d_v

            # Session 1: Midday (13:30)
            c1 = s_close[idx1]
            o1 = c0
            h1 = max(o1, c1) * (1.0 + np.abs(rng.normal(0, 0.0015)))
            l1 = min(o1, c1) * (1.0 - np.abs(rng.normal(0, 0.0015)))
            v1 = 0.20 * d_v

            # Session 2: Close (16:00)
            c2 = s_close[idx2]
            o2 = c1
            h2 = max(d_h, max(o2, c2))
            l2 = min(d_l, min(o2, c2))
            v2 = 0.40 * d_v

            s_open[idx0], s_high[idx0], s_low[idx0], s_vol[idx0] = o0, h0, l0, v0
            s_open[idx1], s_high[idx1], s_low[idx1], s_vol[idx1] = o1, h1, l1, v1
            s_open[idx2], s_high[idx2], s_low[idx2], s_vol[idx2] = o2, h2, l2, v2

        df_ohlcv = pd.DataFrame(
            {
                "Open": s_open,
                "High": s_high,
                "Low": s_low,
                "Close": s_close,
                "Adj Close": s_close,
                "Volume": s_vol,
            },
            index=session_prices.index,
        )
        return df_ohlcv

    def _generate_synthetic_target_ohlcv(self, n_days: int = 1500) -> pd.DataFrame:
        """Generates realistic synthetic OHLCV data for target asset."""
        rng = np.random.default_rng(42)
        dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n_days, freq="B")
        ret = rng.normal(0.0003, 0.008, n_days)
        close = 50.0 * np.exp(np.cumsum(ret))
        open_p = close * (1.0 + rng.normal(0, 0.002, n_days))
        high_p = np.maximum(open_p, close) * (1.0 + np.abs(rng.normal(0, 0.005, n_days)))
        low_p = np.minimum(open_p, close) * (1.0 - np.abs(rng.normal(0, 0.005, n_days)))
        volume = rng.lognormal(mean=16.7, sigma=0.4, size=n_days)
        return pd.DataFrame(
            {
                "Open": open_p,
                "High": high_p,
                "Low": low_p,
                "Close": close,
                "Adj Close": close,
                "Volume": volume,
            },
            index=dates,
        )

    def _generate_synthetic_data(self, n_days: int = 1500) -> pd.DataFrame:
        """
        Generates realistic co-moving geometric Brownian motion series with cointegration
        and cross-asset correlation across equities, commodities, and rates.
        """
        rng = np.random.default_rng(42)
        dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n_days, freq="B")

        # Latent macro factors
        macro_staples = rng.normal(0.0003, 0.007, n_days)
        interest_rate_shock = rng.normal(0.0001, 0.005, n_days)
        commodity_shock = rng.normal(0.0002, 0.012, n_days)

        data = {}
        for ticker in self.universe.all_tickers:
            # Calibrate factor sensitivities
            if ticker in ["KO", "PEP"]:
                returns = 0.90 * macro_staples - 0.25 * interest_rate_shock + rng.normal(0, 0.006, n_days)
                p0 = 50.0 if ticker == "KO" else 165.0
            elif ticker in ["MNST", "CELH"]:
                returns = 1.20 * macro_staples + rng.normal(0, 0.018, n_days)
                p0 = 55.0 if ticker == "MNST" else 40.0
            elif ticker in ["COKE", "KOF"]:
                returns = 0.85 * macro_staples + rng.normal(0, 0.010, n_days)
                p0 = 800.0 if ticker == "COKE" else 90.0
            elif ticker in ["BALL", "CCK"]:
                returns = 0.70 * macro_staples + 0.35 * commodity_shock + rng.normal(0, 0.012, n_days)
                p0 = 60.0
            elif ticker in ["CANE", "DBA", "USO", "DBB"]:
                returns = 0.95 * commodity_shock + rng.normal(0, 0.010, n_days)
                p0 = 12.0 if ticker == "CANE" else 22.0
            elif ticker in ["TLT", "IEF"]:
                returns = -1.2 * interest_rate_shock + rng.normal(0, 0.004, n_days)
                p0 = 95.0
            elif ticker in ["HYG", "LQD"]:
                returns = 0.40 * macro_staples - 0.50 * interest_rate_shock + rng.normal(0, 0.003, n_days)
                p0 = 75.0 if ticker == "HYG" else 105.0
            elif ticker == "XLP":
                returns = 1.0 * macro_staples + rng.normal(0, 0.003, n_days)
                p0 = 75.0
            elif ticker == "UUP":
                returns = -0.30 * macro_staples + 0.30 * interest_rate_shock + rng.normal(0, 0.004, n_days)
                p0 = 28.0
            else:
                # Other peers (PG, MDLZ, KMB, GIS, CL, SPY, EEM, VIXY)
                returns = 0.85 * macro_staples + rng.normal(0, 0.008, n_days)
                p0 = 80.0

            data[ticker] = p0 * np.exp(np.cumsum(returns))

        df = pd.DataFrame(data, index=dates)
        return df

    def save_prices(self, df: pd.DataFrame) -> Path:
        """Saves price matrix to disk cache, merging with existing data."""
        if not self.cache_dir:
            self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / "prices_cache.parquet"
        if cache_file.exists():
            try:
                old_df = pd.read_parquet(cache_file)
                df = df.combine_first(old_df).sort_index()
            except Exception:
                pass
        df.to_parquet(cache_file)
        return cache_file

    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Loads user-provided CSV file containing date index and ticker price columns.
        Validates columns, converts date index, merges with cache if needed, and saves.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Custom data file not found: {file_path}")

        df = pd.read_csv(path)
        date_col = None
        for col in ["Date", "date", "DATETIME", "timestamp", "Timestamp"]:
            if col in df.columns:
                date_col = col
                break

        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        else:
            # Assume first column is date
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0])

        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index().ffill().bfill()
        
        # Merge with existing cache if available
        cache_file = self.cache_dir / "prices_cache.parquet" if self.cache_dir else None
        if cache_file and cache_file.exists():
            try:
                cached_df = pd.read_parquet(cache_file)
                df = df.combine_first(cached_df).sort_index()
            except Exception:
                pass

        self.save_prices(df)
        return df

    def update_latest(self) -> Tuple[pd.DataFrame, int]:
        """
        Fetches newest daily bars from Yahoo Finance and merges into existing cache.
        Returns:
            updated_df: DataFrame with updated price series
            new_bars_count: Number of new trading bars ingested
        """
        current_data = self.fetch_data(use_cache=True)
        latest_date = current_data.index[-1]
        
        # Download from latest_date onwards
        start_fetch = (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        today_str = pd.Timestamp.now().strftime("%Y-%m-%d")

        if start_fetch >= today_str:
            return current_data, 0

        try:
            new_raw = yf.download(
                tickers=self.universe.all_tickers,
                start=start_fetch,
                auto_adjust=False,
                progress=False,
            )
            if new_raw.empty:
                return current_data, 0

            if "Adj Close" in new_raw.columns:
                new_adj = new_raw["Adj Close"].copy()
            elif "Close" in new_raw.columns:
                new_adj = new_raw["Close"].copy()
            else:
                return current_data, 0

            if isinstance(new_adj, pd.Series):
                new_adj = new_adj.to_frame(name=self.universe.all_tickers[0])

            new_adj.index = pd.to_datetime(new_adj.index).tz_localize(None)
            new_adj = new_adj.dropna(how="all")

            # Combine and deduplicate
            combined = pd.concat([current_data, new_adj]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.ffill().bfill()

            new_bars = len(combined) - len(current_data)
            if new_bars > 0:
                self.save_prices(combined)

            return combined, max(0, new_bars)

        except Exception as e:
            print(f"[Warning] Failed to update latest data online: {e}")
            return current_data, 0
