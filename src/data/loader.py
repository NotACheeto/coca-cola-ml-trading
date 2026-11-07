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
    ):
        self.universe = universe
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.start_date = start_date
        self.end_date = end_date

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetches adjusted close prices and volumes for all tickers in universe.
        Returns a multi-indexed DataFrame or aligned wide DataFrame of Adjusted Closes.
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
