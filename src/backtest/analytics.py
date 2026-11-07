"""
Performance analytics and tear-sheet metrics for quantitative strategies.
"""

from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PerformanceAnalytics:
    """Computes comprehensive quantitative performance and risk metrics."""

    @staticmethod
    def calculate_metrics(
        returns: pd.Series,
        risk_free_rate: float = 0.03,
        periods_per_year: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Computes standard portfolio analytics on net returns.
        Auto-detects periods_per_year (756 for 3-session intraday, 252 for daily) if not provided.
        """
        clean_ret = returns.dropna()
        n_bars = len(clean_ret)
        if n_bars < 2:
            return {}

        if periods_per_year is None:
            is_intraday = isinstance(clean_ret.index, pd.DatetimeIndex) and any(clean_ret.index.hour != 0)
            periods_per_year = 756 if is_intraday else 252

        years = n_bars / float(periods_per_year)
        cum_ret = (1.0 + clean_ret).prod() - 1.0
        cagr = (1.0 + cum_ret) ** (1.0 / max(years, 0.01)) - 1.0

        ann_vol = clean_ret.std() * np.sqrt(periods_per_year)

        # Excess return for Sharpe
        rf_period = (1.0 + risk_free_rate) ** (1.0 / float(periods_per_year)) - 1.0
        excess_ret = clean_ret - rf_period
        sharpe = (excess_ret.mean() * np.sqrt(periods_per_year)) / (clean_ret.std() + 1e-8)

        # Sortino (downside deviation below 0)
        downside = clean_ret[clean_ret < 0]
        downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 0 else 1e-8
        sortino = (excess_ret.mean() * np.sqrt(periods_per_year)) / (downside_vol + 1e-8)

        # Drawdowns
        equity = (1.0 + clean_ret).cumprod()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = abs(drawdown.min())

        # Calmar Ratio
        calmar = cagr / max_dd if max_dd > 0 else 0.0

        # Win rate
        active_days = clean_ret[clean_ret != 0]
        win_rate = (active_days > 0).mean() if len(active_days) > 0 else 0.0

        # Profit factor
        gross_wins = active_days[active_days > 0].sum()
        gross_losses = abs(active_days[active_days < 0].sum())
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else np.nan

        return {
            "total_return_pct": float(cum_ret * 100.0),
            "cagr_pct": float(cagr * 100.0),
            "annualized_vol_pct": float(ann_vol * 100.0),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown_pct": float(max_dd * 100.0),
            "calmar_ratio": float(calmar),
            "win_rate_pct": float(win_rate * 100.0),
            "profit_factor": float(profit_factor) if not np.isnan(profit_factor) else 0.0,
        }

    @classmethod
    def generate_comparison_table(cls, backtest_df: pd.DataFrame) -> pd.DataFrame:
        """Compares Strategy, KO Buy & Hold, and XLP Sector Benchmark."""
        strat_metrics = cls.calculate_metrics(backtest_df["net_ret"])
        ko_metrics = cls.calculate_metrics(backtest_df["target_ret"])
        xlp_metrics = cls.calculate_metrics(backtest_df["benchmark_ret"])

        comparison = pd.DataFrame(
            {
                "ML Strategy (Net)": strat_metrics,
                "KO Buy & Hold": ko_metrics,
                "XLP Sector ETF": xlp_metrics,
            }
        )
        return comparison

    @staticmethod
    def plot_tear_sheet(backtest_df: pd.DataFrame, output_path: Optional[str] = None):
        """Generates visual performance tear sheet comparing equity curves and drawdowns."""
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1]})

        # 1. Equity Curves
        strat_norm = backtest_df["strategy_equity"] / backtest_df["strategy_equity"].iloc[0]
        ko_norm = backtest_df["target_bh_equity"] / backtest_df["target_bh_equity"].iloc[0]
        xlp_norm = backtest_df["benchmark_bh_equity"] / backtest_df["benchmark_bh_equity"].iloc[0]

        axes[0].plot(strat_norm.index, strat_norm, label="ML Strategy (Net of Costs)", color="#1f77b4", linewidth=2.0)
        axes[0].plot(ko_norm.index, ko_norm, label="KO Buy & Hold", color="#d62728", linestyle="--", alpha=0.8)
        axes[0].plot(xlp_norm.index, xlp_norm, label="XLP Sector ETF", color="#2ca02c", linestyle=":", alpha=0.8)
        axes[0].set_title("Coca-Cola ML Cross-Asset Alpha Strategy vs Benchmarks", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Normalized Growth ($1 Base)")
        axes[0].legend(loc="upper left")

        # 2. Drawdowns
        strat_peak = strat_norm.cummax()
        strat_dd = (strat_norm - strat_peak) / strat_peak * 100.0
        ko_peak = ko_norm.cummax()
        ko_dd = (ko_norm - ko_peak) / ko_peak * 100.0

        axes[1].fill_between(strat_dd.index, strat_dd, 0, color="#1f77b4", alpha=0.3, label="Strategy Drawdown")
        axes[1].plot(ko_dd.index, ko_dd, color="#d62728", linestyle="--", alpha=0.5, label="KO Drawdown")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].legend(loc="lower left")

        # 3. Position Allocation
        axes[2].plot(backtest_df.index, backtest_df["position"], color="#ff7f0e", linewidth=1.2, label="Position Weight")
        axes[2].set_ylabel("Exposure")
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].set_xlabel("Date")
        axes[2].legend(loc="upper left")

        plt.tight_layout()

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=150)
            plt.close()
        else:
            plt.show()
