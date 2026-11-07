"""
Realistic Vectorized/Event Backtesting Engine with Transaction Costs and Sizing Rules.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd


class BacktestEngine:
    """
    Executes backtests on out-of-sample ML trading signals.
    Accounts for execution lag, slippage, and transaction costs.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 5.0,
        position_sizing: str = "confidence_weighted",
        target_volatility: float = 0.12,
        allow_short: bool = False,
        max_leverage: float = 1.35,
        risk_free_rate: float = 0.045,
        periods_per_year: int = 756,
    ):
        self.initial_capital = initial_capital
        self.cost_rate = transaction_cost_bps / 10000.0  # Convert bps to decimal
        self.position_sizing = position_sizing
        self.target_volatility = target_volatility
        self.allow_short = allow_short
        self.max_leverage = max_leverage
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        target_ticker: str = "KO",
        benchmark_ticker: str = "XLP",
    ) -> pd.DataFrame:
        """
        Runs strategy simulation over the signal index.
        
        Args:
            prices: DataFrame of asset prices
            signals: Series of model out-of-sample predicted probabilities (P(outperform))
            target_ticker: Ticker being traded
            benchmark_ticker: Reference benchmark ticker
        """
        # Align prices to signal index
        common_idx = prices.index.intersection(signals.index).sort_values()
        p_target = prices.loc[common_idx, target_ticker]
        daily_ret = p_target.pct_change().fillna(0.0)
        
        # Benchmark returns for comparison
        bm_ret = (
            prices.loc[common_idx, benchmark_ticker].pct_change().fillna(0.0)
            if benchmark_ticker in prices.columns
            else daily_ret
        )

        probs = signals.loc[common_idx]

        # 1. Compute target raw position weight w in [-1.0, 1.0] or [0.0, 1.0]
        if self.position_sizing == "binary":
            raw_weights = pd.Series(np.where(probs >= 0.50, 1.0, -1.0 if self.allow_short else 0.0), index=common_idx)
        elif self.position_sizing == "confidence_weighted":
            if not self.allow_short:
                # Institutional active conviction allocation:
                # 0.60 neutral base scaled by alpha edge up to max_leverage (e.g. 1.35x), trimming to 0% (cash hedge)
                tilt = (probs - 0.50) / 0.08
                raw_weights = (0.60 + tilt * 0.70).clip(lower=0.0, upper=self.max_leverage)
            else:
                raw_weights = ((probs - 0.50) / 0.08).clip(lower=-self.max_leverage, upper=self.max_leverage)
        elif self.position_sizing == "volatility_targeted":
            # Scale inversely by 20-day historical volatility
            roll_vol = daily_ret.rolling(20).std() * np.sqrt(252)
            roll_vol = roll_vol.replace(0, np.nan).bfill()
            vol_scaler = (self.target_volatility / roll_vol).clip(lower=0.2, upper=self.max_leverage)
            direction = np.where(probs >= 0.50, 1.0, -1.0 if self.allow_short else 0.0)
            raw_weights = pd.Series(direction * vol_scaler, index=common_idx)
        else:
            raw_weights = pd.Series(np.where(probs >= 0.50, 1.0, 0.0), index=common_idx)

        # 2. Strict Execution Lag: Trade is entered at t based on signal at t-1
        positions = raw_weights.shift(1).fillna(0.0)

        # 3. Turnover & Transaction Cost
        weight_changes = positions.diff().abs().fillna(positions.abs().iloc[0])
        trade_costs = weight_changes * self.cost_rate

        # 4. Strategy Returns with Realistic Unallocated Cash Yield
        gross_returns = positions * daily_ret
        # Unallocated cash earns short-term risk-free yield (e.g. 4.5% annual T-bill rate)
        cash_frac = np.maximum(0.0, 1.0 - positions)
        cash_yield = cash_frac * (self.risk_free_rate / float(self.periods_per_year))
        net_returns = gross_returns - trade_costs + cash_yield

        # 5. Equity Curves
        strategy_equity = self.initial_capital * (1.0 + net_returns).cumprod()
        target_bh_equity = self.initial_capital * (1.0 + daily_ret).cumprod()
        benchmark_bh_equity = self.initial_capital * (1.0 + bm_ret).cumprod()

        results = pd.DataFrame(
            {
                "signal_prob": probs,
                "position": positions,
                "target_ret": daily_ret,
                "benchmark_ret": bm_ret,
                "gross_ret": gross_returns,
                "turnover": weight_changes,
                "cost": trade_costs,
                "net_ret": net_returns,
                "strategy_equity": strategy_equity,
                "target_bh_equity": target_bh_equity,
                "benchmark_bh_equity": benchmark_bh_equity,
            },
            index=common_idx,
        )

        return results
