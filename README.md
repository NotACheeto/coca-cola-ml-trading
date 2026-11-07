# Coca-Cola ($KO) Cross-Asset Machine Learning Trading Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch & LightGBM](https://img.shields.io/badge/ML-PyTorch%20%7C%20LightGBM%20%7C%20Ensemble-red.svg)](https://pytorch.org/)
[![Architecture: Deep Tabular ResNet (1400+ Neurons)](https://img.shields.io/badge/Architecture-Deep%20Tabular%20ResNet%20%281400%2B%20Neurons%29-purple.svg)]()
[![Backtest: Purged Walk--Forward & CPCV](https://img.shields.io/badge/Validation-Purged%20Walk--Forward%20%26%20CPCV-orange.svg)]()

A quantitative research and systematic trading framework designed to trade **Coca-Cola ($KO)** by leveraging cross-sectional signals, statistical arbitrage, and lead-lag dynamics across competitors, upstream packaging suppliers, bottlers, and sector proxies.

---

## 📌 Investment Thesis & Cross-Asset Dynamics

Coca-Cola does not operate in a vacuum. Its stock price reflects systemic beverage sector flows, input commodity costs, downstream bottling margins, and competitive market share battles. Standard single-asset time series models often suffer from low signal-to-noise ratios. By modeling **cross-asset relationships**, we extract structural alpha:

```
                      ┌────────────────────────────────────────┐
                      │    Macro & Sector Anchors              │
                      │  • XLP (Staples ETF)                   │
                      │  • SPY (Equity Risk Premium)           │
                      │  • UUP (USD Index - FX Headwinds)      │
                      └──────────────────┬─────────────────────┘
                                         │
        ┌────────────────────────────────┼──────────────────────────────┐
        ▼                                ▼                              ▼
┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│  Direct Competitors  │      │     Target Asset     │      │ Bottlers & Downstream│
│ • PEP  (PepsiCo)     │◄────►│   Coca-Cola ($KO)    │◄────►│ • COKE (Consolidated)│
│ • KDP  (Keurig Dr P) │      │                      │      │ • KOF  (FEMSA LatAm) │
│ • MNST (Monster Bev) │      └──────────▲───────────┘      └──────────────────────┘
│ • CELH (Celsius)     │                 │
└──────────────────────┘                 │
                                         ▼
                              ┌──────────────────────┐
                              │  Upstream Packaging  │
                              │ • BALL (Ball Corp)   │
                              │ • CCK  (Crown Hldgs) │
                              │ (Aluminum can costs) │
                              └──────────────────────┘
```

1. **Direct Peer Divergence ($PEP, $KDP, $MNST, $CELH):** Mean-reverting spreads and cointegration equilibrium. When KO diverges significantly from PepsiCo without idiosyncratic news, statistical arbitrage forces pull them back.
2. **Upstream Packaging Suppliers ($BALL, $CCK):** Beverage canning represents a major direct cost. Commodity and supplier cost pressures lead gross margins by several weeks.
3. **Downstream Bottling Network ($COKE, $KOF):** Bottlers experience volume demand shifts and freight pricing before corporate reports quarterly earnings.
4. **Sector Benchmark ($XLP) & FX Exposure ($UUP):** Over 60% of Coca-Cola’s revenues originate outside North America, making relative dollar strength a crucial predictive input.

---

## 💡 Strategic Advice to Improve the Trading Strategy

When developing a machine learning trading strategy on related equities, several domain-specific pitfalls must be addressed:

### 1. Avoid Spurious Price Regressions: Stationarity vs. Memory
* **The Pitfall:** Regressing raw stock prices produces $R^2 > 0.95$ but is statistically spurious because stock prices are integrated of order 1 ($I(1)$).
* **Solution:** 
  - Standardize returns ($I(0)$) for directional classification.
  - Compute **cointegration spreads** ($S_t = \ln(KO_t) - \beta_t \ln(PEP_t)$) via rolling Engle-Granger or Johansen tests. The spread residual is stationary ($I(0)$) while preserving price-level memory.
  - Employ **Fractional Differentiation** ($d \in [0.3, 0.6]$) to preserve maximum memory while passing the Augmented Dickey-Fuller (ADF) stationarity test.

### 2. Guard Against Data Leakage (Purged & Embargoed Cross-Validation)
* **The Pitfall:** Standard K-Fold Cross-Validation leaks future information into past folds due to serial correlation and overlapping forward-looking labels.
* **Solution:**
  - This repository uses **Purged Walk-Forward Cross-Validation** (Marcos López de Prado).
  - **Purging:** Drops training periods whose forward prediction horizon overlaps test periods.
  - **Embargoing:** Adds a cooldown period following test sets to eliminate autoregressive memory leakage.

### 3. Realistic Trade Execution & Friction Modeling
* **The Pitfall:** High-turnover statistical arbitrage strategies look phenomenal on paper but lose money in live execution due to commissions, bid-ask spread, and market impact.
* **Solution:**
  - Strict **$t-1$ execution lag**: Trades are executed at $t$ using features generated strictly before market open or at $t-1$ close.
  - Built-in transaction cost deduction (5–10 bps per turn).
  - Position sizing modulated by prediction confidence or inverse-volatility targeting to reduce unnecessary whipsaws.

### 4. Asymmetric Event Information Flow
* **The Pitfall:** Assuming prices always move synchronously at daily frequencies.
* **Solution:**
  - Competitors report earnings on different dates (e.g. PepsiCo frequently reports earnings several days before Coca-Cola). PEP's volume, pricing power commentary, and organic volume trends provide an informational lead.
  - Model abnormal trading volume in peers as an early-warning signal for sector rebalancing.

---

## 🏗️ Repository Architecture

```
coca-cola-ml-trading/
├── config/
│   └── default_config.yaml     # Strategy hyperparameters, universe, and risk settings
├── src/
│   ├── data/
│   │   ├── universe.py         # Multi-asset universe definitions (competitors, commodities, rates, FX)
│   │   └── loader.py           # Historical data fetcher, parquet cache, and aligner
│   ├── features/
│   │   ├── engineer.py         # 110+ Cross-asset features (input costs, spreads, RSI, PCA)
│   │   └── labeling.py         # Forward relative alpha & Triple Barrier labeling
│   ├── models/
│   │   ├── cross_val.py        # Purged & Embargoed Walk-Forward CV & Combinatorial Purged CV (CPCV)
│   │   ├── neural_net.py       # DeepQuantNet Tabular ResNet (1400+ neurons) & Ensemble Classifier
│   │   ├── trainer.py          # Multi-architecture production trainer & walk-forward CV
│   │   └── evaluator.py        # Financial ML metrics (AUC, Brier score, Spearman IC)
│   ├── backtest/
│   │   ├── engine.py           # Vectorized backtester with costs and execution lag
│   │   └── analytics.py        # Sharpe, Sortino, Drawdowns, and visual tear sheet generator
│   └── pipeline.py             # End-to-end research and backtesting pipeline
├── tests/
│   ├── test_features.py        # Feature stability and shape verification
│   ├── test_backtest.py        # Execution lag and accounting unit tests
│   ├── test_model_pipeline.py  # DeepQuantNet & Ensemble fit, save, and predict tests
│   └── test_leakage.py         # 5 rigorous zero-data-leakage verification tests
├── run_backtest.py             # Executable CLI entrypoint
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Build system and package configuration
└── README.md                   # System documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install dependencies using `uv` (recommended) or `pip`:

```bash
git clone https://github.com/NotACheeto/coca-cola-ml-trading.git
cd coca-cola-ml-trading

# Using uv
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or using standard pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Unit Tests
Verify that all feature transformations, leak-free splitters, and backtest math are passing:

```bash
pytest tests/
```

### 3. Step-by-Step Training & Data Feeding Workflow

#### Step A: Train Model Strictly on Data $\le$ 2023
Train the model on historical market data up to December 31, 2023, while keeping 2024+ completely unseen:

```bash
python train_model.py --cutoff-date 2023-12-31
```
* **Learns:** Optimal decision tree splits and feature weights from 2019–2023.
* **Outputs:** Serialized model bundle saved to `models/ko_model_up_to_2023.joblib`.

#### Step B: Test on Reserved 2024+ Out-of-Sample Data
Evaluate the 2023-trained model against the held-out 2024–present market data:

```bash
python test_out_of_sample.py --test-start 2024-01-01
```
* **Simulates:** Execution lag, 5 bps transaction costs, and confidence-weighted sizing on unseen data.
* **Outputs:** Comparison metrics vs. Coca-Cola and XLP benchmarks, plus a tear sheet in `results/out_of_sample_2024_2026.png`.

#### Step C: Feed New Data & Adapt Model Weights
As new market bars arrive (or from a custom CSV), feed the data to adapt the model weights:

```bash
# Option 1: Ingest latest daily market bars online and adapt weights
python feed_data.py --live --rolling-window 500

# Option 2: Feed a custom proprietary CSV file
python feed_data.py --csv path/to/my_prices.csv

# Option 3: Adjust weights using trailing 500 trading days
python feed_data.py --rolling-window 500
```
This script will:
1. Ingest the new data and compute updated cross-asset features.
2. Display a table showing **how model weights shifted** compared to the prior checkpoint.
3. Save the updated model checkpoint to `models/ko_model_updated.joblib`.
4. Output the **latest trade decision for tomorrow** (Probability, Action, Confidence, Position Size).

---

## ⚙️ Configuration

Key parameters can be adjusted in [`config/default_config.yaml`](file:///home/antigravity/coca-cola-ml-trading/config/default_config.yaml):

```yaml
target: "KO"

universe:
  direct_competitors: ["PEP", "KDP", "MNST", "CELH"]
  bottlers: ["COKE", "KOF"]
  suppliers: ["BALL", "CCK"]
  macro_and_sector: ["XLP", "SPY", "UUP"]

features:
  rolling_windows: [5, 10, 21, 63]
  lead_lag_lags: [1, 2, 3]
  cointegration_pairs:
    - ["KO", "PEP"]
    - ["KO", "XLP"]
  pca:
    enabled: true
    n_components: 3

labeling:
  forward_horizon: 5               # 5 trading days (~1 week)
  target_type: "relative_alpha"    # Outperformance relative to XLP
  neutral_threshold: 0.0015

model:
  type: "lightgbm"
  cv:
    n_splits: 5
    purge_window: 5
    embargo_pct: 0.01

backtest:
  initial_capital: 100000.0
  transaction_cost_bps: 5.0        # 5 bps per trade
  position_sizing: "confidence_weighted"
```

---

## 🔬 Advanced Enhancements & Roadmap

- [ ] **Earnings Calendar Lead-Lag Module:** Programmatically parse SEC 10-Q filing dates to trigger asymmetric positioning when competitors report before KO.
- [ ] **Alternative Data Ingestion:** Integrate point-of-sale scanner data or Google Trends search volume for Coca-Cola vs. Pepsi.
- [ ] **Reinforcement Learning:** Train a PPO (Proximal Policy Optimization) agent directly on transaction-cost-penalized PnL.
- [ ] **Fractional Differentiation:** Incorporate automated fractional calculus ($d$) parameter optimization.

---

## 📜 License
MIT License. See `LICENSE` for details.
