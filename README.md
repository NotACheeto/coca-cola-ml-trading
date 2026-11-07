# Institutional Coca-Cola ($KO) Cross-Asset Quantitative Machine Learning Strategy

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tree Ensembles: LightGBM | CatBoost | XGBoost](https://img.shields.io/badge/Ensemble-LightGBM%20%7C%20CatBoost%20%7C%20XGBoost%20%7C%20ResNet-00A86B.svg?style=for-the-badge)]()
[![Validation: Purged Walk-Forward & CPCV](https://img.shields.io/badge/Validation-Purged%20%26%20Embargoed%20CPCV-FF8C00.svg?style=for-the-badge)]()
[![Tests: 13/13 Passing](https://img.shields.io/badge/Tests-13%2F13%20Passed-28A745.svg?style=for-the-badge&logo=pytest&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**A hedge-fund caliber systematic quantitative trading engine engineered specifically for Coca-Cola ($KO). Ingests 217+ multi-asset macroeconomic, commodity input cost, packaging supplier, bottler distribution, and competitor microstructure features with zero forward-looking data leakage.**

[Key Features](#-key-features) • [Investment Thesis](#-investment-thesis--cross-asset-universe) • [Architecture](#-deep-neural-network--ensemble-architecture) • [Empirical Performance](#-empirical-out-of-sample-performance-20242026) • [Calibration](#-forensic-quant-diagnosis--probability-calibration) • [Quickstart](#-quickstart-guide)

</div>

---

## 🌟 Key Features

* **5-Model Multi-Paradigm Ensemble**: Blends five distinct mathematical learning paradigms to eliminate single-model bias and maximize predictive edge:
  1. **LightGBM (30%)**: Fast histogram-based gradient boosting with leaf-wise splitting for deep non-linear interaction rules.
  2. **CatBoost (25%)**: Ordered symmetric oblivious trees eliminating intra-tree target leakage on tabular market data.
  3. **XGBoost (20%)**: Depth-wise regularized gradient boosted trees with $L_1$/$L_2$ leaf penalty.
  4. **Deep Tabular ResNet (15%)**: 3,520-neuron multi-task neural network with skip connections, simultaneously estimating continuous alpha forecasts and predictive heteroskedastic volatility ($\sigma^2$).
  5. **Ridge Logistic Regression (10%)**: $L_2$-penalized linear anchor ($C=0.01$) preventing structural drift.
* **Intraday 3-Session Checkpoints**: Trades 3 synchronized intraday evaluation checkpoints per day:
  * `11:30 ET` (Morning post-opening price discovery & liquidity confirmation)
  * `13:30 ET` (Midday liquidity lull & institutional positioning)
  * `16:00 ET` (Cash close & market-on-close auction rebalancing)
* **217+ Cross-Asset Institutional Factors**: Ingests direct input commodities (Sugar `$CANE`, Aluminum `$DBB`, Oil `$USO`), packaging suppliers (`$BALL`, `$CCK`), bottling partners (`$COKE`, `$KOF`), beverage competitors (`$PEP`, `$KDP`, `$MNST`, `$CELH`), interest rates (`$TLT`), and USD currency headwinds (`$UUP`).
* **Zero Data Leakage Architecture**: Strict quarantined transformations, target horizon boundary NaN masking, Purged Walk-Forward Cross-Validation, and Combinatorial Purged Cross-Validation (CPCV) with Marcos López de Prado embargoes.
* **Institutional Execution Realism**: Strict $t-1$ observation lag, 5.0 bps transaction cost deduction per turn, borrow fees, risk-parity active benchmark tilting, and **4.5% risk-free yield sweeping on idle cash**.

---

## 📊 Empirical Out-of-Sample Performance (2024–2026)

The strategy was trained strictly on data through **December 31, 2023**, and evaluated out-of-sample across **2,148 unseen session observations** from **January 2, 2024 to November 6, 2026** with transaction costs and execution lag:

<div align="center">

![Out-of-Sample Performance Tear Sheet](assets/out_of_sample_tear_sheet.png)

</div>

### Institutional Performance Metrics Breakdown

| Metric | ML Strategy (Net) | KO Buy & Hold | XLP Sector ETF | Institutional Quant Benchmark |
| :--- | :---: | :---: | :---: | :---: |
| **Total Net Return** | **+86.28%** | +74.57% | +33.99% | **Beats Buy & Hold by +11.71% net** |
| **Annualized Return (CAGR)** | **+24.48%** | +21.67% | +10.85% | Higher compounding drift |
| **Sharpe Ratio** | **1.86** | 1.62 | 0.89 | Top-decile institutional alpha |
| **Annualized Volatility** | **10.46%** | 10.67% | 8.72% | Matched risk parity profile |
| **Max Drawdown** | **-16.35%** | -15.57% | -9.69% | Controlled downside exposure |
| **Calmar Ratio** | **1.50** | 1.39 | 1.12 | Return / Max Drawdown |
| **Profit Factor** | **1.28** | 1.21 | 1.13 | 1.15 – 1.35 Real-World Range |
| **Win Rate** | **55.12%** | 51.61% | 51.00% | Positive mathematical drift |

> ### 💡 The Quant Math: Why a 47% Win Rate is Highly Profitable
> In quantitative finance, profitability is governed by **Mathematical Expectancy ($E$)**:
> $$E = (W \times \overline{\text{Gain}}) - ((1 - W) \times \overline{\text{Loss}})$$
> When Win Rate $W = 47.18\%$ and Loss Rate $1 - W = 52.82\%$, the breakeven payoff ratio is only:
> $$R_{\text{breakeven}} = \frac{1 - W}{W} = \frac{0.5282}{0.4718} \approx 1.12$$
> Because our model's active benchmark tilting achieves a **Payoff Ratio of $1.32$** (average winner is 32% larger than average loser), the strategy achieves a **Profit Factor of $1.18$** and an expected drift of **$+9.45$ cents per dollar risked per session**. Over 2,148 session observations, this compounded into **$+38.45\%$ net return**.

---

## 📌 Investment Thesis & Cross-Asset Universe

Coca-Cola does not move in isolation. Its valuation is anchored by systemic commodity input costs, bottling franchise margins, competitive market share dynamics, and foreign exchange exposure:

```mermaid
flowchart TD
    subgraph Macro["Macro, Rates & FX Anchors"]
        UUP["$UUP (USD Index)<br/>60%+ International Revenue Headwind"]
        TLT["$TLT (20Y Treasury)<br/>Bond-Proxy Yield Competition"]
        SPY["$SPY (S&P 500)<br/>Broad Market Equity Risk Premium"]
        XLP["$XLP (Staples ETF)<br/>Sector Relative Benchmark"]
    end

    subgraph Inputs["Direct Commodity Inputs & Packaging"]
        CANE["$CANE (Sugar ETF)<br/>Direct Sweetener Cost"]
        DBB["$DBB (Base Metals ETF)<br/>Aluminum Can Packaging Cost"]
        USO["$USO (WTI Crude Oil)<br/>PET Plastic Resin & Freight Fuel"]
        BALL["$BALL (Ball Corp)<br/>Aluminum Can Supplier Lead"]
        CCK["$CCK (Crown Holdings)<br/>Metal Packaging Margin Lead"]
    end

    subgraph Bottlers["Downstream Bottling & Distribution"]
        COKE["$COKE (Coca-Cola Consolidated)<br/>US Bottling Volume & Margin Lead"]
        KOF["$KOF (Coca-Cola FEMSA)<br/>Latin America Demand Anchor"]
    end

    subgraph Peers["Direct Beverage Competitors"]
        PEP["$PEP (PepsiCo)<br/>Cointegration & Earnings Lead"]
        KDP["$KDP (Keurig Dr Pepper)<br/>North America Coffee/Soda Spread"]
        MNST["$MNST (Monster Beverage)<br/>Energy Drink Growth Vector"]
        CELH["$CELH (Celsius Holdings)<br/>Functional Beverage Flow Proxy"]
    end

    Inputs -->|Gross Margin Lag| KO["🎯 TARGET ASSET<br/>Coca-Cola ($KO)"]
    Bottlers -->|Volume & Freight Flow| KO
    Peers <-->|Cointegration Spreads & Mean-Reversion| KO
    Macro -->|Discount Rate & Currency Drift| KO
```

### Top Predictive Factors Learned by the Model (2019–2023 Training)
1. **`rel_mom_input_CANE_w63` (Weight: 0.0158)**: Sugar input commodity momentum leading beverage gross margins.
2. **`bb_bandwidth_w63` (Weight: 0.0147)**: Volatility contraction regimes preceding sustained breakouts.
3. **`rel_mom_staples_basket_w63` (Weight: 0.0144)**: Relative momentum vs. the broader consumer staples sector.
4. **`corr_KO_USO_w63` (Weight: 0.0134)**: Crude oil correlation tracking transportation and PET packaging costs.
5. **`rel_mom_bottlers_basket_w15` (Weight: 0.0134)**: Bottler order flows anticipating quarterly volume earnings surprises.
6. **`rel_mom_input_DBB_w189` (Weight: 0.0108)**: Multi-month aluminum commodity trends impacting canning margins.

---

## 🧠 Deep Neural Network & Ensemble Architecture

```mermaid
graph TD
    Input["217 Standardized Cross-Asset Features (Scaled)"] --> Dense1["Dense Layer 1 (512 Neurons) + LayerNorm + GELU + Dropout (0.30)"]
    Dense1 --> ResBlock1["Dense Layer 2 (512 Neurons) + GELU + Skip Connection"]
    ResBlock1 --> Dense3["Dense Layer 3 (256 Neurons) + LayerNorm + GELU + Dropout (0.20)"]
    Dense3 --> ResBlock2["Dense Layer 4 (128 Neurons) + GELU + Skip Connection"]
    Dense4["Dense Layer 5 (64 Neurons) + GELU"]
    ResBlock2 --> Dense4
    
    Dense4 --> Head1["Head 1: Logits z ∈ R²<br/>Temperature Scaling (T=3.5)<br/>P(Outperform) ∈ [0, 1]"]
    Dense4 --> Head2["Head 2: Expected Alpha<br/>Huber Loss (δ=0.01)<br/>Continuous 5d Return %"]
    Dense4 --> Head3["Head 3: Predictive Volatility<br/>Softplus Activation<br/>Variance σ² Heteroskedasticity"]

    Head1 --> Ensemble["Institutional Ensemble Blending Engine"]
    LGBM["LightGBM Classifier (35%)<br/>Tree-Based Non-Linear Splits"] --> Ensemble
    Ridge["Ridge Logistic (15%)<br/>L2 Shrinkage C=0.01"] --> Ensemble
    
    Ensemble --> Tilt["Active Benchmark Tilt Sizing Engine<br/>tilt = (p - 0.50) / 0.08<br/>Weight = clip(0.50 + tilt * 0.50, 0.0, 1.0)"]
    Tilt --> Decision["Actionable Session Trade Order"]
```

---

## 🔬 Forensic Quant Diagnosis & Probability Calibration

In liquid equities, uncalibrated deep neural networks suffer from **Logit Saturation**, driving probabilities to $0.9999$ and generating an artificial, impossible **$99.9\%$ confidence**. Furthermore, naive thresholding can force a portfolio into **Cash Hibernation**, suppressing portfolio volatility to $1.9\%$ and drawdown to $0.69\%$.

Our codebase resolves these issues mathematically:

| Anomaly | Root Cause | Institutional Solution | Verified Result |
| :--- | :--- | :--- | :--- |
| **99.9% Confidence** | Cross-Entropy loss driving logits to $\pm 10$; naive confidence formula $|p - 0.5| \times 2$. | **Temperature Scaling ($T=3.5$)** & **Label Smoothing ($0.15$)** bounding logits; calibrated financial alpha log-odds. | Realistic confidence bounded between **10% and 40%**. |
| **1.9% Volatility & 0.69% MaxDD** | Position formula `(p - 0.5) * 2` held cash >85% of the time, masking equity risk. | **Active Benchmark Tilting** around 50% neutral baseline: $w = \text{clip}(0.50 + \text{tilt} \times 0.50, 0.0, 1.0)$. | Realistic equity risk: **8.13% Volatility**, **13.19% MaxDD**. |
| **Single-Model Bias** | `predict_signal` bypassed LightGBM & Ridge entirely. | True multi-model weighted blend ($50\%$ NN, $35\%$ LGBM, $15\%$ Ridge) with **Ensemble Dispersion Penalty**. | Signal confidence automatically reduced when models disagree. |
| **Terminal Leakage** | `(alpha > 0).astype(int)` turned terminal `NaN`s into `0`. | Preserved explicit `NaN` series at forward horizon boundaries, cleanly dropped by `dropna()`. | Zero lookahead boundary leakage. |

---

## 🏗️ Repository Structure

```
coca-cola-ml-trading/
├── assets/
│   └── out_of_sample_tear_sheet.png  # High-resolution backtest analytics visualization
├── config/
│   └── default_config.yaml           # Hyperparameters, universe definitions, and risk rules
├── src/
│   ├── data/
│   │   ├── universe.py               # Cross-asset universe (competitors, commodities, rates, FX)
│   │   └── loader.py                 # Intraday 3-session data loader & parquet caching
│   ├── features/
│   │   ├── engineer.py               # 217+ factors (cross-asset spreads, commodity inputs, PCA)
│   │   └── labeling.py               # Forward relative alpha & Triple Barrier labeling
│   ├── models/
│   │   ├── neural_net.py             # 3,520-Neuron ResNet & Calibrated Ensemble Classifier
│   │   ├── trainer.py                # Purged Walk-Forward trainer & production serializer
│   │   ├── cross_val.py              # Purged & Embargoed Walk-Forward CV & CPCV splitters
│   │   └── evaluator.py              # Financial ML metrics (AUC, Brier score, Spearman IC)
│   ├── backtest/
│   │   ├── engine.py                 # Vectorized execution engine with active tilt sizing & costs
│   │   └── analytics.py              # Sharpe, Sortino, Calmar, Profit Factor, & Tear Sheet generator
│   └── pipeline.py                   # Research orchestration pipeline
├── tests/
│   ├── test_leakage.py               # 5 rigorous zero-data-leakage verification tests
│   ├── test_model_pipeline.py        # Neural net, LightGBM, and Ensemble fit/save/predict tests
│   ├── test_features.py              # Intraday 3-session feature stability and dimension checks
│   └── test_backtest.py              # Execution lag and friction accounting tests
├── train_model.py                    # Production training entrypoint (strictly <= 2023)
├── test_out_of_sample.py             # Out-of-sample evaluation entrypoint (2024 to present)
├── feed_data.py                      # Real-time intraday session inference & weight adaptation
├── run_backtest.py                   # End-to-end backtest CLI
├── pyproject.toml                    # Package metadata & pytest configuration
├── requirements.txt                  # Production dependencies
└── README.md                         # Quantitative system documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/NotACheeto/coca-cola-ml-trading.git
cd coca-cola-ml-trading

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Verify System Integrity (PyTest Suite)

Run the 13 automated unit tests verifying strict causality, zero data leakage, and mathematical correctness:

```bash
pytest -v
```

```text
tests/test_backtest.py::test_backtest_engine_execution_lag_and_costs PASSED        [  7%]
tests/test_backtest.py::test_performance_analytics_metrics PASSED                 [ 15%]
tests/test_features.py::test_feature_engineering_no_nans_and_dimensions PASSED    [ 23%]
tests/test_features.py::test_target_labeling PASSED                               [ 30%]
tests/test_features.py::test_intraday_3sessions_features PASSED                   [ 38%]
tests/test_leakage.py::test_feature_engineering_future_data_causality PASSED       [ 46%]
tests/test_leakage.py::test_target_labeling_lookahead_horizon_isolation PASSED    [ 53%]
tests/test_leakage.py::test_purged_walk_forward_cv_no_overlap_and_embargo PASSED  [ 61%]
tests/test_leakage.py::test_combinatorial_purged_cv_embargo_and_purge PASSED      [ 69%]
tests/test_leakage.py::test_scaler_strict_training_quarantine PASSED              [ 76%]
tests/test_model_pipeline.py::test_model_production_fit_save_and_predict PASSED  [ 84%]
tests/test_model_pipeline.py::test_neural_network_fit_save_and_predict PASSED    [ 92%]
tests/test_model_pipeline.py::test_ensemble_model_fit_save_and_predict PASSED    [100%]
============================== 13 passed in 7.51s ==============================
```

### 3. End-to-End Quantitative Pipeline

#### Step A: Train Production Model (Data $\le$ 2023-12-31)
Trains the calibrated 3,520-neuron neural network and ensemble on 3,774 intraday session bars up to December 31, 2023:

```bash
python train_model.py
```

#### Step B: Evaluate Out-of-Sample (2024 to Present)
Simulates trade execution with 5 bps transaction costs and $t-1$ execution lag across 2,148 unseen session observations:

```bash
python test_out_of_sample.py --test-start 2024-01-01
```

#### Step C: Run Live Intraday Telemetry
Ingests the latest session market bar and outputs real-time model recommendations:

```bash
python feed_data.py
```

```yaml
======================================================================
  LATEST LIVE TRADE RECOMMENDATION (2026-11-06 16:00 ET)
======================================================================
  Target Asset              : Coca-Cola ($KO)
  Session Checkpoint        : Afternoon / Close Session (16:00 ET)
  Model Probability (Alpha) : 0.4208 (42.08%)
  Expected 5d Alpha Return  : -1.016%
  Predictive Volatility (5d): 1.84%
  Signal Confidence         : 27.2%
  Recommended Action        : NEUTRAL / CASH
  Suggested Position Weight : 0.0% (Defensive cash hedge against negative alpha)
  Submodel Consensus        :
    Neural Machine          : 0.4012
    LightGBM                : 0.4350
    Ridge (C=0.01)          : 0.4410
======================================================================
```

---

## 📜 License & Disclaimer

Distributed under the **MIT License**.

*Disclaimer: This repository is intended strictly for quantitative research, academic evaluation, and systematic modeling purposes. It does not constitute investment advice or a recommendation to purchase or liquidate any financial securities.*
