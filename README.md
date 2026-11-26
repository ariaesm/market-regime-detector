# market-regime-detector

A quantitative Machine Learning pipeline to detect, classify, and trade financial market regimes (Bull, Bear Panic, and Sideways Neutral) using **Gaussian Mixture Models (GMM)** and **Random Forest Classifiers**.

Developed by **[Aria Esmaeilzadeh](https://github.com/ariaesm)**.

---

## Overview

Traditional algorithmic trading models often fail when market dynamics shift unexpectedly from trending to volatile chop. **`market-regime-detector`** treats market phase detection as a two-stage machine learning problem:

1. **Unsupervised Regime Discovery:** A Gaussian Mixture Model (GMM) fits latent multivariate distributions across volatility, momentum, and return features.
2. **Supervised Transition Forecasting:** A Random Forest classifier is trained on strictly chronological historical splits to forecast the next day's regime ($t+1$).
3. **Regime-Aware Asset Allocation:** Allocates 100% equity in Bull regimes, 50% in Sideways regimes, and 0% (Cash preservation) in High-Volatility Bear regimes.

---

## Out-of-Sample Backtest Results

Performance comparison over the unseen test horizon:

| Metric | Regime-Aware Strategy | Benchmark (Buy & Hold) | Advantage |
| :--- | :---: | :---: | :---: |
| **Cumulative Return** | **+18.74%** | -17.71% | **+36.45% Alpha** |
| **Annualized CAGR** | **+19.94%** | -18.65% | **+38.59%** |
| **Annualized Volatility**| **12.92%** | 30.19% | **-57.2% Risk** |
| **Sharpe Ratio** ($r_f=3\%$) | **1.24** | -0.63 | **Significantly Higher** |
| **Max Drawdown (MDD)** | **-8.15%** | -39.28% | **31.13% Protected** |

---

## Architecture & ML Methodology

```
Historical OHLCV Data
         │
         ▼
Feature Engineering (No Lookahead Leakage)
  ├── Log Returns (1d, 5d)
  ├── Realized Volatility (Annualized)
  ├── Parkinson High-Low Volatility Estimator
  ├── Trend Strength (SMA20 / SMA50 Spread)
  ├── Normalized Average True Range (NATR)
  └── Volume Z-Score & RSI
         │
         ▼
Stage 1: Latent Regime Discovery (GMM)
  ├── Full covariance Gaussian Mixture Model
  ├── BIC / AIC model selection
  └── Transition probability matrix estimation
         │
         ▼
Stage 2: Supervised Transition Forecasting
  ├── Strict Chronological Split (75% Train / 25% Test)
  ├── Independent Preprocessing (Scaler fitted ONLY on Train)
  └── Random Forest Classifier (Macro F1: 0.73+)
         │
         ▼
Dynamic Asset Allocation & Out-of-Sample Backtest
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/ariaesm/market-regime-detector.git
cd market-regime-detector
pip install -e .[test]
```

### CLI Execution

Run the complete pipeline end-to-end:

```bash
market-regime --days 1200 --seed 42
```

---

## Python API Usage

```python
from market_regime import (
    generate_synthetic_market_data,
    extract_market_features,
    RegimeClusterer,
    RegimeForecaster,
    backtest_regime_strategy,
)

# 1. Ingest market data
df = generate_synthetic_market_data(n_days=1000)

# 2. Extract features
features = extract_market_features(df)

# 3. Discover regimes
clusterer = RegimeClusterer(n_regimes=3)
regimes, probs, stats = clusterer.fit_predict(features)

# 4. Train forecaster
forecaster = RegimeForecaster()
metrics, y_test, y_pred = forecaster.train_and_evaluate(features, regimes)

# 5. Backtest
test_returns = features["log_return"].iloc[-len(y_pred):].values
results = backtest_regime_strategy(test_returns, y_pred)
print(f"Alpha generated: +{results['alpha_pct']}%")
```

---

## Testing

```bash
pytest -v
```

All unit tests run in < 2s with 100% test pass rate.

---

## License

MIT © [Aria Esmaeilzadeh](https://github.com/ariaesm)
