# Stock Return Predictor

Pulling historical price data with yfinance, creating features like moving averages, lagged returns and rolling volatility, then training a random forest and XGBoost model to predict next day returns.

---

## What I Have Learned

### Moving Averages
- **Simple Moving Average (SMA)** — averages the closing price over a fixed window of n days with equal weight. The first n-1 values are NaN since there aren't enough prior observations.
- **Exponential Moving Average (EMA)** — similar to SMA but gives more weight to recent prices using a smoothing factor. Unlike SMA it has no NaN gap at the start since it computes recursively from the first row.
- EMA reacts faster to recent price changes, SMA is smoother and better for long-term trends.

### Lagged Returns
- A lag is a time-shifted copy of the returns column — Lag1 = yesterday's return, Lag2 = two days ago, etc.
- Lagged returns turn a time series prediction problem into a regular regression problem that linear regression can handle.
- Stock returns tend to show near-zero correlation between lags and today's return, consistent with the Efficient Market Hypothesis (EMH).

### Linear Regression (from scratch)
- Implemented using the Normal Equation instead of gradient descent since the dataset is small-to-medium sized.
- The Normal Equation solves for optimal weights in one shot: `θ = (XᵀX)⁻¹ Xᵀy`
- Added a bias column of ones to X so the model learns an intercept alongside the lag weights.
- Evaluated using MSE (Mean Squared Error) and R-squared.

### Feature Engineering
- Raw price is non-stationary (trends upward over time) which hurts model performance — returns fix this.
- Adding volatility (rolling std), momentum (rolling mean), and SMA ratio gives the model more signal beyond just lagged returns.

---

## Environment Setup

### Requirements
- Python 3.10+
- Jupyter Notebook or JupyterLab

### Install dependencies

```bash
pip install -r requirement.txt
```

### Run the notebook

```bash
jupyter notebook notebooks/main.ipynb
```

---

## What I Am Working On Next

### Feature Engineering (in progress)
- [x] Lagged returns (Lag1, Lag2, Lag3)
- [ ] Volatility — rolling standard deviation of returns over 5 days
- [ ] Momentum — rolling mean of returns over 5 days
- [ ] SMA ratio — closing price divided by the 30-day SMA

### Better Models
- [ ] Random Forest — captures non-linear patterns that linear regression misses
- [ ] XGBoost — handles complex feature interactions, typically stronger than random forest on tabular data

### Evaluation
- [ ] Compare MSE and R-squared across all three models
- [ ] Plot predicted vs actual returns for each model
- [ ] Backtest a long/short strategy against buy and hold
