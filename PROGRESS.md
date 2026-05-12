# Stock Return Predictor — Session Progress

This file summarises everything completed so far and what needs to be done next. Use it to get up to speed at the start of a new session.

---

## Project Overview

Predicting next-day returns for AAPL stock using machine learning. Data is pulled from Yahoo Finance via `yfinance`. The notebook is at `notebooks/main.ipynb`.

**Stack:** Python, yfinance, pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

**Install:** `pip install -r requirement.txt`

---

## Notebook Structure (in order)

1. Imports
2. yfinance data exploration (`data = yf.Ticker("AAPL")`)
3. SMA and EMA markdown + code
4. Lagged Returns markdown
5. Normal Equation markdown
6. Linear regression function (from scratch)
7. Feature engineering cell — lags, volatility, momentum all computed here
8. Sanity check display (dropna preview)
9. Volatility markdown + grid display + plot
10. Momentum markdown + grid display + plot
11. SMA Ratio markdown + calculation + plot
12. `dropna()` + build X and y (cell `eced3c42`)
13. Train/test split, fit model, predict
14. Reading Visualizations markdown
15. Returns over time plot
16. Lag correlation scatter plots
17. Correlation heatmap
18. MSE
19. R²
20. Feature weights (theta) printout
21. Train vs Test R² overfitting check
22. Actual vs Predicted returns plot
23. Conclusions markdown

---

## What Has Been Implemented

### Moving Averages
- SMA30 and EMA50 computed on closing price using `.rolling().mean()` and `.ewm()`
- `sma_30` stored as a separate variable (used later for SMA Ratio)

### Feature Engineering
All features are computed in a single cell before the `dropna()`:

| Feature | Method | Notes |
|---------|--------|-------|
| Lag1, Lag2, Lag3 | `.shift(n)` | Past 1/2/3 day returns |
| Vol_5day | `.rolling(5).std()` | Short term volatility |
| Vol_10day | `.rolling(10).std()` | Dropped from model |
| Vol_15day | `.rolling(15).std()` | Dropped from model |
| Vol_30day | `.rolling(30).std()` | Long term volatility |
| Momentum_5days | `.rolling(5).mean()` | Short term trend |
| Momentum_10days | `.rolling(10).mean()` | Dropped from model |
| Momentum_15days | `.rolling(15).mean()` | Dropped from model |
| Momentum_30days | `.rolling(30).mean()` | Long term trend |
| SMA_Ratio | `Close / sma_30` | Computed separately, dropped from model |

**Note:** `SMA_Ratio` is computed in its own cell using `sma_30` (not `history['SMA30']`) to avoid a KeyError after the dropna trims the DataFrame.

### Features Currently Used in Model
```python
feature_cols = ['Lag1', 'Lag2', 'Lag3', 'Vol_5day', 'Vol_30day', 'Momentum_5days', 'Momentum_30days']
```
SMA_Ratio and 10/15 day windows were dropped after inspecting theta weights — they were near zero and causing overfitting.

### Linear Regression (from scratch)
Implemented using the Normal Equation — no gradient descent needed for this dataset size:
```python
def linear_regression(X, y):
    x_b = np.c_[np.ones((len(X), 1)), X]
    theta = np.linalg.inv(x_b.T @ x_b) @ x_b.T @ y
    return theta
```

### Data Pipeline
```python
history = history[['Close', 'Return'] + feature_cols].dropna()
X = history[feature_cols].values
y = history['Return'].values

split = int(len(X) * 0.8)  # 80/20 chronological split
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
```

---

## Current Results (Linear Regression)

| Metric | Value |
|--------|-------|
| MSE | 0.000145 |
| Test R² | 0.3092 |
| Train R² | 0.5256 |

The gap between train and test R² (0.52 vs 0.31) indicates mild overfitting — expected given the small dataset (~220 rows after dropna). Linear regression has no built-in overfitting controls.

### Feature Weights (theta)
```
Lag1:            -0.418704   (mean reversion signal)
Lag2:            -0.498672   (mean reversion signal)
Lag3:            -0.582546   (mean reversion signal)
Vol_5day:         0.298715   (positive — higher vol predicts higher return)
Vol_30day:       -0.417566   (negative — high long term vol predicts lower return)
Momentum_5days:   2.420970   (strongest feature — short term trend continuation)
Momentum_30days:  0.432776   (positive — long term trend continuation)
SMA_Ratio:       -0.010483   (near zero — dropped)
```

---

## Key Decisions Made

- **Dropped SMA Ratio** from the model — weight was -0.01, contributing no signal
- **Dropped 10-day and 15-day** volatility/momentum windows — caused overfitting with minimal gain
- **Chronological train/test split** used (not random) since time series data must not be shuffled
- **Normal Equation** used instead of gradient descent — dataset is small enough that it is faster and exact
- **`dropna()` done once at the end** after all features are built, not in individual cells

---

## What Is Next

### 1. Random Forest
- Import from `sklearn.ensemble import RandomForestRegressor`
- Use the same `X_train`, `X_test`, `y_train`, `y_test`
- Key hyperparameters to tune: `n_estimators`, `max_depth`, `min_samples_split`
- Compare MSE and R² against linear regression baseline
- Check train vs test R² gap to assess overfitting

### 2. XGBoost
- Import from `xgboost import XGBRegressor`
- Same feature set and train/test split
- Key hyperparameters: `n_estimators`, `learning_rate`, `max_depth`
- Compare against both linear regression and random forest

### 3. Model Comparison
- Side by side table of MSE and R² for all three models
- Plot actual vs predicted for each model on the same chart

### 4. Backtesting (stretch goal)
- When model predicts positive return → go long (buy)
- When model predicts negative return → go short (sell)
- Compare cumulative strategy returns vs buy and hold
- Evaluate with a simple equity curve

---

## Notes for Next Session

- The `sma_30` variable must be created in the same cell as `history['SMA30']` (cell `73d4f6fd`) for SMA_Ratio to work downstream
- All feature engineering happens in cell `3e5846ca` before the `dropna()`
- The `8374d676` cell is just a display/sanity check — it does not save the result back to `history`
- Visualizations (vol, momentum, SMA ratio plots) come before the model cells
- Model cells start at `eced3c42`
