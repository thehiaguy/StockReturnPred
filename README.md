# Stock Return Predictor

Pulling historical price data with yfinance, creating features like moving averages, lagged returns and rolling volatility, then training a linear regression, random forest, and XGBoost model to predict next day returns.

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
- Added a bias column of ones to X so the model learns an intercept alongside the feature weights.
- Evaluated using MSE (Mean Squared Error) and R-squared.

### Feature Engineering
- Raw price is non-stationary (trends upward over time) which hurts model performance — returns fix this.
- Adding volatility (rolling std), momentum (rolling mean), and SMA ratio gives the model more signal beyond just lagged returns.
- More features does not always mean a better model — inspecting feature weights revealed which features were near zero and causing overfitting. SMA Ratio and 10/15-day windows were dropped.

### Random Forest (from scratch + sklearn)
- Built from scratch using three classes: `Node`, `DecisionTree`, and `RandomForest`.
- Each tree is trained on a bootstrap sample (random rows with replacement) and considers only `int(sqrt(n_features))` features per split — this keeps trees decorrelated.
- Hyperparameters (`max_depth`, `min_samples_split`) were tuned using grid search on a held-out validation set carved from the training data, not the test set.
- sklearn's `RandomForestRegressor` outperformed the from-scratch version (Test R² 0.09 vs 0.06) due to more optimised internal splitting logic.
- **Linear regression still beat all RF variants** on this dataset — with only ~220 rows, there is not enough data for trees to find reliable non-linear patterns that generalise.

### XGBoost (C++ from scratch + sklearn)
- Implemented in C++ using the XGBoost structure score gain criterion, compiled as a Python extension via pybind11.
- Builds trees sequentially — each tree corrects the residual errors of the ensemble so far, unlike Random Forest which builds trees in parallel.
- The C++ implementation scored Test R² 0.140, close to sklearn's 0.152 — validates the implementation is correct.
- **Same conclusion as Random Forest:** Train R² of 0.936 vs Test R² of 0.152 is severe overfitting. Linear regression (0.309) still wins on this small dataset.
- The consistent finding across all tree-based models is that ~220 rows is simply too small for complex models to generalise.

### Backtesting
- Converted linear regression predictions into a long/short trading strategy using `np.sign()` — +1 (buy) when predicting positive returns, -1 (short) when predicting negative.
- Strategy returns = signal × actual return each day. Correct shorts flip a loss into a gain by multiplying −1 × negative return.
- Cumulative portfolio value tracked using `np.cumprod(1 + strategy_returns)` and compared against a buy-and-hold benchmark.
- **Result:** strategy returned ~40% over the test period vs ~15% for buy-and-hold AAPL.

### Strategy Metrics
- **Beta = 0.38** against the S&P 500 — the strategy moves at only 38% of market swings, meaning most returns come from stock-specific signals (momentum, volatility) rather than just riding the market. Low beta combined with high returns implies genuine alpha.
- **Daily Sharpe Ratio = 0.58** — in the "good" range (0.5–1.0). Return per unit of risk is solid for a simple daily strategy.
- **Annualized Sharpe = 9.2** — artificially inflated by the tiny ~45 day test set. With only 45 observations, a good run with low volatility produces extreme annualized numbers. The daily Sharpe is the more honest figure until the dataset is expanded.

---

## Results

| Model | MSE | Test R² | Train R² |
|-------|-----|---------|---------|
| Linear Regression | 0.000145 | 0.3092 | 0.5256 |
| RF from scratch (grid search) | 0.000199 | 0.0575 | 0.6082 |
| sklearn RandomForest | 0.000192 | 0.0896 | 0.6824 |
| C++ XGBoost | 0.000181 | 0.1400 | 0.9361 |
| sklearn XGBoost | 0.000179 | 0.1519 | 0.9379 |

### Backtesting (Linear Regression Strategy, 1-year dataset)

| Metric | Value |
|--------|-------|
| Strategy return | ~40% |
| Buy-and-hold return | ~15% |
| Beta vs S&P 500 | 0.38 |
| Daily Sharpe Ratio | 0.58 |
| Annualized Sharpe | 9.2 (inflated — only 45 test days) |

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

### Feature Engineering
- [x] Lagged returns (Lag1, Lag2, Lag3)
- [x] Volatility — rolling standard deviation of returns over 5 and 30 days
- [x] Momentum — rolling mean of returns over 5 and 30 days
- [x] SMA ratio — closing price divided by the 30-day SMA (dropped — near-zero weight)

### Models
- [x] Linear Regression — Normal Equation, from scratch (baseline)
- [x] Random Forest — from scratch + sklearn comparison + grid search hyperparameter tuning
- [x] XGBoost — C++ from scratch (pybind11 extension) + sklearn comparison

### Evaluation
- [x] Plot predicted vs actual returns for all models
- [x] Backtest a long/short strategy against buy-and-hold benchmark
- [x] Beta against S&P 500
- [x] Sharpe Ratio (daily and annualized)

### Next Steps
- [ ] Expand dataset from 1 year to 5 years (`period='5y'`) — increases training rows from ~175 to ~1,000 and test days from ~45 to ~250, making all metrics statistically meaningful
- [ ] Re-evaluate all models on the larger dataset — tree-based models may close the gap with linear regression given more data
- [ ] Re-run backtesting metrics on the larger test set for a reliable annualized Sharpe
