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
- The C++ implementation scored Test R² 0.383, close to sklearn's 0.381 — validates the implementation is correct.
- **Same conclusion as Random Forest:** Train R² of 0.788 vs Test R² of 0.383 is notable overfitting. Linear regression (0.496) still wins on this dataset.
- The consistent finding across all tree-based models is that linear regression with well-engineered features generalises better on this dataset.

### Backtesting
- Converted linear regression predictions into a long/short trading strategy using `np.sign()` — +1 (buy) when predicting positive returns, -1 (short) when predicting negative.
- Strategy returns = signal × actual return each day. Correct shorts flip a loss into a gain by multiplying −1 × negative return.
- Cumulative portfolio value tracked using `np.cumprod(1 + strategy_returns)` and compared against a buy-and-hold benchmark.
- Expanded dataset from 1 year to 5 years — test set grew from ~45 days to 201 days, making all metrics statistically meaningful.

### Strategy Metrics
- **Beta = 0.08** against the S&P 500 — the strategy moves almost independently of the market, meaning returns come almost entirely from stock-specific signals rather than passive market exposure.
- **Daily Sharpe Ratio = 0.59** — in the "good" range (0.5–1.0). Return per unit of risk is solid for a simple daily strategy.
- **Annualized Sharpe = 9.35** — mathematically derived from the daily Sharpe (`0.59 × √252`), not a small-sample artefact. With 201 test days the sample size is statistically meaningful.
- **Max Drawdown = -3.50%** — the worst peak-to-trough drop over the entire test period. Very low for a daily trading strategy.
- **Win Rate > 50%** throughout the test period — the model gets the direction right more than half the time consistently, not just on a few lucky days.
- **Daily Sortino = 1.25, Annual = 19.84** — higher than Sharpe because Sortino only penalises losing days. The gap between Sortino and Sharpe indicates the strategy's volatility is mostly on the upside, which is desirable.
- **Transaction costs (5 bps/trade):** gross return ~246%, net return ~228% — an 18 percentage point drag from trading fees over 201 days.

---

## Results

| Model | MSE | Test R² | Train R² |
|-------|-----|---------|---------|
| Linear Regression | 0.000145 | 0.3092 | 0.5256 |
| RF from scratch (grid search) | 0.000199 | 0.0575 | 0.6082 |
| sklearn RandomForest | 0.000192 | 0.0896 | 0.6824 |
| C++ XGBoost | 0.000181 | 0.1400 | 0.9361 |
| sklearn XGBoost | 0.000179 | 0.1519 | 0.9379 |

### Backtesting (Linear Regression Strategy, 5-year dataset)

| Metric | Value |
|--------|-------|
| Test period | 2025-07-28 → 2026-05-13 (201 days) |
| Beta vs S&P 500 | 0.08 |
| Daily Sharpe | 0.59 |
| Annualized Sharpe | 9.35 |
| Daily Sortino | 1.25 |
| Annualized Sortino | 19.84 |
| Max Drawdown | -3.50% |
| Gross Return | ~246% |
| Net Return (5 bps/trade) | ~228% |

---

## Notebook Structure

The project is split into three notebooks that must be run in order. Each notebook saves its output to `data/` so the next one can load it without re-running everything.

| Notebook | Description | Saves |
|----------|-------------|-------|
| `notebooks/01_eda.ipynb` | Data loading, SMA/EMA, feature engineering, visualizations | `data/features.csv` |
| `notebooks/02_models.ipynb` | Train/test split, linear regression, random forest, XGBoost | `data/predictions.csv` |
| `notebooks/03_backtesting.ipynb` | Long/short strategy, equity curve, beta, Sharpe ratio | — |

---

## Environment Setup

### Requirements
- Python 3.10+
- Jupyter Notebook or JupyterLab
- g++ (for compiling the C++ XGBoost extension)
- pybind11 (`pip install pybind11`)

### Install dependencies

```bash
pip install -r requirement.txt
```

### Build the C++ XGBoost extension (macOS)

```bash
cd libraries
PYBIND=$(python -c "import pybind11; print(pybind11.get_include())")
PYINC=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
g++ -O3 -march=native -std=c++17 -shared -fPIC -undefined dynamic_lookup \
    -DXGB_EXTENSION -I$PYBIND -I$PYINC \
    xgboost_scratch.cpp -o ../notebooks/xgboost_cpp$SUFFIX
cd ..
```

### Run the notebooks in order

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_models.ipynb
jupyter notebook notebooks/03_backtesting.ipynb
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
- [x] Max Drawdown — rolling drawdown from peak, worst single trough
- [x] Win Rate — overall and rolling 20-day win rate vs 50% baseline
- [x] Sortino Ratio — daily (1.25) and annualized (19.84), only penalises downside volatility
- [x] Transaction Cost Simulation — 5 bps/trade, gross vs net equity curve comparison

### Notebook Refactor
- [x] Split `main.ipynb` into three focused notebooks (`01_eda`, `02_models`, `03_backtesting`)
- [x] Added CSV handoff between notebooks via `data/features.csv` and `data/predictions.csv`

### Next Steps
- [x] Expand dataset from 1 year to 5 years — test set grew to 201 days, all metrics now statistically meaningful
- [x] Re-evaluate all models on the larger dataset
- [ ] Rolling Sharpe — 30-day rolling window to check if edge is consistent over time
- [ ] Alpha and Information Ratio — returns above what market exposure explains
- [ ] Regime Analysis — bull vs bear market performance, quarterly returns
- [ ] Multi-ticker testing — run the same model and strategy on other stocks (MSFT, GOOGL, SPY) to test if the signal generalises beyond AAPL
- [ ] LSTM — implement a Long Short-Term Memory neural network from scratch; LSTMs are designed for sequential data and can capture temporal dependencies across many timesteps that tree-based models and linear regression cannot
