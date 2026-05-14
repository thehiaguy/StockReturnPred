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
- **Rolling Sharpe** — 30-day rolling window stays consistently positive throughout the test period, confirming the edge is stable and not driven by a few lucky stretches.
- **Alpha = 0.0075, Information Ratio = 0.51** — the strategy earns 0.75% per day above what market exposure predicts. IR of 0.51 is in the "good" range, meaning the active risk taken is being rewarded.
- **Regime Analysis** — bear regime win rate (76.1%) and mean return (0.0092) are both higher than bull regime (69.8%, 0.0066), confirming the strategy profits from short signals in down markets and is not market-dependent.

### Multi-Ticker Testing
- Ran the same linear regression pipeline on AAPL, MSFT, GOOGL, and SPY to test if the signal generalises beyond a single stock.
- **Test R² between 0.45–0.54 across all four tickers** — consistent performance with almost no train/test gap, confirming the feature set captures a real and generalisable signal.
- GOOGL produced the highest cumulative return (7.7x) due to higher volatility — correct directional calls on a volatile stock produce larger gains.
- SPY produced the lowest cumulative return (1.4x) because it is a low-volatility index ETF — same win rate, smaller per-day gains.

### LSTM (in progress)
- A standard neural network treats every input independently with no memory of previous days — useless for time series where sequence matters.
- An RNN fixes this with a hidden state passed between time steps, but gradients vanish over long sequences so it can't learn patterns spanning more than ~10 steps.
- An LSTM fixes the vanishing gradient problem with two states: **`h`** (short-term hidden state) and **`c`** (long-term cell state that flows through time with only small controlled changes).
- The 4 gates control memory: **forget** (how much old memory to keep), **input** (how much new info to write), **cell/candidate** (what new info to write), **output** (what to expose as the hidden state).
- Sigmoid is used on gates as a continuous 0–1 valve (not a probability); tanh is used for content values since returns can be positive or negative.

---

## Results

| Model | MSE | Test R² | Train R² |
|-------|-----|---------|---------|
| Linear Regression | 0.000110 | 0.4958 | 0.4649 |
| RF from scratch | 0.000172 | 0.2133 | 0.4409 |
| RF grid search | 0.000168 | 0.2281 | 0.4012 |
| sklearn RandomForest | 0.000153 | 0.3001 | 0.5398 |
| C++ XGBoost | 0.000135 | 0.3831 | 0.7877 |
| sklearn XGBoost | 0.000135 | 0.3807 | 0.7799 |

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
| `notebooks/03_backtesting.ipynb` | Long/short strategy, equity curve, beta, Sharpe, Sortino, drawdown, win rate, alpha, regime analysis | — |
| `notebooks/04_multiticker.ipynb` | Linear regression pipeline across AAPL, MSFT, GOOGL, SPY — generalisation check | — |
| `notebooks/05_lstm.ipynb` | LSTM from scratch (numpy), then PyTorch with GPU acceleration | — |

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
- [x] LSTM — numpy from scratch: forward pass, BPTT, and training loop implemented in `05_lstm.ipynb`
- [ ] LSTM — PyTorch with GPU (RTX 5080)

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
- [x] Rolling Sharpe — 30-day rolling window, edge is consistent throughout test period
- [x] Alpha and Information Ratio — alpha 0.0075/day, IR 0.51
- [x] Regime Analysis — bull vs bear market performance, strategy profitable in both regimes
- [x] Multi-ticker testing — Test R² 0.45–0.54 across AAPL, MSFT, GOOGL, SPY; signal generalises
- [x] LSTM numpy from scratch — forward pass (4 gates), BPTT, and training loop implemented in `05_lstm.ipynb`
- [ ] Load features data and run training loop — verify loss decreases over epochs
- [ ] LSTM PyTorch with GPU — re-implement using PyTorch on RTX 5080, compare training speed and Test R² against numpy version and linear regression baseline
