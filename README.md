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
- sklearn's `RandomForestRegressor` outperformed the from-scratch version (Test R² 0.15 vs 0.09) due to more optimised internal splitting logic.
- **Linear regression still beat all RF variants** on this dataset — with only ~220 rows, there is not enough data for trees to find reliable non-linear patterns that generalise.

---

## Results So Far

| Model | MSE | Test R² | Train R² |
|-------|-----|---------|---------|
| Linear Regression | 0.000145 | 0.3092 | 0.5256 |
| RF from scratch (grid search) | 0.000192 | 0.0861 | 0.5915 |
| sklearn RandomForest | 0.000180 | 0.1468 | 0.6747 |

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
- [ ] XGBoost — gradient boosted trees, expected to handle small datasets better than RF

### Evaluation
- [ ] Compare MSE and R-squared across all three models in a final summary
- [ ] Plot predicted vs actual returns for each model
- [ ] Backtest a long/short strategy against buy and hold
