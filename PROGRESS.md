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
23. Linear Regression conclusions markdown
24. Random Forest from scratch markdown (explains Node, DecisionTree, RandomForest)
25. Random Forest from scratch implementation (cell `de08afea`)
26. Random Forest evaluation (cell `d877c97e`)
27. Hyperparameter Tuning markdown (explains grid search + validation set)
28. Grid search + best RF retrain + evaluation (cell `0951f941`)
29. sklearn RF comparison markdown
30. sklearn RandomForestRegressor evaluation (cell `08e99d08`)
31. Random Forest conclusions markdown
32. XGBoost from scratch markdown (boosting loop, split criterion, structure score)
33. C++ XGBoost evaluation (cell `f3e250f7`)
34. sklearn XGBRegressor comparison markdown
35. sklearn XGBRegressor evaluation (cell `a1062688`)
36. XGBoost conclusions markdown

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

### Random Forest (from scratch)
Implemented three classes: `Node`, `DecisionTree`, `RandomForest`.

Key implementation details:
- Each tree trained on a bootstrap sample: `idxs = np.random.choice(len(X), len(X), replace=True)`
- Each split considers `int(np.sqrt(X.shape[1]))` = 2 random features (keeps trees decorrelated)
- Split criterion: weighted MSE = `(len(left)*var(left) + len(right)*var(right)) / n`
- Leaf prediction: `np.mean(y)` of all samples that reached that leaf
- Prediction: average across all trees via `np.mean([tree.predict(X) for tree in self.trees], axis=0)`
- Hyperparameters tuned via grid search on a validation set (last 20% of training data)
- Best params found: `max_depth=5`, `min_samples_split=10`

### sklearn RandomForestRegressor
- Same hyperparameters as from-scratch for fair comparison
- `random_state=42` for reproducibility

### XGBoost (C++ from scratch)
Implemented in `libraries/xgboost_scratch.cpp`, compiled as a Python extension via pybind11 and imported as `XGBoostCpp`.

Key implementation details:
- **Boosting loop:** starts from `F_0 = mean(y)`, each round fits a tree to the first-order gradients `g_i = F(x_i) - y_i` and constant hessians `h_i = 1` (MSE loss)
- **Split criterion:** maximises XGBoost structure score gain: `0.5*(G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)) - γ`
- **Leaf value:** optimal weight `w* = -G/(H+λ)` derived from the second-order Taylor expansion
- **Column-major XGBMatrix:** feature columns stored contiguously — cache-friendly for the split-finding inner loop
- **Pre-sorted indices:** `sorted_idx[f]` built once before training; every node reuses it (O(n_total) scan vs O(n_node log n_node) re-sort per node)
- **OpenMP parallelism:** `#pragma omp parallel for` over the feature loop — features are independent so no synchronisation needed; each thread writes to its own `alignas(64)` SplitResult slot to prevent false sharing

Build system:
- `libraries/build.bat` — runs the full g++ command, copies `.pyd` to `notebooks/`
- Compile command: `C:\msys64\ucrt64\bin\g++ -O3 -march=native -fopenmp -std=c++17 -shared -DXGB_EXTENSION ...`
- The `-DXGB_EXTENSION` flag enables the pybind11 block and disables `main()`
- `os.add_dll_directory(r"C:\msys64\ucrt64\bin")` must be called before the import to resolve MSYS2 runtime DLLs

### sklearn XGBRegressor
- Same hyperparameters as C++ version for fair comparison (`n_estimators=100`, `learning_rate=0.1`, `max_depth=3`)
- `random_state=42` for reproducibility

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

## Current Results

| Model | MSE | Test R² | Train R² |
|-------|-----|---------|---------|
| Linear Regression | 0.000145 | 0.3092 | 0.5256 |
| RF from scratch (grid search) | 0.000199 | 0.0575 | 0.6082 |
| sklearn RandomForest | 0.000192 | 0.0896 | 0.6824 |
| C++ XGBoost | 0.000181 | 0.1400 | 0.9361 |
| sklearn XGBoost | 0.000179 | 0.1519 | 0.9379 |

---

## What Went Well This Session

- **C++ XGBoost implementation validated.** The from-scratch C++ version scored Test R² 0.140 vs sklearn's 0.152 — only an 8% gap, much closer than the RF from-scratch vs sklearn gap. This confirms the implementation is correct.
- **pybind11 extension compiled and working.** `xgboost_scratch.cpp` successfully compiled into a `.pyd` file and imported as `XGBoostCpp` in the notebook using `os.add_dll_directory` to resolve MSYS2 runtime DLLs.
- **build.bat created for one-command rebuilds.** All paths hardcoded so it works from any terminal without needing g++ on PATH or activating the venv.
- **Consistent finding across all models.** Linear regression (Test R² 0.309) outperforms all tree-based models on this dataset. This is a real and meaningful result — not a failure — confirming that dataset size is the binding constraint, not the model.
- **Visualizations added after every model.** Actual vs predicted plots now appear after each model's MSE/R² output for easy comparison.

## What Did Not Go Well This Session

- **Build process was painful on Windows.** Multiple issues encountered: PowerShell comma-parsing breaking `-Wl,--enable-auto-import`, `g++` not on PATH, `python` pointing to system Python instead of venv, `--- stderr ---` printed with no content (g++ errors swallowed by subprocess capture). Resolved by using `build.bat` via `cmd /c` and running g++ directly.
- **DLL load failed on first import.** The `.pyd` compiled successfully but Python couldn't find MSYS2 runtime DLLs (`libgomp`, `libgcc`). Fixed with `os.add_dll_directory(r"C:\msys64\ucrt64\bin")` before the import.
- **XGBoost also failed to beat linear regression.** Train R² of 0.936 vs Test R² of 0.152 — the most severe overfitting of any model. The sequential boosting loop memorises the training data almost perfectly but the small dataset means those patterns don't hold on the test set.
- **File split into headers was reverted.** Attempted to split `xgboost_scratch.cpp` into multiple header files (`xgboost_matrix.hpp`, `xgboost_tree.hpp`, etc.) but added complexity without benefit. Reverted to the single-file approach.

---

## Key Decisions Made

- **Dropped SMA Ratio** from the model — weight was -0.01, contributing no signal
- **Dropped 10-day and 15-day** volatility/momentum windows — caused overfitting with minimal gain
- **Chronological train/test split** used (not random) since time series data must not be shuffled
- **Normal Equation** used instead of gradient descent — dataset is small enough that it is faster and exact
- **`dropna()` done once at the end** after all features are built, not in individual cells
- **Validation set for grid search** carved from the last 20% of training data, not from the test set
- **Single-file C++ kept** over multi-header split — pybind11 extension only needs one source file to compile, splitting added no practical benefit
- **`os.add_dll_directory` used at import time** rather than adding MSYS2 to system PATH — keeps the environment clean

---

## What Is Next

### 1. Model Comparison Plot
- Plot actual vs predicted returns for all models on one chart

### 2. Backtesting
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
- Linear regression model cells start at `eced3c42`
- Random Forest implementation is in cell `de08afea` — must be run before the evaluation and grid search cells
- `self.trees = []` is set in `RandomForest.__init__` — if `fit` is called twice on the same instance it will keep appending trees. Always instantiate a fresh `RandomForest` object before fitting.
- To rebuild the C++ extension: run `.\libraries\build.bat` from the project root — it compiles and copies the `.pyd` to `notebooks/` automatically
- The `.pyd` filename is `xgboost_cpp.cp314-win_amd64.pyd` (Python 3.14 specific)
- Always call `os.add_dll_directory(r"C:\msys64\ucrt64\bin")` before `from xgboost_cpp import XGBoostCpp` or it will throw `ImportError: DLL load failed`
- C++ XGBoost cells start at `f3e250f7` — must have the `.pyd` in `notebooks/` before running
