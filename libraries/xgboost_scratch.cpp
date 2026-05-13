// Optimised XGBoost from scratch — C++17
//
// Three targeted optimisations over the naive version:
//
//  1. Column-major XGBMatrix
//     Feature columns are stored contiguously in memory.  The inner loop of
//     split-finding scans one feature column at a time, so every access is a
//     sequential read — the hardware prefetcher handles it perfectly.
//     Compare with vector<vector<double>> where each row is a separate heap
//     allocation; scanning a column forces a pointer-chase per row.
//
//  2. Pre-sorted feature indices
//     For each feature f, sort row indices by feature value once before
//     training.  During split-finding each node scans the pre-sorted list
//     (O(n_total)) instead of re-sorting the node's own samples every time
//     (O(n_node * log n_node) * n_nodes_per_level).
//
//  3. OpenMP feature parallelism
//     Split-finding across features is embarrassingly parallel — features are
//     independent.  #pragma omp parallel for over the feature loop gives
//     near-linear speedup on multi-core CPUs with no synchronisation cost
//     (each thread writes to its own SplitResult slot, reduced afterward).
//
// Compile (GCC / MinGW):
//   g++ -O3 -march=native -fopenmp -std=c++17 xgboost_scratch.cpp -o xgboost
// Without OpenMP (single-threaded, pragmas silently ignored):
//   g++ -O3 -march=native -std=c++17 xgboost_scratch.cpp -o xgboost
// ---------------------------------------------------------------------------

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>
#include <cmath>
#include <iomanip>
#include <random>
#include <limits>

#ifdef _OPENMP
  #include <omp.h>
  static inline int n_threads()  { return omp_get_max_threads(); }
  static inline int thread_id()  { return omp_get_thread_num();  }
#else
  static inline int n_threads()  { return 1; }
  static inline int thread_id()  { return 0; }
#endif

// ===========================================================================
// XGBMatrix — column-major storage + pre-sorted row indices
// ===========================================================================

struct XGBMatrix {
    int n_rows = 0, n_cols = 0;

    // Column-major flat array: element [row, col] lives at data[col*n_rows + row].
    // A full column is therefore a contiguous block — cache-friendly for the
    // split-finding inner loop which scans one column at a time.
    std::vector<double> data;

    // sorted_idx[f][k] = row index whose value for feature f has rank k.
    // Built once; reused by every node at every depth.
    std::vector<std::vector<int>> sorted_idx;

    XGBMatrix() = default;
    XGBMatrix(int rows, int cols)
        : n_rows(rows), n_cols(cols), data(rows * cols, 0.0) {}

    double  at(int row, int col) const { return data[col * n_rows + row]; }
    double& at(int row, int col)       { return data[col * n_rows + row]; }

    // Factory: accepts row-major input (e.g. from user code), optionally a
    // [start, end) slice so we don't copy the full dataset for train/test.
    static XGBMatrix from_rows(const std::vector<std::vector<double>>& X,
                               int start = 0, int end = -1) {
        if (end < 0) end = static_cast<int>(X.size());
        int rows = end - start;
        int cols = static_cast<int>(X[0].size());
        XGBMatrix m(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int f = 0; f < cols; ++f)
                m.at(i, f) = X[start + i][f];
        m.build_sorted();
        return m;
    }

    void build_sorted() {
        sorted_idx.assign(n_cols, std::vector<int>(n_rows));
        for (int f = 0; f < n_cols; ++f) {
            std::iota(sorted_idx[f].begin(), sorted_idx[f].end(), 0);
            std::sort(sorted_idx[f].begin(), sorted_idx[f].end(),
                [&](int a, int b) { return at(a, f) < at(b, f); });
        }
    }
};

// ===========================================================================
// Node
// ===========================================================================

struct Node {
    int    feature   = -1;
    double threshold = 0.0;
    double value     = 0.0;   // leaf prediction (XGBoost optimal weight)
    bool   is_leaf   = false;
    std::unique_ptr<Node> left, right;
};

// ===========================================================================
// SplitResult — one per thread to avoid false sharing
// ===========================================================================

struct alignas(64) SplitResult {   // 64-byte alignment = one cache line per slot
    double gain      = -std::numeric_limits<double>::infinity();
    int    feature   = -1;
    double threshold = 0.0;
};

// ===========================================================================
// XGBTree
//
// Regression tree whose splits maximise the XGBoost structure score gain.
//
// Leaf value  : w* = -G / (H + λ)            [XGBoost paper eq. 5]
// Struct score: S(G,H) = G² / (H + λ)
// Split gain  : 0.5*(S(G_L,H_L) + S(G_R,H_R) - S(G,H)) - γ
//
// min_child_weight is the minimum sum(h) required in each child — the
// principled XGBoost equivalent of min_samples_split (for MSE, h_i=1 so
// it is equivalent to the sample count).
// ===========================================================================

class XGBTree {
public:
    XGBTree(int max_depth, double min_child_weight, double lambda, double gamma)
        : max_depth_(max_depth), min_child_weight_(min_child_weight),
          lambda_(lambda), gamma_(gamma) {}

    // Move-only (unique_ptr member)
    XGBTree(XGBTree&&) noexcept = default;
    XGBTree& operator=(XGBTree&&) noexcept = default;
    XGBTree(const XGBTree&) = delete;

    void fit(const XGBMatrix& mat,
             const std::vector<double>& g,
             const std::vector<double>& h) {
        std::vector<int> idx(mat.n_rows);
        std::iota(idx.begin(), idx.end(), 0);
        root_ = grow(mat, g, h, idx, 0);
    }

    std::vector<double> predict(const XGBMatrix& mat) const {
        std::vector<double> out(mat.n_rows);
        for (int i = 0; i < mat.n_rows; ++i)
            out[i] = traverse(mat, i, root_.get());
        return out;
    }

private:
    int    max_depth_;
    double min_child_weight_;
    double lambda_, gamma_;
    std::unique_ptr<Node> root_;

    inline double structure_score(double G, double H) const {
        return (G * G) / (H + lambda_);
    }

    std::unique_ptr<Node> grow(const XGBMatrix& mat,
                                const std::vector<double>& g,
                                const std::vector<double>& h,
                                const std::vector<int>& idx,
                                int depth) {
        auto node = std::make_unique<Node>();

        double G_node = 0.0, H_node = 0.0;
        for (int i : idx) { G_node += g[i]; H_node += h[i]; }

        if (depth >= max_depth_ ||
            static_cast<int>(idx.size()) < 2 ||
            H_node < min_child_weight_) {
            node->is_leaf = true;
            node->value   = -G_node / (H_node + lambda_);
            return node;
        }

        // O(n_total) membership lookup: built once per grow() call.
        std::vector<bool> in_node(mat.n_rows, false);
        for (int i : idx) in_node[i] = true;

        double parent_score = structure_score(G_node, H_node);

        // Thread-local best split results (cache-line aligned to avoid false sharing)
        int nt = n_threads();
        std::vector<SplitResult> best(nt);

        // --- Feature scan: O(n_total) per feature, parallelised ---
        // Each thread independently finds the best split for its features and
        // writes to its own best[tid] slot.  No synchronisation needed during
        // the loop; we reduce afterward.
        #pragma omp parallel for schedule(dynamic) num_threads(nt)
        for (int f = 0; f < mat.n_cols; ++f) {
            int tid = thread_id();

            double G_left = 0.0, H_left = 0.0;
            double prev_val = std::numeric_limits<double>::lowest();
            bool   seen     = false;

            // Walk pre-sorted indices for feature f.
            // Skip samples not in this node (O(n_total) scan, O(1) lookup).
            // Evaluate a split point every time the feature value changes:
            //   threshold = prev_val  →  left: x[f] <= prev_val, right: x[f] > prev_val
            for (int rank = 0; rank < mat.n_rows; ++rank) {
                int i = mat.sorted_idx[f][rank];
                if (!in_node[i]) continue;

                double cur_val = mat.at(i, f);

                if (seen && cur_val > prev_val) {
                    double G_right = G_node - G_left;
                    double H_right = H_node - H_left;

                    if (H_left  >= min_child_weight_ &&
                        H_right >= min_child_weight_) {
                        double gain = 0.5 * (structure_score(G_left,  H_left)
                                           + structure_score(G_right, H_right)
                                           - parent_score)
                                      - gamma_;

                        if (gain > best[tid].gain)
                            best[tid] = { gain, f, prev_val };
                    }
                }

                G_left += g[i];
                H_left += h[i];
                prev_val = cur_val;
                seen     = true;
            }
        }

        // Reduce: pick the globally best split across all threads
        SplitResult global;
        for (auto& s : best)
            if (s.gain > global.gain) global = s;

        if (global.feature == -1 || global.gain <= 0.0) {
            node->is_leaf = true;
            node->value   = -G_node / (H_node + lambda_);
            return node;
        }

        // Partition node indices into left / right children
        std::vector<int> left_idx, right_idx;
        left_idx.reserve(idx.size());
        right_idx.reserve(idx.size());
        for (int i : idx) {
            if (mat.at(i, global.feature) <= global.threshold)
                left_idx.push_back(i);
            else
                right_idx.push_back(i);
        }

        node->feature   = global.feature;
        node->threshold = global.threshold;
        node->left  = grow(mat, g, h, left_idx,  depth + 1);
        node->right = grow(mat, g, h, right_idx, depth + 1);
        return node;
    }

    double traverse(const XGBMatrix& mat, int row, const Node* node) const {
        if (node->is_leaf) return node->value;
        return (mat.at(row, node->feature) <= node->threshold)
            ? traverse(mat, row, node->left.get())
            : traverse(mat, row, node->right.get());
    }
};

// ===========================================================================
// XGBoost
//
// Boosting loop:
//   F_0 = mean(y)
//   for m in 1..n_estimators:
//       g_i = F_{m-1}(x_i) - y_i          (MSE gradient)
//       h_i = 1                             (MSE hessian — constant)
//       tree_m = fit XGBTree to (g, h)
//       F_m(x) = F_{m-1}(x) + eta * tree_m(x)
// ===========================================================================

class XGBoost {
public:
    XGBoost(int    n_estimators     = 100,
            double learning_rate    = 0.1,
            int    max_depth        = 3,
            double min_child_weight = 1.0,
            double lambda           = 1.0,
            double gamma            = 0.0)
        : n_estimators_(n_estimators), learning_rate_(learning_rate),
          max_depth_(max_depth), min_child_weight_(min_child_weight),
          lambda_(lambda), gamma_(gamma), base_pred_(0.0) {}

    void fit(const XGBMatrix& mat, const std::vector<double>& y) {
        int n = mat.n_rows;
        base_pred_ = std::accumulate(y.begin(), y.end(), 0.0) / n;
        std::vector<double> F(n, base_pred_);

        for (int m = 0; m < n_estimators_; ++m) {
            std::vector<double> g(n), h(n, 1.0);
            for (int i = 0; i < n; ++i) g[i] = F[i] - y[i];

            XGBTree tree(max_depth_, min_child_weight_, lambda_, gamma_);
            tree.fit(mat, g, h);

            auto delta = tree.predict(mat);
            for (int i = 0; i < n; ++i)
                F[i] += learning_rate_ * delta[i];

            trees_.push_back(std::move(tree));
        }
    }

    std::vector<double> predict(const XGBMatrix& mat) const {
        std::vector<double> F(mat.n_rows, base_pred_);
        for (const auto& tree : trees_) {
            auto delta = tree.predict(mat);
            for (int i = 0; i < mat.n_rows; ++i)
                F[i] += learning_rate_ * delta[i];
        }
        return F;
    }

private:
    int    n_estimators_;
    double learning_rate_;
    int    max_depth_;
    double min_child_weight_;
    double lambda_, gamma_;
    double base_pred_;
    std::vector<XGBTree> trees_;
};

// ===========================================================================
// Metrics
// ===========================================================================

double compute_mse(const std::vector<double>& y, const std::vector<double>& p) {
    double s = 0.0;
    for (size_t i = 0; i < y.size(); ++i) { double d = y[i]-p[i]; s += d*d; }
    return s / y.size();
}

double compute_r2(const std::vector<double>& y, const std::vector<double>& p) {
    double mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    double ss_tot = 0.0, ss_res = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        ss_tot += (y[i]-mean)*(y[i]-mean);
        ss_res += (y[i]-p[i]) *(y[i]-p[i]);
    }
    return 1.0 - ss_res / ss_tot;
}

// ===========================================================================
// main — synthetic demo matching the Python notebook's feature structure
// (excluded when compiling as a Python extension)
// ===========================================================================

#ifndef XGB_EXTENSION
int main() {
    std::mt19937 rng(42);
    std::normal_distribution<double>  noise(0.0, 0.005);
    std::uniform_real_distribution<double> feat(-0.05, 0.05);

    // 220 samples, 7 features — same shape as the Python notebook after dropna
    int n = 220, n_features = 7;
    std::vector<std::vector<double>> X(n, std::vector<double>(n_features));
    std::vector<double> y(n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n_features; ++j)
            X[i][j] = feat(rng);
        // Mimics the learned theta weights from linear regression:
        // Momentum_5days (f4) dominates, lags and vol are negative
        y[i] =  2.4  * X[i][4]    // Momentum_5days
              - 0.42 * X[i][0]    // Lag1
              - 0.42 * X[i][3]    // Vol_30day
              + noise(rng);
    }

    int split = static_cast<int>(n * 0.8);

    // Build XGBMatrix — sorted indices computed once here
    auto mat_train = XGBMatrix::from_rows(X, 0,     split);
    auto mat_test  = XGBMatrix::from_rows(X, split, n);
    std::vector<double> y_train(y.begin(), y.begin() + split);
    std::vector<double> y_test (y.begin() + split, y.end());

    XGBoost model(
        /*n_estimators=*/    100,
        /*learning_rate=*/   0.1,
        /*max_depth=*/       3,
        /*min_child_weight=*/1.0,
        /*lambda=*/          1.0,
        /*gamma=*/           0.0
    );

    model.fit(mat_train, y_train);

    auto p_train = model.predict(mat_train);
    auto p_test  = model.predict(mat_test);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Optimised XGBoost (C++) ===\n";
    #ifdef _OPENMP
    std::cout << "Threads   : " << n_threads() << "\n";
    #else
    std::cout << "Threads   : 1 (OpenMP not enabled)\n";
    #endif
    std::cout << "Train MSE : " << compute_mse(y_train, p_train) << "\n";
    std::cout << "Test  MSE : " << compute_mse(y_test,  p_test)  << "\n";
    std::cout << "Train R²  : " << compute_r2 (y_train, p_train) << "\n";
    std::cout << "Test  R²  : " << compute_r2 (y_test,  p_test)  << "\n";

    return 0;
}
#endif // !XGB_EXTENSION

// ===========================================================================
// pybind11 Python extension
// Compiled when -DXGB_EXTENSION is passed.  Exposes XGBoostCpp to Python
// as a class that accepts and returns numpy float64 arrays.
// ===========================================================================

#ifdef XGB_EXTENSION

#define _hypot hypot          // MinGW/Windows fix for Python math.h conflict
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using NpArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Convert a 2D numpy array (row-major float64) into an XGBMatrix.
// forcecast ensures non-float64 arrays are automatically converted.
static XGBMatrix from_numpy(NpArray arr) {
    auto buf = arr.request();
    if (buf.ndim != 2)
        throw std::runtime_error("X must be a 2D array");
    int n_rows = static_cast<int>(buf.shape[0]);
    int n_cols = static_cast<int>(buf.shape[1]);
    double* ptr = static_cast<double*>(buf.ptr);
    XGBMatrix mat(n_rows, n_cols);
    for (int i = 0; i < n_rows; ++i)
        for (int f = 0; f < n_cols; ++f)
            mat.at(i, f) = ptr[i * n_cols + f];
    mat.build_sorted();
    return mat;
}

class PyXGBoost {
    XGBoost model_;
public:
    PyXGBoost(int n_est, double lr, int depth, double mcw, double lam, double gam)
        : model_(n_est, lr, depth, mcw, lam, gam) {}

    void fit(NpArray X, NpArray y) {
        auto mat   = from_numpy(X);
        auto y_buf = y.request();
        std::vector<double> y_vec(
            static_cast<double*>(y_buf.ptr),
            static_cast<double*>(y_buf.ptr) + y_buf.shape[0]);
        model_.fit(mat, y_vec);
    }

    NpArray predict(NpArray X) {
        auto mat   = from_numpy(X);
        auto preds = model_.predict(mat);
        auto out   = py::array_t<double>(static_cast<py::ssize_t>(preds.size()));
        std::copy(preds.begin(), preds.end(),
                  static_cast<double*>(out.request().ptr));
        return out;
    }
};

PYBIND11_MODULE(xgboost_cpp, m) {
    m.doc() = "XGBoost from scratch — optimised C++ Python extension";
    py::class_<PyXGBoost>(m, "XGBoostCpp")
        .def(py::init<int, double, int, double, double, double>(),
             py::arg("n_estimators")     = 100,
             py::arg("learning_rate")    = 0.1,
             py::arg("max_depth")        = 3,
             py::arg("min_child_weight") = 1.0,
             py::arg("lambda_")          = 1.0,
             py::arg("gamma")            = 0.0)
        .def("fit",     &PyXGBoost::fit,     "fit(X, y) — X: 2D float64 array, y: 1D float64 array")
        .def("predict", &PyXGBoost::predict, "predict(X) — returns 1D float64 array");
}

#endif // XGB_EXTENSION
