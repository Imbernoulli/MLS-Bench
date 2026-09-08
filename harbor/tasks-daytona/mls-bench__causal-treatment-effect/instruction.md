# MLS-Bench: causal-treatment-effect

# Causal Treatment Effect Estimation

## Research Question
Design a novel estimator for **Conditional Average Treatment Effects (CATE)**
from observational data that is accurate, robust to confounding, and
generalizes across synthetic data-generating processes.

## Background
Estimating heterogeneous treatment effects -- how the causal effect of a
treatment varies across individuals -- is a core problem in causal inference.
Given observational data with covariates `X`, binary treatment `T`, and
outcome `Y`, the goal is to estimate
`tau(x) = E[Y(1) - Y(0) | X = x]`, the conditional average treatment effect.

Key challenges include:
- **Confounding**: treatment assignment depends on covariates, so naive
  comparisons are biased.
- **Heterogeneity**: treatment effects vary across the covariate space in
  complex, nonlinear ways.
- **Model misspecification**: true response surfaces may not match parametric
  assumptions.
- **Double robustness**: ideally, the estimator is consistent if either the
  outcome model or the propensity model is correct.

Classical approaches include S-Learner (single model), T-Learner (separate
outcome models per arm), and IPW (propensity reweighting). Modern methods use
orthogonalization or debiasing for better convergence rates: see Athey & Wager,
"Estimation and Inference of Heterogeneous Treatment Effects using Random
Forests," JASA 113(523), 2018 (arXiv:1510.04342); Kennedy, "Towards optimal
doubly robust estimation of heterogeneous causal effects," Electronic Journal
of Statistics 17(2), 2023 (arXiv:2004.14497); and Nie & Wager, "Quasi-Oracle
Estimation of Heterogeneous Treatment Effects," Biometrika 108(2), 2021
(arXiv:1712.04912).

## Task
Modify the `CATEEstimator` class in `custom_cate.py`. The estimator must
implement:

```python
class CATEEstimator:
    def fit(self, X, T, Y) -> "CATEEstimator":
        """Learn from observational covariates X, binary treatment T, outcome Y."""

    def predict(self, X):
        """Return predicted individual treatment effects tau_hat for each row of X."""
```

scikit-learn, numpy, and scipy are available.

Valid contributions may combine outcome modeling, propensity modeling,
orthogonalization, weighting, residualization, forests, neural models, or other
modular CATE ideas, as long as they address confounding and treatment-effect
heterogeneity.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `scikit-learn/custom_cate.py`
- editable lines **84–155**




## Readable Context


### `scikit-learn/custom_cate.py`  [EDITABLE — lines 84–155 only]

```python
     1: # Custom CATE Estimator for MLS-Bench
     2: #
     3: # EDITABLE section: CATEEstimator class (the treatment effect estimator).
     4: # FIXED sections: everything else (input loading, cross-fitting, CLI).
     5: #
     6: # Research question: Design a novel estimator for Conditional Average Treatment
     7: # Effects (CATE) across explicitly synthetic observational DGP families.
     8: #
     9: # NOTE: The data-generating process and the ground-truth treatment effect are
    10: # NOT part of this program. The harness pre-generates the observational inputs
    11: # (X, T, Y) and scores your cross-fitted predictions against held-out truth in
    12: # a separate process. Your estimator only ever sees (X, T, Y).
    13: 
    14: import os
    15: import io
    16: import argparse
    17: import base64
    18: import time
    19: import warnings
    20: from abc import ABC, abstractmethod
    21: 
    22: import numpy as np
    23: from scipy import stats
    24: from sklearn.model_selection import KFold, cross_val_predict
    25: from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
    26: from sklearn.ensemble import (
    27:     RandomForestRegressor,
    28:     RandomForestClassifier,
    29:     GradientBoostingRegressor,
    30:     GradientBoostingClassifier,
    31: )
    32: from sklearn.tree import DecisionTreeRegressor
    33: from sklearn.neural_network import MLPRegressor, MLPClassifier
    34: from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    35: from sklearn.pipeline import Pipeline
    36: from sklearn.base import clone
    37: 
    38: warnings.filterwarnings("ignore")
    39: 
    40: 
    41: # =====================================================================
    42: # FIXED: Base class for CATE estimators
    43: # =====================================================================
    44: 
    45: class BaseCATEEstimator(ABC):
    46:     """Abstract base class for CATE estimators.
    47: 
    48:     All estimators must implement:
    49:         fit(X, T, Y) -> self
    50:         predict(X) -> tau_hat array of shape (n,)
    51:     """
    52: 
    53:     @abstractmethod
    54:     def fit(self, X, T, Y):
    55:         """Fit the estimator on observational data.
    56: 
    57:         Args:
    58:             X: (n, p) covariate matrix (numpy array)
    59:             T: (n,) binary treatment indicator (0 or 1)
    60:             Y: (n,) observed outcomes (continuous)
    61: 
    62:         Returns:
    63:             self
    64:         """
    65:         pass
    66: 
    67:     @abstractmethod
    68:     def predict(self, X):
    69:         """Predict CATE for given covariates.
    70: 
    71:         Args:
    72:             X: (n, p) covariate matrix
    73: 
    74:         Returns:
    75:             tau_hat: (n,) array of estimated treatment effects
    76:         """
    77:         pass
    78: 
    79: 
    80: # =====================================================================
    81: # EDITABLE: Custom CATE Estimator
    82: # =====================================================================
    83: 
    84: class CATEEstimator(BaseCATEEstimator):
    85:     """Custom CATE (Conditional Average Treatment Effect) estimator.
    86: 
    87:     Design a novel estimator for heterogeneous treatment effects from
    88:     observational data. Your estimator receives covariates X, binary
    89:     treatment T, and outcomes Y, and must estimate tau(x) = E[Y(1)-Y(0)|X=x].
    90: 
    91:     Key challenges:
    92:     - Confounding: treatment assignment depends on covariates
    93:     - Heterogeneity: treatment effects vary across individuals
    94:     - Model misspecification: response surfaces may be nonlinear
    95:     - Finite-sample performance: must work well with limited data
    96: 
    97:     Approaches to consider:
    98:     - Meta-learners (S/T/X/R/DR-Learner frameworks)
    99:     - Propensity score methods (weighting, matching, doubly robust)
   100:     - Tree-based methods (causal forests, Bayesian additive regression trees)
   101:     - Representation learning for treatment effects
   102:     - Kernel methods or local regression for CATE
   103:     - Ensemble methods combining multiple estimators
   104: 
   105:     Available imports (in FIXED section above):
   106:         numpy, scipy.stats, sklearn (all submodules)
   107: 
   108:     Interface contract:
   109:         fit(X, T, Y) -> self
   110:         predict(X) -> tau_hat of shape (n,)
   111:     """
   112: 
   113:     def __init__(self):
   114:         """Initialize the CATE estimator.
   115: 
   116:         TODO: Set up any models, hyperparameters, or data structures needed.
   117:         """
   118:         pass
   119: 
   120:     def fit(self, X, T, Y):
   121:         """Fit the estimator on observational data.
   122: 
   123:         Args:
   124:             X: (n, p) numpy array of covariates
   125:             T: (n,) numpy array of binary treatment indicators (0 or 1)
   126:             Y: (n,) numpy array of observed outcomes
   127: 
   128:         Returns:
   129:             self
   130: 
   131:         TODO: Implement your CATE estimation algorithm.
   132:         The default implementation is a simple S-Learner placeholder.
   133:         """
   134:         # Placeholder: simple S-Learner (augmented features)
   135:         n, p = X.shape
   136:         XT = np.column_stack([X, T.reshape(-1, 1)])
   137:         self._model = Ridge(alpha=1.0)
   138:         self._model.fit(XT, Y)
   139:         return self
   140: 
   141:     def predict(self, X):
   142:         """Predict CATE for given covariates.
   143: 
   144:         Args:
   145:             X: (n, p) numpy array of covariates
   146: 
   147:         Returns:
   148:             tau_hat: (n,) numpy array of estimated treatment effects
   149: 
   150:         TODO: Implement prediction of individual treatment effects.
   151:         """
   152:         n = X.shape[0]
   153:         X1 = np.column_stack([X, np.ones((n, 1))])
   154:         X0 = np.column_stack([X, np.zeros((n, 1))])
   155:         return self._model.predict(X1) - self._model.predict(X0)
   156: 
   157: 
   158: # =====================================================================
   159: # FIXED: Input loading + cross-fitting + prediction emit
   160: # =====================================================================
   161: 
   162: # --dataset is supplied as an opaque per-dataset token; the real dataset
   163: # identity is held out by the harness and is not present in this process.
   164: 
   165: 
   166: def _inputs_dir():
   167:     """Directory holding the pre-generated (X, T, Y) inputs for this task."""
   168:     env = os.environ.get("CATE_INPUTS_DIR")
   169:     if env:
   170:         return env
   171:     return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cate_inputs")
   172: 
   173: 
   174: def load_inputs(dataset, data_seed):
   175:     """Load the pre-generated observational inputs for (dataset, data_seed).
   176: 
   177:     Only (X, T, Y) are available — the true treatment effect is held out by the
   178:     harness and is not present in this process.
   179:     """
   180:     path = os.path.join(_inputs_dir(), f"{dataset}_seed{data_seed}.npz.b64")
   181:     with open(path, "r") as f:
   182:         raw = base64.b64decode(f.read())
   183:     with np.load(io.BytesIO(raw)) as d:
   184:         return d["X"], d["T"], d["Y"]
   185: 
   186: 
   187: def cross_fit_predict(X, T, Y, n_splits, seed):
   188:     """Cross-fitted CATE prediction (same K-fold protocol as before).
   189: 
   190:     Fit on K-1 folds, predict held-out fold; aggregate to per-row tau_hat.
   191:     A fresh deep-copy of the estimator is used per fold (cross-fitting).
   192:     """
   193:     import copy
   194:     base = CATEEstimator()
   195:     kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
   196:     tau_hat_all = np.zeros(len(X))
   197:     for train_idx, test_idx in kf.split(X):
   198:         est = copy.deepcopy(base)
   199:         est.fit(X[train_idx], T[train_idx], Y[train_idx])
   200:         tau_hat_all[test_idx] = est.predict(X[test_idx])
   201:     return tau_hat_all
   202: 
   203: 
   204: def main():
   205:     parser = argparse.ArgumentParser(description="CATE Estimation Benchmark")
   206:     parser.add_argument("--dataset", type=str, required=True,
   207:                         help="Opaque dataset token (the real dataset is held out)")
   208:     parser.add_argument("--seed", type=int, default=42,
   209:                         help="Base random seed (selects the pre-generated inputs)")
   210:     parser.add_argument("--n-splits", type=int, default=5,
   211:                         help="Number of cross-validation folds")
   212:     parser.add_argument("--n-reps", type=int, default=10,
   213:                         help="Number of repetitions with different data seeds")
   214:     args = parser.parse_args()
   215: 
   216:     print(f"Evaluating on {args.dataset} (seed={args.seed}, "
   217:           f"n_splits={args.n_splits}, n_reps={args.n_reps})", flush=True)
   218: 
   219:     start = time.time()
   220:     for rep in range(args.n_reps):
   221:         data_seed = args.seed + rep * 1000
   222:         X, T, Y = load_inputs(args.dataset, data_seed)
   223:         tau_hat = cross_fit_predict(X, T, Y, args.n_splits, data_seed)
   224: 
   225:         # Emit the cross-fitted predictions for the held-out scorer. We do NOT
   226:         # have the true tau, so we cannot (and do not) compute the metric here.
   227:         payload = base64.b64encode(
   228:             np.ascontiguousarray(tau_hat, dtype=np.float64).tobytes()
   229:         ).decode("ascii")
   230:         print(f"CATE_PRED dataset={args.dataset} seed={args.seed} rep={rep} "
   231:               f"data_seed={data_seed} n={tau_hat.shape[0]} tau_hat={payload}",
   232:               flush=True)
   233: 
   234:     print(f"Done {args.dataset}: emitted {args.n_reps} prediction sets in "
   235:           f"{time.time() - start:.1f}s", flush=True)
   236: 
   237: 
   238: if __name__ == "__main__":
   239:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `s_learner` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_cate.py`:

```python
Lines 84–113:
    81: # EDITABLE: Custom CATE Estimator
    82: # =====================================================================
    83: 
    84: class CATEEstimator(BaseCATEEstimator):
    85:     """S-Learner: single model approach to CATE estimation.
    86: 
    87:     Fits a single outcome model mu(X, T) on the combined data, then
    88:     estimates CATE as mu(X, 1) - mu(X, 0).
    89:     Uses GradientBoostingRegressor as the base learner for flexibility.
    90:     """
    91: 
    92:     def __init__(self):
    93:         self._seed = int(os.environ.get("SEED", "42"))
    94:         self._model = GradientBoostingRegressor(
    95:             n_estimators=200,
    96:             max_depth=4,
    97:             learning_rate=0.1,
    98:             min_samples_leaf=20,
    99:             subsample=0.8,
   100:             random_state=self._seed,
   101:         )
   102: 
   103:     def fit(self, X, T, Y):
   104:         n, p = X.shape
   105:         XT = np.column_stack([X, T.reshape(-1, 1)])
   106:         self._model.fit(XT, Y)
   107:         return self
   108: 
   109:     def predict(self, X):
   110:         n = X.shape[0]
   111:         X1 = np.column_stack([X, np.ones((n, 1))])
   112:         X0 = np.column_stack([X, np.zeros((n, 1))])
   113:         return self._model.predict(X1) - self._model.predict(X0)
   114: 
   115: 
   116: # =====================================================================
```

### `t_learner` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_cate.py`:

```python
Lines 84–119:
    81: # EDITABLE: Custom CATE Estimator
    82: # =====================================================================
    83: 
    84: class CATEEstimator(BaseCATEEstimator):
    85:     """T-Learner: two separate models for treated and control groups.
    86: 
    87:     Fits mu0(X) on control data and mu1(X) on treated data, then
    88:     estimates CATE as mu1(X) - mu0(X).
    89:     Uses GradientBoostingRegressor for both models.
    90:     """
    91: 
    92:     def __init__(self):
    93:         self._seed = int(os.environ.get("SEED", "42"))
    94:         self._model0 = GradientBoostingRegressor(
    95:             n_estimators=200,
    96:             max_depth=4,
    97:             learning_rate=0.1,
    98:             min_samples_leaf=20,
    99:             subsample=0.8,
   100:             random_state=self._seed,
   101:         )
   102:         self._model1 = GradientBoostingRegressor(
   103:             n_estimators=200,
   104:             max_depth=4,
   105:             learning_rate=0.1,
   106:             min_samples_leaf=20,
   107:             subsample=0.8,
   108:             random_state=self._seed + 1,
   109:         )
   110: 
   111:     def fit(self, X, T, Y):
   112:         mask0 = T == 0
   113:         mask1 = T == 1
   114:         self._model0.fit(X[mask0], Y[mask0])
   115:         self._model1.fit(X[mask1], Y[mask1])
   116:         return self
   117: 
   118:     def predict(self, X):
   119:         return self._model1.predict(X) - self._model0.predict(X)
   120: 
   121: 
   122: # =====================================================================
```

### `ipw` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_cate.py`:

```python
Lines 84–123:
    81: # EDITABLE: Custom CATE Estimator
    82: # =====================================================================
    83: 
    84: class CATEEstimator(BaseCATEEstimator):
    85:     """IPW-based CATE estimator with propensity score weighting.
    86: 
    87:     1. Estimate propensity score e(X) = P(T=1|X) via logistic regression.
    88:     2. Construct IPW pseudo-outcomes: Y_ipw = T*Y/e(X) - (1-T)*Y/(1-e(X)).
    89:     3. Fit a regression model on X -> Y_ipw for CATE estimation.
    90: 
    91:     Clips propensity scores to [0.05, 0.95] for stability.
    92:     """
    93: 
    94:     def __init__(self):
    95:         self._seed = int(os.environ.get("SEED", "42"))
    96:         self._prop_model = GradientBoostingClassifier(
    97:             n_estimators=200, max_depth=3, learning_rate=0.1,
    98:             min_samples_leaf=20, subsample=0.8, random_state=self._seed,
    99:         )
   100:         self._outcome_model = GradientBoostingRegressor(
   101:             n_estimators=200,
   102:             max_depth=4,
   103:             learning_rate=0.1,
   104:             min_samples_leaf=20,
   105:             subsample=0.8,
   106:             random_state=self._seed + 1,
   107:         )
   108: 
   109:     def fit(self, X, T, Y):
   110:         # Estimate propensity scores
   111:         self._prop_model.fit(X, T)
   112:         e_hat = self._prop_model.predict_proba(X)[:, 1]
   113:         e_hat = np.clip(e_hat, 0.05, 0.95)
   114: 
   115:         # IPW pseudo-outcomes
   116:         pseudo_outcome = T * Y / e_hat - (1 - T) * Y / (1 - e_hat)
   117: 
   118:         # Fit outcome model on pseudo-outcomes
   119:         self._outcome_model.fit(X, pseudo_outcome)
   120:         return self
   121: 
   122:     def predict(self, X):
   123:         return self._outcome_model.predict(X)
   124: 
   125: 
   126: # =====================================================================
```

### `causal_forest` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_cate.py`:

```python
Lines 84–165:
    81: # EDITABLE: Custom CATE Estimator
    82: # =====================================================================
    83: 
    84: class CATEEstimator(BaseCATEEstimator):
    85:     """Causal Forest (via econml CausalForestDML).
    86: 
    87:     Combines double machine learning (DML) for debiasing with
    88:     generalized random forests for heterogeneous effect estimation.
    89: 
    90:     Steps:
    91:     1. Cross-fit nuisance models: E[Y|X] and E[T|X]
    92:     2. Compute residuals: Y_res = Y - E[Y|X], T_res = T - E[T|X]
    93:     3. Fit a causal forest on residualized outcomes
    94: 
    95:     Falls back to a pure-sklearn implementation if econml is unavailable.
    96:     """
    97: 
    98:     def __init__(self):
    99:         self._seed = int(os.environ.get("SEED", "42"))
   100:         self._use_econml = True
   101:         try:
   102:             from econml.dml import CausalForestDML
   103:             self._cf = CausalForestDML(
   104:                 model_y=GradientBoostingRegressor(
   105:                     n_estimators=100, max_depth=3, learning_rate=0.1,
   106:                     min_samples_leaf=20, random_state=self._seed,
   107:                 ),
   108:                 model_t=GradientBoostingRegressor(
   109:                     n_estimators=100, max_depth=3, learning_rate=0.1,
   110:                     min_samples_leaf=20, random_state=self._seed + 1,
   111:                 ),
   112:                 n_estimators=500,
   113:                 min_samples_leaf=5,
   114:                 max_depth=None,
   115:                 honest=True,
   116:                 inference=False,
   117:                 random_state=self._seed + 2,
   118:                 cv=3,
   119:             )
   120:         except ImportError:
   121:             self._use_econml = False
   122:             # Fallback: manual residualization + random forest
   123:             self._model_y = GradientBoostingRegressor(
   124:                 n_estimators=200, max_depth=4, learning_rate=0.1,
   125:                 min_samples_leaf=20, random_state=self._seed,
   126:             )
   127:             self._model_t = GradientBoostingClassifier(
   128:                 n_estimators=200, max_depth=4, learning_rate=0.1,
   129:                 min_samples_leaf=20, random_state=self._seed + 1,
   130:             )
   131:             self._cate_model = RandomForestRegressor(
   132:                 n_estimators=500, min_samples_leaf=5,
   133:                 max_features="sqrt", random_state=self._seed + 2,
   134:             )
   135: 
   136:     def fit(self, X, T, Y):
   137:         if self._use_econml:
   138:             self._cf.fit(Y, T, X=X)
   139:         else:
   140:             # Manual DML: cross-fit residuals
   141:             kf = KFold(n_splits=3, shuffle=True, random_state=self._seed)
   142:             Y_res = np.zeros_like(Y)
   143:             T_res = np.zeros_like(T, dtype=float)
   144: 
   145:             for train_idx, val_idx in kf.split(X):
   146:                 my = clone(self._model_y).fit(X[train_idx], Y[train_idx])
   147:                 mt = clone(self._model_t).fit(X[train_idx], T[train_idx])
   148:                 Y_res[val_idx] = Y[val_idx] - my.predict(X[val_idx])
   149:                 T_res[val_idx] = T[val_idx] - mt.predict_proba(X[val_idx])[:, 1]
   150: 
   151:             # R-Learner-style pseudo-outcome with stable divisor + sample
   152:             # weighting so small |T_res| doesn't explode the fit.
   153:             safe_T = np.where(np.abs(T_res) > 0.01, T_res, np.sign(T_res) * 0.01 + 1e-8)
   154:             pseudo = Y_res / safe_T
   155:             weights = T_res ** 2
   156:             q = np.percentile(np.abs(pseudo), 95)
   157:             pseudo = np.clip(pseudo, -q, q)
   158:             self._cate_model.fit(X, pseudo, sample_weight=weights)
   159:         return self
   160: 
   161:     def predict(self, X):
   162:         if self._use_econml:
   163:             return self._cf.effect(X).flatten()
   164:         else:
   165:             return self._cate_model.predict(X)
   166: 
   167: 
   168: # =====================================================================
```

### `dr_learner` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_cate.py`:

```python
Lines 84–173:
    81: # EDITABLE: Custom CATE Estimator
    82: # =====================================================================
    83: 
    84: class CATEEstimator(BaseCATEEstimator):
    85:     """DR-Learner: Doubly Robust CATE estimation.
    86: 
    87:     Steps:
    88:     1. Cross-fit nuisance models:
    89:        - mu0(X) = E[Y|X, T=0], mu1(X) = E[Y|X, T=1]  (outcome models)
    90:        - e(X) = P(T=1|X)  (propensity score)
    91:     2. Compute doubly-robust pseudo-outcomes:
    92:        phi(X) = mu1(X) - mu0(X)
    93:               + T*(Y - mu1(X))/e(X)
    94:               - (1-T)*(Y - mu0(X))/(1-e(X))
    95:     3. Fit a final CATE model on X -> phi(X)
    96:     """
    97: 
    98:     def __init__(self):
    99:         self._seed = int(os.environ.get("SEED", "42"))
   100: 
   101:     def _make_model_y(self):
   102:         return GradientBoostingRegressor(
   103:             n_estimators=200, max_depth=4, learning_rate=0.1,
   104:             min_samples_leaf=20, subsample=0.8, random_state=self._seed,
   105:         )
   106: 
   107:     def _make_model_t(self):
   108:         return GradientBoostingClassifier(
   109:             n_estimators=200, max_depth=3, learning_rate=0.1,
   110:             min_samples_leaf=20, subsample=0.8, random_state=self._seed + 1,
   111:         )
   112: 
   113:     def _make_cate_model(self):
   114:         return GradientBoostingRegressor(
   115:             n_estimators=200, max_depth=3, learning_rate=0.05,
   116:             min_samples_leaf=20, subsample=0.8, random_state=self._seed + 2,
   117:         )
   118: 
   119:     def fit(self, X, T, Y):
   120:         n = len(Y)
   121: 
   122:         # Cross-fit nuisance models
   123:         kf = KFold(n_splits=5, shuffle=True, random_state=self._seed)
   124:         mu0_hat = np.zeros(n)
   125:         mu1_hat = np.zeros(n)
   126:         e_hat = np.zeros(n)
   127: 
   128:         for train_idx, val_idx in kf.split(X):
   129:             # Outcome models (separate for T=0 and T=1)
   130:             mask0_train = T[train_idx] == 0
   131:             mask1_train = T[train_idx] == 1
   132: 
   133:             m0 = self._make_model_y()
   134:             m1 = self._make_model_y()
   135: 
   136:             if mask0_train.sum() > 5:
   137:                 m0.fit(X[train_idx[mask0_train]], Y[train_idx[mask0_train]])
   138:                 mu0_hat[val_idx] = m0.predict(X[val_idx])
   139:             else:
   140:                 mu0_hat[val_idx] = Y[T == 0].mean() if (T == 0).sum() > 0 else Y.mean()
   141: 
   142:             if mask1_train.sum() > 5:
   143:                 m1.fit(X[train_idx[mask1_train]], Y[train_idx[mask1_train]])
   144:                 mu1_hat[val_idx] = m1.predict(X[val_idx])
   145:             else:
   146:                 mu1_hat[val_idx] = Y[T == 1].mean() if (T == 1).sum() > 0 else Y.mean()
   147: 
   148:             # Propensity model
   149:             mt = self._make_model_t()
   150:             mt.fit(X[train_idx], T[train_idx])
   151:             e_hat[val_idx] = mt.predict_proba(X[val_idx])[:, 1]
   152: 
   153:         # Clip propensity scores
   154:         e_hat = np.clip(e_hat, 0.05, 0.95)
   155: 
   156:         # Doubly-robust pseudo-outcomes
   157:         pseudo = (
   158:             mu1_hat - mu0_hat
   159:             + T * (Y - mu1_hat) / e_hat
   160:             - (1 - T) * (Y - mu0_hat) / (1 - e_hat)
   161:         )
   162: 
   163:         # Clip extreme pseudo-outcomes
   164:         q = np.percentile(np.abs(pseudo), 97)
   165:         pseudo = np.clip(pseudo, -q, q)
   166: 
   167:         # Fit CATE model on pseudo-outcomes
   168:         self._cate_model = self._make_cate_model()
   169:         self._cate_model.fit(X, pseudo)
   170:         return self
   171: 
   172:     def predict(self, X):
   173:         return self._cate_model.predict(X)
   174: 
   175: 
   176: # =====================================================================
```

### `r_learner` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_cate.py`:

```python
Lines 84–161:
    81: # EDITABLE: Custom CATE Estimator
    82: # =====================================================================
    83: 
    84: class CATEEstimator(BaseCATEEstimator):
    85:     """R-Learner: Robinson decomposition for CATE estimation.
    86: 
    87:     Based on the Robinson (1988) decomposition:
    88:         Y - m(X) = (T - e(X)) * tau(X) + epsilon
    89: 
    90:     Steps:
    91:     1. Cross-fit nuisance models:
    92:        - m(X) = E[Y|X]  (marginal outcome model)
    93:        - e(X) = P(T=1|X)  (propensity score)
    94:     2. Compute residuals: Y_tilde = Y - m(X), T_tilde = T - e(X)
    95:     3. Estimate tau(X) by minimizing: sum_i (Y_tilde_i - T_tilde_i * tau(X_i))^2
    96:        This is equivalent to weighted least squares with weight T_tilde^2.
    97:     """
    98: 
    99:     def __init__(self):
   100:         self._seed = int(os.environ.get("SEED", "42"))
   101: 
   102:     def _make_model_y(self):
   103:         return GradientBoostingRegressor(
   104:             n_estimators=200, max_depth=4, learning_rate=0.1,
   105:             min_samples_leaf=20, subsample=0.8, random_state=self._seed,
   106:         )
   107: 
   108:     def _make_model_t(self):
   109:         return GradientBoostingClassifier(
   110:             n_estimators=200, max_depth=3, learning_rate=0.1,
   111:             min_samples_leaf=20, subsample=0.8, random_state=self._seed + 1,
   112:         )
   113: 
   114:     def fit(self, X, T, Y):
   115:         n = len(Y)
   116: 
   117:         # Cross-fit nuisance models
   118:         kf = KFold(n_splits=5, shuffle=True, random_state=self._seed)
   119:         m_hat = np.zeros(n)
   120:         e_hat = np.zeros(n)
   121: 
   122:         for train_idx, val_idx in kf.split(X):
   123:             # Outcome model E[Y|X]
   124:             my = self._make_model_y()
   125:             my.fit(X[train_idx], Y[train_idx])
   126:             m_hat[val_idx] = my.predict(X[val_idx])
   127: 
   128:             # Propensity model P(T=1|X)
   129:             mt = self._make_model_t()
   130:             mt.fit(X[train_idx], T[train_idx])
   131:             e_hat[val_idx] = mt.predict_proba(X[val_idx])[:, 1]
   132: 
   133:         # Clip propensity scores
   134:         e_hat = np.clip(e_hat, 0.05, 0.95)
   135: 
   136:         # Residuals
   137:         Y_tilde = Y - m_hat
   138:         T_tilde = T - e_hat
   139: 
   140:         # R-Learner: pseudo-outcome = Y_tilde / T_tilde
   141:         # Weight = T_tilde^2 (higher weight where treatment variation is larger)
   142:         weights = T_tilde ** 2
   143:         # Avoid division by zero
   144:         safe_T = np.where(np.abs(T_tilde) > 0.01, T_tilde, np.sign(T_tilde) * 0.01 + 1e-8)
   145:         pseudo = Y_tilde / safe_T
   146: 
   147:         # Clip extreme pseudo-outcomes
   148:         q = np.percentile(np.abs(pseudo), 95)
   149:         pseudo = np.clip(pseudo, -q, q)
   150: 
   151:         # Weighted regression for CATE
   152:         # Use sample_weight = T_tilde^2 to prioritize informative samples
   153:         self._cate_model = GradientBoostingRegressor(
   154:             n_estimators=200, max_depth=3, learning_rate=0.05,
   155:             min_samples_leaf=20, subsample=0.8, random_state=self._seed + 2,
   156:         )
   157:         self._cate_model.fit(X, pseudo, sample_weight=weights)
   158:         return self
   159: 
   160:     def predict(self, X):
   161:         return self._cate_model.predict(X)
   162: 
   163: 
   164: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

## Time Budget

You have **5 hours** of wall-clock time before submission, covering
everything you do here: reading the code, editing it, and any trial runs
you launch.

Good luck.
