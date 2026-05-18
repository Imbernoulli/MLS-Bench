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

## Evaluation
Evaluation uses three task-local synthetic benchmarks with known ground-truth
treatment effects. These are inspired by common causal-inference benchmark
families, but they are **not** the official IHDP, Jobs/LaLonde, or ACIC
datasets/settings:

| Label         | Inspired by   | n    | p   | Notes                                |
|---------------|---------------|------|-----|--------------------------------------|
| ihdp_synth    | IHDP-like     | 747  | 25  | Nonlinear effects                    |
| jobs_synth    | Jobs/LaLonde  | 2000 | 10  | Economic outcomes                    |
| acic_synth    | ACIC-like     | 4000 | 50  | High-dimensional complex confounding |

Each dataset is evaluated with 5-fold cross-fitting over 10 repetitions with
different random seeds, so the estimator should be stable across train/test
splits rather than tuned to one realization.

Metrics (both lower is better):
- **PEHE**: Precision in Estimation of Heterogeneous Effects =
  `sqrt(mean((tau_hat - tau_true)^2))`.
- **ATE error**: `|mean(tau_hat) - ATE_true|`.

Valid contributions may combine outcome modeling, propensity modeling,
orthogonalization, weighting, residualization, forests, neural models, or other
modular CATE ideas, as long as they address confounding and treatment-effect
heterogeneity.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `scikit-learn/custom_cate.py`
- editable lines **344–416**




## Readable Context


### `scikit-learn/custom_cate.py`  [EDITABLE — lines 344–416 only]

```python
     1: # Custom CATE Estimator for MLS-Bench
     2: #
     3: # EDITABLE section: CATEEstimator class (the treatment effect estimator).
     4: # FIXED sections: everything else (data generation, evaluation, CLI).
     5: #
     6: # Research question: Design a novel estimator for Conditional Average Treatment
     7: # Effects (CATE) across explicitly synthetic observational DGP families.
     8: 
     9: import os
    10: import argparse
    11: import json
    12: import time
    13: import warnings
    14: from abc import ABC, abstractmethod
    15: 
    16: import numpy as np
    17: from scipy import stats
    18: from sklearn.model_selection import KFold, cross_val_predict
    19: from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
    20: from sklearn.ensemble import (
    21:     RandomForestRegressor,
    22:     RandomForestClassifier,
    23:     GradientBoostingRegressor,
    24:     GradientBoostingClassifier,
    25: )
    26: from sklearn.tree import DecisionTreeRegressor
    27: from sklearn.neural_network import MLPRegressor, MLPClassifier
    28: from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    29: from sklearn.pipeline import Pipeline
    30: from sklearn.base import clone
    31: 
    32: warnings.filterwarnings("ignore")
    33: 
    34: 
    35: # =====================================================================
    36: # FIXED: Data Generating Processes (synthetic benchmark families)
    37: # =====================================================================
    38: 
    39: def generate_ihdp(n=747, p=25, seed=42):
    40:     """Task-local synthetic IHDP-inspired DGP.
    41: 
    42:     This is not the official IHDP benchmark. It preserves the small-sample,
    43:     mixed-feature, nonlinear-effect flavor of IHDP-style CATE evaluations while
    44:     using fully synthetic covariates, treatment assignment, and outcomes.
    45: 
    46:     Features strong confounding: propensity depends on the same variables
    47:     that drive heterogeneous treatment effects, with nonlinear interactions.
    48: 
    49:     Returns:
    50:         X: (n, p) covariate matrix
    51:         T: (n,) binary treatment indicator
    52:         Y: (n,) observed outcomes
    53:         tau: (n,) true individual treatment effects (CATE)
    54:         ate: scalar true average treatment effect
    55:     """
    56:     rng = np.random.RandomState(seed)
    57: 
    58:     # Covariates: mix of continuous and binary
    59:     X_cont = rng.randn(n, 6)
    60:     X_bin = rng.binomial(1, 0.5, size=(n, p - 6)).astype(float)
    61:     X = np.hstack([X_cont, X_bin])
    62: 
    63:     # Nonlinear propensity score with interactions (strong confounding)
    64:     logit_e = (
    65:         0.5 * X[:, 0]
    66:         - 0.3 * X[:, 1]
    67:         + 0.2 * X[:, 2]
    68:         + 0.15 * X[:, 0] * X[:, 1]       # interaction
    69:         - 0.2 * X[:, 2] ** 2              # quadratic
    70:         + 0.25 * X[:, 3] * X[:, 5]        # confounders shared with tau
    71:         + 0.1 * np.sin(X[:, 4] * np.pi)   # nonlinear
    72:     )
    73:     e = 1.0 / (1.0 + np.exp(-logit_e))
    74:     e = np.clip(e, 0.05, 0.95)
    75:     T = rng.binomial(1, e)
    76: 
    77:     # Response surfaces (complex nonlinear)
    78:     mu0 = (
    79:         np.exp(0.8 * X[:, 0] + 0.5 * X[:, 1])
    80:         + X[:, 2] * X[:, 3]
    81:         + 0.5 * X[:, 4]
    82:         + 0.3 * X[:, 0] * X[:, 2]
    83:         + 0.2 * np.cos(X[:, 5] * np.pi)
    84:         + rng.randn(n) * 0.5
    85:     )
    86:     # Heterogeneous treatment effect with confounder-dependent terms
    87:     tau = (
    88:         1.0
    89:         + 0.5 * X[:, 0]
    90:         + 0.3 * X[:, 1] ** 2
    91:         - 0.4 * X[:, 2] * X[:, 5]         # interaction with confounder
    92:         + 0.5 * np.sin(X[:, 3] * np.pi)
    93:         + 0.3 * np.maximum(X[:, 4], 0)     # ReLU-like
    94:         - 0.2 * X[:, 0] * X[:, 3]          # cross-interaction
    95:         + 0.15 * X[:, 6]                   # binary covariate effect
    96:     )
    97:     mu1 = mu0 + tau + rng.randn(n) * 0.5
    98: 
    99:     Y = T * mu1 + (1 - T) * mu0
   100:     ate = tau.mean()
   101: 
   102:     return X, T, Y, tau, ate
   103: 
   104: 
   105: def generate_jobs(n=2000, p=10, seed=42):
   106:     """Task-local synthetic Jobs/LaLonde-inspired DGP.
   107: 
   108:     This is not the official NSW/LaLonde or Jobs benchmark. It simulates a job
   109:     training program with observational assignment, earnings-like outcomes, and
   110:     heterogeneous effects over age, education, earnings, and demographics.
   111: 
   112:     Features nonlinear confounding and complex treatment effect heterogeneity
   113:     with interactions between confounders and effect modifiers.
   114: 
   115:     Returns:
   116:         X: (n, p) covariate matrix
   117:         T: (n,) binary treatment indicator
   118:         Y: (n,) observed outcomes (earnings)
   119:         tau: (n,) true individual treatment effects
   120:         ate: scalar true average treatment effect
   121:     """
   122:     rng = np.random.RandomState(seed)
   123: 
   124:     # Covariates simulating demographic features
   125:     age = rng.uniform(18, 55, n)
   126:     education = rng.uniform(8, 18, n)
   127:     prior_earnings = np.maximum(0, rng.normal(10000, 5000, n))
   128:     married = rng.binomial(1, 0.4, n).astype(float)
   129:     black = rng.binomial(1, 0.3, n).astype(float)
   130:     hispanic = rng.binomial(1, 0.15, n).astype(float)
   131: 
   132:     # Additional covariates
   133:     extra = rng.randn(n, p - 6)
   134:     X = np.column_stack([age, education, prior_earnings, married, black, hispanic, extra])
   135: 
   136:     # Normalize for stability
   137:     X_scaled = (X - X.mean(0)) / (X.std(0) + 1e-8)
   138: 
   139:     # Nonlinear propensity with interactions (strong confounding)
   140:     logit_e = (
   141:         -0.3 * X_scaled[:, 0]
   142:         - 0.2 * X_scaled[:, 1]
   143:         - 0.4 * X_scaled[:, 2]
   144:         + 0.1 * X_scaled[:, 3]
   145:         + 0.25 * X_scaled[:, 0] * X_scaled[:, 1]   # age-education interaction
   146:         - 0.15 * X_scaled[:, 2] ** 2                # quadratic earnings effect
   147:         + 0.2 * X_scaled[:, 1] * X_scaled[:, 3]     # education-married interaction
   148:     )
   149:     e = 1.0 / (1.0 + np.exp(-logit_e))
   150:     e = np.clip(e, 0.05, 0.95)
   151:     T = rng.binomial(1, e)
   152: 
   153:     # Base outcome (earnings) with nonlinearities
   154:     mu0 = (
   155:         5000
   156:         + 200 * X_scaled[:, 0]
   157:         + 500 * X_scaled[:, 1]
   158:         + 0.3 * prior_earnings
   159:         + 1000 * married
   160:         + 300 * X_scaled[:, 0] * X_scaled[:, 1]     # age-education interaction
   161:         + 200 * np.maximum(X_scaled[:, 2], 0)        # ReLU on earnings
   162:         + 150 * X_scaled[:, 4] * X_scaled[:, 5]      # race interaction
   163:         + rng.randn(n) * 800
   164:     )
   165: 
   166:     # Complex heterogeneous treatment effect
   167:     tau = (
   168:         1500
   169:         + 300 * X_scaled[:, 1]                        # more education -> bigger effect
   170:         - 200 * X_scaled[:, 0]                        # younger -> bigger effect
   171:         + 250 * np.abs(X_scaled[:, 2])                # nonlinear prior earnings
   172:         + 100 * X_scaled[:, 3]
   173:         + 400 * np.sin(X_scaled[:, 0] * np.pi / 2)   # periodic age effect
   174:         - 200 * X_scaled[:, 1] * X_scaled[:, 2]       # education-earnings interaction
   175:         + 300 * np.maximum(X_scaled[:, 6], 0)          # ReLU on extra covariate
   176:         + 150 * X_scaled[:, 0] * X_scaled[:, 3]       # age-married interaction
   177:     )
   178: 
   179:     mu1 = mu0 + tau + rng.randn(n) * 500
   180:     Y = T * mu1 + (1 - T) * mu0
   181:     ate = tau.mean()
   182: 
   183:     return X, T, Y, tau, ate
   184: 
   185: 
   186: def generate_acic(n=4000, p=50, seed=42):
   187:     """Task-local synthetic ACIC-inspired DGP.
   188: 
   189:     This is not an official ACIC competition scenario. It uses synthetic
   190:     high-dimensional correlated covariates with complex nonlinear response
   191:     surfaces and strong confounding to test robustness to misspecification.
   192: 
   193:     Returns:
   194:         X: (n, p) covariate matrix
   195:         T: (n,) binary treatment indicator
   196:         Y: (n,) observed outcomes
   197:         tau: (n,) true individual treatment effects
   198:         ate: scalar true average treatment effect
   199:     """
   200:     rng = np.random.RandomState(seed)
   201: 
   202:     # High-dimensional covariates with correlations
   203:     mean = np.zeros(p)
   204:     # Block-diagonal correlation structure
   205:     cov = np.eye(p)
   206:     for i in range(0, p - 1, 2):
   207:         cov[i, i + 1] = 0.3
   208:         cov[i + 1, i] = 0.3
   209:     X = rng.multivariate_normal(mean, cov, n)
   210: 
   211:     # Complex propensity model (strong confounding)
   212:     logit_e = (
   213:         0.4 * X[:, 0]
   214:         + 0.3 * X[:, 1]
   215:         - 0.2 * X[:, 2]
   216:         + 0.15 * X[:, 0] * X[:, 1]
   217:         - 0.1 * X[:, 3] ** 2
   218:         + 0.05 * np.sum(X[:, 4:10], axis=1)
   219:     )
   220:     e = 1.0 / (1.0 + np.exp(-logit_e))
   221:     e = np.clip(e, 0.05, 0.95)  # Overlap enforcement
   222:     T = rng.binomial(1, e)
   223: 
   224:     # Complex response surface (nonlinear, interactions)
   225:     mu0 = (
   226:         2.0 * np.sin(X[:, 0] * np.pi)
   227:         + X[:, 1] ** 2
   228:         + 0.5 * X[:, 2] * X[:, 3]
   229:         - 1.5 * np.abs(X[:, 4])
   230:         + 0.3 * np.sum(X[:, 5:15], axis=1)
   231:         + rng.randn(n) * 0.5
   232:     )
   233: 
   234:     # Complex heterogeneous treatment effect
   235:     tau = (
   236:         0.8
   237:         + 0.6 * X[:, 0]
   238:         - 0.3 * X[:, 1] ** 2
   239:         + 0.4 * np.maximum(X[:, 2], 0)
   240:         + 0.2 * X[:, 3] * X[:, 4]
   241:         - 0.15 * np.abs(X[:, 5])
   242:         + 0.1 * np.cos(X[:, 6] * np.pi)
   243:     )
   244: 
   245:     mu1 = mu0 + tau + rng.randn(n) * 0.3
   246:     Y = T * mu1 + (1 - T) * mu0
   247:     ate = tau.mean()
   248: 
   249:     return X, T, Y, tau, ate
   250: 
   251: 
   252: # =====================================================================
   253: # FIXED: Base class for CATE estimators
   254: # =====================================================================
   255: 
   256: class BaseCATEEstimator(ABC):
   257:     """Abstract base class for CATE estimators.
   258: 
   259:     All estimators must implement:
   260:         fit(X, T, Y) -> self
   261:         predict(X) -> tau_hat array of shape (n,)
   262:     """
   263: 
   264:     @abstractmethod
   265:     def fit(self, X, T, Y):
   266:         """Fit the estimator on observational data.
   267: 
   268:         Args:
   269:             X: (n, p) covariate matrix (numpy array)
   270:             T: (n,) binary treatment indicator (0 or 1)
   271:             Y: (n,) observed outcomes (continuous)
   272: 
   273:         Returns:
   274:             self
   275:         """
   276:         pass
   277: 
   278:     @abstractmethod
   279:     def predict(self, X):
   280:         """Predict CATE for given covariates.
   281: 
   282:         Args:
   283:             X: (n, p) covariate matrix
   284: 
   285:         Returns:
   286:             tau_hat: (n,) array of estimated treatment effects
   287:         """
   288:         pass
   289: 
   290: 
   291: # =====================================================================
   292: # FIXED: Evaluation utilities
   293: # =====================================================================
   294: 
   295: def compute_pehe(tau_true, tau_hat):
   296:     """Precision in Estimation of Heterogeneous Effects (lower is better).
   297: 
   298:     PEHE = sqrt(mean((tau_hat - tau_true)^2))
   299:     """
   300:     return np.sqrt(np.mean((tau_hat - tau_true) ** 2))
   301: 
   302: 
   303: def compute_ate_error(ate_true, tau_hat):
   304:     """Absolute error in ATE estimation (lower is better).
   305: 
   306:     ATE_error = |mean(tau_hat) - ATE_true|
   307:     """
   308:     return np.abs(np.mean(tau_hat) - ate_true)
   309: 
   310: 
   311: def evaluate_estimator(estimator, X, T, Y, tau, ate, n_splits=5, seed=42):
   312:     """Evaluate CATE estimator using cross-fitting.
   313: 
   314:     Performs K-fold cross-validation: fit on K-1 folds, predict on held-out fold.
   315:     Aggregates PEHE and ATE error across all held-out predictions.
   316: 
   317:     Returns:
   318:         dict with PEHE and ATE_error metrics
   319:     """
   320:     kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
   321:     tau_hat_all = np.zeros(len(X))
   322: 
   323:     for train_idx, test_idx in kf.split(X):
   324:         est = clone_estimator(estimator)
   325:         est.fit(X[train_idx], T[train_idx], Y[train_idx])
   326:         tau_hat_all[test_idx] = est.predict(X[test_idx])
   327: 
   328:     pehe = compute_pehe(tau, tau_hat_all)
   329:     ate_err = compute_ate_error(ate, tau_hat_all)
   330: 
   331:     return {"PEHE": pehe, "ATE_error": ate_err}
   332: 
   333: 
   334: def clone_estimator(estimator):
   335:     """Create a fresh copy of a CATE estimator."""
   336:     import copy
   337:     return copy.deepcopy(estimator)
   338: 
   339: 
   340: # =====================================================================
   341: # EDITABLE: Custom CATE Estimator (lines 344-416)
   342: # =====================================================================
   343: 
   344: class CATEEstimator(BaseCATEEstimator):
   345:     """Custom CATE (Conditional Average Treatment Effect) estimator.
   346: 
   347:     Design a novel estimator for heterogeneous treatment effects from
   348:     observational data. Your estimator receives covariates X, binary
   349:     treatment T, and outcomes Y, and must estimate tau(x) = E[Y(1)-Y(0)|X=x].
   350: 
   351:     Key challenges:
   352:     - Confounding: treatment assignment depends on covariates
   353:     - Heterogeneity: treatment effects vary across individuals
   354:     - Model misspecification: response surfaces may be nonlinear
   355:     - Finite-sample performance: must work well with limited data
   356: 
   357:     Approaches to consider:
   358:     - Meta-learners (S/T/X/R/DR-Learner frameworks)
   359:     - Propensity score methods (weighting, matching, doubly robust)
   360:     - Tree-based methods (causal forests, Bayesian additive regression trees)
   361:     - Representation learning for treatment effects
   362:     - Kernel methods or local regression for CATE
   363:     - Ensemble methods combining multiple estimators
   364: 
   365:     Available imports (in FIXED section above):
   366:         numpy, scipy.stats, sklearn (all submodules)
   367: 
   368:     Interface contract:
   369:         fit(X, T, Y) -> self
   370:         predict(X) -> tau_hat of shape (n,)
   371:     """
   372: 
   373:     def __init__(self):
   374:         """Initialize the CATE estimator.
   375: 
   376:         TODO: Set up any models, hyperparameters, or data structures needed.
   377:         """
   378:         pass
   379: 
   380:     def fit(self, X, T, Y):
   381:         """Fit the estimator on observational data.
   382: 
   383:         Args:
   384:             X: (n, p) numpy array of covariates
   385:             T: (n,) numpy array of binary treatment indicators (0 or 1)
   386:             Y: (n,) numpy array of observed outcomes
   387: 
   388:         Returns:
   389:             self
   390: 
   391:         TODO: Implement your CATE estimation algorithm.
   392:         The default implementation is a simple S-Learner placeholder.
   393:         """
   394:         # Placeholder: simple S-Learner (augmented features)
   395:         n, p = X.shape
   396:         XT = np.column_stack([X, T.reshape(-1, 1)])
   397:         self._model = Ridge(alpha=1.0)
   398:         self._model.fit(XT, Y)
   399:         return self
   400: 
   401:     def predict(self, X):
   402:         """Predict CATE for given covariates.
   403: 
   404:         Args:
   405:             X: (n, p) numpy array of covariates
   406: 
   407:         Returns:
   408:             tau_hat: (n,) numpy array of estimated treatment effects
   409: 
   410:         TODO: Implement prediction of individual treatment effects.
   411:         """
   412:         n = X.shape[0]
   413:         X1 = np.column_stack([X, np.ones((n, 1))])
   414:         X0 = np.column_stack([X, np.zeros((n, 1))])
   415:         return self._model.predict(X1) - self._model.predict(X0)
   416: 
   417: 
   418: # =====================================================================
   419: # FIXED: Main evaluation loop
   420: # =====================================================================
   421: 
   422: DATASETS = {
   423:     "ihdp_synth": generate_ihdp,
   424:     "jobs_synth": generate_jobs,
   425:     "acic_synth": generate_acic,
   426: }
   427: 
   428: 
   429: def main():
   430:     parser = argparse.ArgumentParser(description="CATE Estimation Benchmark")
   431:     parser.add_argument("--dataset", type=str, required=True,
   432:                         choices=list(DATASETS.keys()),
   433:                         help="Dataset to evaluate on")
   434:     parser.add_argument("--seed", type=int, default=42,
   435:                         help="Random seed for data generation and evaluation")
   436:     parser.add_argument("--n-splits", type=int, default=5,
   437:                         help="Number of cross-validation folds")
   438:     parser.add_argument("--n-reps", type=int, default=10,
   439:                         help="Number of repetitions with different data seeds")
   440:     args = parser.parse_args()
   441: 
   442:     print(f"Evaluating on {args.dataset} (seed={args.seed}, "
   443:           f"n_splits={args.n_splits}, n_reps={args.n_reps})", flush=True)
   444: 
   445:     pehe_values = []
   446:     ate_err_values = []
   447: 
   448:     for rep in range(args.n_reps):
   449:         data_seed = args.seed + rep * 1000
   450:         X, T, Y, tau, ate = DATASETS[args.dataset](seed=data_seed)
   451: 
   452:         estimator = CATEEstimator()
   453:         metrics = evaluate_estimator(
   454:             estimator, X, T, Y, tau, ate,
   455:             n_splits=args.n_splits, seed=data_seed,
   456:         )
   457: 
   458:         pehe_values.append(metrics["PEHE"])
   459:         ate_err_values.append(metrics["ATE_error"])
   460: 
   461:         print(f"TRAIN_METRICS rep={rep} PEHE={metrics['PEHE']:.6f} "
   462:               f"ATE_error={metrics['ATE_error']:.6f}", flush=True)
   463: 
   464:     # Aggregate across repetitions
   465:     mean_pehe = np.mean(pehe_values)
   466:     std_pehe = np.std(pehe_values)
   467:     mean_ate_err = np.mean(ate_err_values)
   468:     std_ate_err = np.std(ate_err_values)
   469: 
   470:     print(f"\n=== Results on {args.dataset} ===", flush=True)
   471:     print(f"PEHE: {mean_pehe:.6f} +/- {std_pehe:.6f}", flush=True)
   472:     print(f"ATE_error: {mean_ate_err:.6f} +/- {std_ate_err:.6f}", flush=True)
   473: 
   474:     # Final metrics for parser
   475:     print(f"TEST_METRICS PEHE={mean_pehe:.6f} ATE_error={mean_ate_err:.6f}", flush=True)
   476: 
   477: 
   478: if __name__ == "__main__":
   479:     main()
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **ihdp_synth** — wall-clock budget `01:00:00`, compute share `0.33`
- **jobs_synth** — wall-clock budget `01:00:00`, compute share `0.33`
- **acic_synth** — wall-clock budget `01:00:00`, compute share `0.33`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `s_learner` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_cate.py`:

```python
Lines 344–373:
   341: # EDITABLE: Custom CATE Estimator (lines 344-416)
   342: # =====================================================================
   343: 
   344: class CATEEstimator(BaseCATEEstimator):
   345:     """S-Learner: single model approach to CATE estimation.
   346: 
   347:     Fits a single outcome model mu(X, T) on the combined data, then
   348:     estimates CATE as mu(X, 1) - mu(X, 0).
   349:     Uses GradientBoostingRegressor as the base learner for flexibility.
   350:     """
   351: 
   352:     def __init__(self):
   353:         self._seed = int(os.environ.get("SEED", "42"))
   354:         self._model = GradientBoostingRegressor(
   355:             n_estimators=200,
   356:             max_depth=4,
   357:             learning_rate=0.1,
   358:             min_samples_leaf=20,
   359:             subsample=0.8,
   360:             random_state=self._seed,
   361:         )
   362: 
   363:     def fit(self, X, T, Y):
   364:         n, p = X.shape
   365:         XT = np.column_stack([X, T.reshape(-1, 1)])
   366:         self._model.fit(XT, Y)
   367:         return self
   368: 
   369:     def predict(self, X):
   370:         n = X.shape[0]
   371:         X1 = np.column_stack([X, np.ones((n, 1))])
   372:         X0 = np.column_stack([X, np.zeros((n, 1))])
   373:         return self._model.predict(X1) - self._model.predict(X0)
   374: 
   375: # =====================================================================
   376: # FIXED: Main evaluation loop
```

### `t_learner` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_cate.py`:

```python
Lines 344–379:
   341: # EDITABLE: Custom CATE Estimator (lines 344-416)
   342: # =====================================================================
   343: 
   344: class CATEEstimator(BaseCATEEstimator):
   345:     """T-Learner: two separate models for treated and control groups.
   346: 
   347:     Fits mu0(X) on control data and mu1(X) on treated data, then
   348:     estimates CATE as mu1(X) - mu0(X).
   349:     Uses GradientBoostingRegressor for both models.
   350:     """
   351: 
   352:     def __init__(self):
   353:         self._seed = int(os.environ.get("SEED", "42"))
   354:         self._model0 = GradientBoostingRegressor(
   355:             n_estimators=200,
   356:             max_depth=4,
   357:             learning_rate=0.1,
   358:             min_samples_leaf=20,
   359:             subsample=0.8,
   360:             random_state=self._seed,
   361:         )
   362:         self._model1 = GradientBoostingRegressor(
   363:             n_estimators=200,
   364:             max_depth=4,
   365:             learning_rate=0.1,
   366:             min_samples_leaf=20,
   367:             subsample=0.8,
   368:             random_state=self._seed + 1,
   369:         )
   370: 
   371:     def fit(self, X, T, Y):
   372:         mask0 = T == 0
   373:         mask1 = T == 1
   374:         self._model0.fit(X[mask0], Y[mask0])
   375:         self._model1.fit(X[mask1], Y[mask1])
   376:         return self
   377: 
   378:     def predict(self, X):
   379:         return self._model1.predict(X) - self._model0.predict(X)
   380: 
   381: # =====================================================================
   382: # FIXED: Main evaluation loop
```

### `ipw` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_cate.py`:

```python
Lines 344–383:
   341: # EDITABLE: Custom CATE Estimator (lines 344-416)
   342: # =====================================================================
   343: 
   344: class CATEEstimator(BaseCATEEstimator):
   345:     """IPW-based CATE estimator with propensity score weighting.
   346: 
   347:     1. Estimate propensity score e(X) = P(T=1|X) via logistic regression.
   348:     2. Construct IPW pseudo-outcomes: Y_ipw = T*Y/e(X) - (1-T)*Y/(1-e(X)).
   349:     3. Fit a regression model on X -> Y_ipw for CATE estimation.
   350: 
   351:     Clips propensity scores to [0.05, 0.95] for stability.
   352:     """
   353: 
   354:     def __init__(self):
   355:         self._seed = int(os.environ.get("SEED", "42"))
   356:         self._prop_model = GradientBoostingClassifier(
   357:             n_estimators=200, max_depth=3, learning_rate=0.1,
   358:             min_samples_leaf=20, subsample=0.8, random_state=self._seed,
   359:         )
   360:         self._outcome_model = GradientBoostingRegressor(
   361:             n_estimators=200,
   362:             max_depth=4,
   363:             learning_rate=0.1,
   364:             min_samples_leaf=20,
   365:             subsample=0.8,
   366:             random_state=self._seed + 1,
   367:         )
   368: 
   369:     def fit(self, X, T, Y):
   370:         # Estimate propensity scores
   371:         self._prop_model.fit(X, T)
   372:         e_hat = self._prop_model.predict_proba(X)[:, 1]
   373:         e_hat = np.clip(e_hat, 0.05, 0.95)
   374: 
   375:         # IPW pseudo-outcomes
   376:         pseudo_outcome = T * Y / e_hat - (1 - T) * Y / (1 - e_hat)
   377: 
   378:         # Fit outcome model on pseudo-outcomes
   379:         self._outcome_model.fit(X, pseudo_outcome)
   380:         return self
   381: 
   382:     def predict(self, X):
   383:         return self._outcome_model.predict(X)
   384: 
   385: # =====================================================================
   386: # FIXED: Main evaluation loop
```

### `causal_forest` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_cate.py`:

```python
Lines 344–425:
   341: # EDITABLE: Custom CATE Estimator (lines 344-416)
   342: # =====================================================================
   343: 
   344: class CATEEstimator(BaseCATEEstimator):
   345:     """Causal Forest (via econml CausalForestDML).
   346: 
   347:     Combines double machine learning (DML) for debiasing with
   348:     generalized random forests for heterogeneous effect estimation.
   349: 
   350:     Steps:
   351:     1. Cross-fit nuisance models: E[Y|X] and E[T|X]
   352:     2. Compute residuals: Y_res = Y - E[Y|X], T_res = T - E[T|X]
   353:     3. Fit a causal forest on residualized outcomes
   354: 
   355:     Falls back to a pure-sklearn implementation if econml is unavailable.
   356:     """
   357: 
   358:     def __init__(self):
   359:         self._seed = int(os.environ.get("SEED", "42"))
   360:         self._use_econml = True
   361:         try:
   362:             from econml.dml import CausalForestDML
   363:             self._cf = CausalForestDML(
   364:                 model_y=GradientBoostingRegressor(
   365:                     n_estimators=100, max_depth=3, learning_rate=0.1,
   366:                     min_samples_leaf=20, random_state=self._seed,
   367:                 ),
   368:                 model_t=GradientBoostingRegressor(
   369:                     n_estimators=100, max_depth=3, learning_rate=0.1,
   370:                     min_samples_leaf=20, random_state=self._seed + 1,
   371:                 ),
   372:                 n_estimators=500,
   373:                 min_samples_leaf=5,
   374:                 max_depth=None,
   375:                 honest=True,
   376:                 inference=False,
   377:                 random_state=self._seed + 2,
   378:                 cv=3,
   379:             )
   380:         except ImportError:
   381:             self._use_econml = False
   382:             # Fallback: manual residualization + random forest
   383:             self._model_y = GradientBoostingRegressor(
   384:                 n_estimators=200, max_depth=4, learning_rate=0.1,
   385:                 min_samples_leaf=20, random_state=self._seed,
   386:             )
   387:             self._model_t = GradientBoostingClassifier(
   388:                 n_estimators=200, max_depth=4, learning_rate=0.1,
   389:                 min_samples_leaf=20, random_state=self._seed + 1,
   390:             )
   391:             self._cate_model = RandomForestRegressor(
   392:                 n_estimators=500, min_samples_leaf=5,
   393:                 max_features="sqrt", random_state=self._seed + 2,
   394:             )
   395: 
   396:     def fit(self, X, T, Y):
   397:         if self._use_econml:
   398:             self._cf.fit(Y, T, X=X)
   399:         else:
   400:             # Manual DML: cross-fit residuals
   401:             kf = KFold(n_splits=3, shuffle=True, random_state=self._seed)
   402:             Y_res = np.zeros_like(Y)
   403:             T_res = np.zeros_like(T, dtype=float)
   404: 
   405:             for train_idx, val_idx in kf.split(X):
   406:                 my = clone(self._model_y).fit(X[train_idx], Y[train_idx])
   407:                 mt = clone(self._model_t).fit(X[train_idx], T[train_idx])
   408:                 Y_res[val_idx] = Y[val_idx] - my.predict(X[val_idx])
   409:                 T_res[val_idx] = T[val_idx] - mt.predict_proba(X[val_idx])[:, 1]
   410: 
   411:             # R-Learner-style pseudo-outcome with stable divisor + sample
   412:             # weighting so small |T_res| doesn't explode the fit.
   413:             safe_T = np.where(np.abs(T_res) > 0.01, T_res, np.sign(T_res) * 0.01 + 1e-8)
   414:             pseudo = Y_res / safe_T
   415:             weights = T_res ** 2
   416:             q = np.percentile(np.abs(pseudo), 95)
   417:             pseudo = np.clip(pseudo, -q, q)
   418:             self._cate_model.fit(X, pseudo, sample_weight=weights)
   419:         return self
   420: 
   421:     def predict(self, X):
   422:         if self._use_econml:
   423:             return self._cf.effect(X).flatten()
   424:         else:
   425:             return self._cate_model.predict(X)
   426: 
   427: # =====================================================================
   428: # FIXED: Main evaluation loop
```

### `dr_learner` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_cate.py`:

```python
Lines 344–433:
   341: # EDITABLE: Custom CATE Estimator (lines 344-416)
   342: # =====================================================================
   343: 
   344: class CATEEstimator(BaseCATEEstimator):
   345:     """DR-Learner: Doubly Robust CATE estimation.
   346: 
   347:     Steps:
   348:     1. Cross-fit nuisance models:
   349:        - mu0(X) = E[Y|X, T=0], mu1(X) = E[Y|X, T=1]  (outcome models)
   350:        - e(X) = P(T=1|X)  (propensity score)
   351:     2. Compute doubly-robust pseudo-outcomes:
   352:        phi(X) = mu1(X) - mu0(X)
   353:               + T*(Y - mu1(X))/e(X)
   354:               - (1-T)*(Y - mu0(X))/(1-e(X))
   355:     3. Fit a final CATE model on X -> phi(X)
   356:     """
   357: 
   358:     def __init__(self):
   359:         self._seed = int(os.environ.get("SEED", "42"))
   360: 
   361:     def _make_model_y(self):
   362:         return GradientBoostingRegressor(
   363:             n_estimators=200, max_depth=4, learning_rate=0.1,
   364:             min_samples_leaf=20, subsample=0.8, random_state=self._seed,
   365:         )
   366: 
   367:     def _make_model_t(self):
   368:         return GradientBoostingClassifier(
   369:             n_estimators=200, max_depth=3, learning_rate=0.1,
   370:             min_samples_leaf=20, subsample=0.8, random_state=self._seed + 1,
   371:         )
   372: 
   373:     def _make_cate_model(self):
   374:         return GradientBoostingRegressor(
   375:             n_estimators=200, max_depth=3, learning_rate=0.05,
   376:             min_samples_leaf=20, subsample=0.8, random_state=self._seed + 2,
   377:         )
   378: 
   379:     def fit(self, X, T, Y):
   380:         n = len(Y)
   381: 
   382:         # Cross-fit nuisance models
   383:         kf = KFold(n_splits=5, shuffle=True, random_state=self._seed)
   384:         mu0_hat = np.zeros(n)
   385:         mu1_hat = np.zeros(n)
   386:         e_hat = np.zeros(n)
   387: 
   388:         for train_idx, val_idx in kf.split(X):
   389:             # Outcome models (separate for T=0 and T=1)
   390:             mask0_train = T[train_idx] == 0
   391:             mask1_train = T[train_idx] == 1
   392: 
   393:             m0 = self._make_model_y()
   394:             m1 = self._make_model_y()
   395: 
   396:             if mask0_train.sum() > 5:
   397:                 m0.fit(X[train_idx[mask0_train]], Y[train_idx[mask0_train]])
   398:                 mu0_hat[val_idx] = m0.predict(X[val_idx])
   399:             else:
   400:                 mu0_hat[val_idx] = Y[T == 0].mean() if (T == 0).sum() > 0 else Y.mean()
   401: 
   402:             if mask1_train.sum() > 5:
   403:                 m1.fit(X[train_idx[mask1_train]], Y[train_idx[mask1_train]])
   404:                 mu1_hat[val_idx] = m1.predict(X[val_idx])
   405:             else:
   406:                 mu1_hat[val_idx] = Y[T == 1].mean() if (T == 1).sum() > 0 else Y.mean()
   407: 
   408:             # Propensity model
   409:             mt = self._make_model_t()
   410:             mt.fit(X[train_idx], T[train_idx])
   411:             e_hat[val_idx] = mt.predict_proba(X[val_idx])[:, 1]
   412: 
   413:         # Clip propensity scores
   414:         e_hat = np.clip(e_hat, 0.05, 0.95)
   415: 
   416:         # Doubly-robust pseudo-outcomes
   417:         pseudo = (
   418:             mu1_hat - mu0_hat
   419:             + T * (Y - mu1_hat) / e_hat
   420:             - (1 - T) * (Y - mu0_hat) / (1 - e_hat)
   421:         )
   422: 
   423:         # Clip extreme pseudo-outcomes
   424:         q = np.percentile(np.abs(pseudo), 97)
   425:         pseudo = np.clip(pseudo, -q, q)
   426: 
   427:         # Fit CATE model on pseudo-outcomes
   428:         self._cate_model = self._make_cate_model()
   429:         self._cate_model.fit(X, pseudo)
   430:         return self
   431: 
   432:     def predict(self, X):
   433:         return self._cate_model.predict(X)
   434: 
   435: # =====================================================================
   436: # FIXED: Main evaluation loop
```

### `r_learner` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_cate.py`:

```python
Lines 344–421:
   341: # EDITABLE: Custom CATE Estimator (lines 344-416)
   342: # =====================================================================
   343: 
   344: class CATEEstimator(BaseCATEEstimator):
   345:     """R-Learner: Robinson decomposition for CATE estimation.
   346: 
   347:     Based on the Robinson (1988) decomposition:
   348:         Y - m(X) = (T - e(X)) * tau(X) + epsilon
   349: 
   350:     Steps:
   351:     1. Cross-fit nuisance models:
   352:        - m(X) = E[Y|X]  (marginal outcome model)
   353:        - e(X) = P(T=1|X)  (propensity score)
   354:     2. Compute residuals: Y_tilde = Y - m(X), T_tilde = T - e(X)
   355:     3. Estimate tau(X) by minimizing: sum_i (Y_tilde_i - T_tilde_i * tau(X_i))^2
   356:        This is equivalent to weighted least squares with weight T_tilde^2.
   357:     """
   358: 
   359:     def __init__(self):
   360:         self._seed = int(os.environ.get("SEED", "42"))
   361: 
   362:     def _make_model_y(self):
   363:         return GradientBoostingRegressor(
   364:             n_estimators=200, max_depth=4, learning_rate=0.1,
   365:             min_samples_leaf=20, subsample=0.8, random_state=self._seed,
   366:         )
   367: 
   368:     def _make_model_t(self):
   369:         return GradientBoostingClassifier(
   370:             n_estimators=200, max_depth=3, learning_rate=0.1,
   371:             min_samples_leaf=20, subsample=0.8, random_state=self._seed + 1,
   372:         )
   373: 
   374:     def fit(self, X, T, Y):
   375:         n = len(Y)
   376: 
   377:         # Cross-fit nuisance models
   378:         kf = KFold(n_splits=5, shuffle=True, random_state=self._seed)
   379:         m_hat = np.zeros(n)
   380:         e_hat = np.zeros(n)
   381: 
   382:         for train_idx, val_idx in kf.split(X):
   383:             # Outcome model E[Y|X]
   384:             my = self._make_model_y()
   385:             my.fit(X[train_idx], Y[train_idx])
   386:             m_hat[val_idx] = my.predict(X[val_idx])
   387: 
   388:             # Propensity model P(T=1|X)
   389:             mt = self._make_model_t()
   390:             mt.fit(X[train_idx], T[train_idx])
   391:             e_hat[val_idx] = mt.predict_proba(X[val_idx])[:, 1]
   392: 
   393:         # Clip propensity scores
   394:         e_hat = np.clip(e_hat, 0.05, 0.95)
   395: 
   396:         # Residuals
   397:         Y_tilde = Y - m_hat
   398:         T_tilde = T - e_hat
   399: 
   400:         # R-Learner: pseudo-outcome = Y_tilde / T_tilde
   401:         # Weight = T_tilde^2 (higher weight where treatment variation is larger)
   402:         weights = T_tilde ** 2
   403:         # Avoid division by zero
   404:         safe_T = np.where(np.abs(T_tilde) > 0.01, T_tilde, np.sign(T_tilde) * 0.01 + 1e-8)
   405:         pseudo = Y_tilde / safe_T
   406: 
   407:         # Clip extreme pseudo-outcomes
   408:         q = np.percentile(np.abs(pseudo), 95)
   409:         pseudo = np.clip(pseudo, -q, q)
   410: 
   411:         # Weighted regression for CATE
   412:         # Use sample_weight = T_tilde^2 to prioritize informative samples
   413:         self._cate_model = GradientBoostingRegressor(
   414:             n_estimators=200, max_depth=3, learning_rate=0.05,
   415:             min_samples_leaf=20, subsample=0.8, random_state=self._seed + 2,
   416:         )
   417:         self._cate_model.fit(X, pseudo, sample_weight=weights)
   418:         return self
   419: 
   420:     def predict(self, X):
   421:         return self._cate_model.predict(X)
   422: 
   423: # =====================================================================
   424: # FIXED: Main evaluation loop
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
