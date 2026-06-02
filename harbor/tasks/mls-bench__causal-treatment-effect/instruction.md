# MLS-Bench: causal-treatment-effect

# Causal Treatment Effect Estimation

## Research Question
Design a novel estimator for **Conditional Average Treatment Effects (CATE)**
from observational data that is accurate, robust to confounding, and
generalizes across data-generating processes.

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

Classical approaches include single-model learners, two-model learners, and
propensity reweighting. Modern methods use orthogonalization or debiasing
for better convergence rates.

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
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be rejected.

- `scikit-learn/custom_cate.py`
- editable lines **344–416**

## Readable Context

### `scikit-learn/custom_cate.py`  [EDITABLE — lines 344–416 only]

```python
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
   374:         """Initialize the CATE estimator."""
   375:         pass
   376:
   377:     def fit(self, X, T, Y):
   378:         """Fit the estimator on observational data.
   379:
   380:         Args:
   381:             X: (n, p) numpy array of covariates
   382:             T: (n,) numpy array of binary treatment indicators (0 or 1)
   383:             Y: (n,) numpy array of observed outcomes
   384:
   385:         Returns:
   386:             self
   387:         """
   388:         n, p = X.shape
   389:         XT = np.column_stack([X, T.reshape(-1, 1)])
   390:         self._model = Ridge(alpha=1.0)
   391:         self._model.fit(XT, Y)
   392:         return self
   393:
   394:     def predict(self, X):
   395:         """Predict CATE for given covariates.
   396:
   397:         Args:
   398:             X: (n, p) numpy array of covariates
   399:
   400:         Returns:
   401:             tau_hat: (n,) numpy array of estimated treatment effects
   402:         """
   403:         n = X.shape[0]
   404:         X1 = np.column_stack([X, np.ones((n, 1))])
   405:         X0 = np.column_stack([X, np.zeros((n, 1))])
   406:         return self._model.predict(X1) - self._model.predict(X0)
```

## Tips

- Keep the function/class signatures of the editable regions identical;
  they are imported by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- Aim for an *algorithmic* contribution — many hyperparameters are locked
  outside the editable surface anyway.

Good luck.
