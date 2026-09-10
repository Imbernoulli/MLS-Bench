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
