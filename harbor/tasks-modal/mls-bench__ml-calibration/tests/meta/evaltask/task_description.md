# Probability Calibration Method Design

## Research Question
Design a novel post-hoc probability calibration method that maps a classifier's raw confidence estimates into well-calibrated probabilities. The base classifier and train/calibration/test splits are fixed; the contribution is the *calibration mapping itself*, learned only from a held-out calibration set.

## Background
A well-calibrated model satisfies: among all predictions where the model outputs probability p for a class, the empirical fraction that are correct is approximately p. Modern neural networks, GBMs, RFs, and SVMs are routinely miscalibrated.

Reference baselines:
- **Platt scaling** — Platt, 1999. Fit a sigmoid `1 / (1 + exp(a*x + b))` on classifier scores via maximum likelihood. Designed for SVM margins.
- **Isotonic regression** — Zadrozny & Elkan, 2002. Non-parametric monotonic mapping; can overfit on small calibration sets.
- **Temperature scaling** — Guo, Pleiss, Sun, Weinberger, ICML 2017 ([arXiv:1706.04599](https://arxiv.org/abs/1706.04599)). Single scalar temperature `T` divides logits before softmax; fit by minimizing NLL on calibration set. Preserves accuracy (argmax invariant).

## Implementation Contract
Implement `CalibrationMethod` in `custom_calibration.py`:

```python
class CalibrationMethod(BaseEstimator):
    def fit(self, probs, labels):
        # probs: (n,) for binary (positive-class probability)
        #        or (n, C) for multiclass (rows sum to 1)
        # labels: (n,) integer class labels
        return self

    def predict_proba(self, probs):
        # Returns calibrated probabilities of the same shape as input.
        return calibrated_probs
```

Available imports: `numpy`, `scipy` (`optimize`, `interpolate`, `special`), `sklearn`. The output must remain a valid probability distribution (non-negative, sums to 1 for multiclass).

## Fixed Pipeline & Evaluation
Four classifier-dataset combinations:
- **Random Forest on MNIST** (10-class).
- **MLP on Fashion-MNIST** (10-class).
- **GBM on Madelon** (binary).
- **SVM on Breast Cancer** (binary).

Metrics (all lower is better):
- **ECE (Expected Calibration Error)** — weighted mean of `|accuracy − confidence|` across probability bins.
- **Brier score** — mean squared error between predicted probability vector and one-hot label.
- **NLL (Negative Log-Likelihood)** — cross-entropy between predicted probabilities and true labels.
