# Subgroup Calibration Under Distribution Shift

## Research Question
Design a post-hoc calibration method that remains reliable across subgroups when the test distribution shifts relative to calibration. The base tabular classifier and the (intentionally shifted) train/calibration/test splits are fixed; the contribution is the *calibration mapping* applied to positive-class probabilities, optionally using subgroup IDs.

## Background
Many calibration methods look accurate on average while remaining unreliable for protected or operationally meaningful subgroups, especially once subgroup prevalence or score distribution shifts at test time. The challenge is to improve worst-subgroup calibration without overfitting small per-group calibration samples.

Reference baselines:
- **Temperature scaling** — Guo, Pleiss, Sun, Weinberger, ICML 2017 ([arXiv:1706.04599](https://arxiv.org/abs/1706.04599)). Single global scalar `T` divides logits before sigmoid; fit by NLL on the calibration set.
- **Isotonic regression** — Zadrozny & Elkan, KDD 2002. Non-parametric monotonic mapping of probabilities to empirical accuracies.
- **Beta calibration** — Kull, Silva Filho, Flach, AISTATS 2017 ([proceedings](https://proceedings.mlr.press/v54/kull17a.html)). Three-parameter beta-distribution mapping; subsumes sigmoids, inverse sigmoids, and identity.
- **Group-wise temperature scaling** — fit one temperature per subgroup (with optional shrinkage toward the global temperature for small groups).

## Implementation Contract
Modify `CalibrationMethod` in `scikit-learn/custom_subgroup_calibration.py`:

```python
class CalibrationMethod:
    def fit(self, probs, labels, groups=None):
        # probs: (n,) positive-class probabilities from the base classifier
        # labels: (n,) integer labels {0,1}
        # groups: (n,) integer subgroup IDs (may be None for group-agnostic methods)
        return self

    def predict_proba(self, probs, groups=None):
        # Returns (n,) calibrated positive-class probabilities in [0, 1].
        ...
```

The method must produce valid probabilities; `groups` may be ignored by group-agnostic methods.

## Fixed Pipeline & Evaluation
Datasets (cached high-stakes tabular data from AIF360):
- **Adult** — Census income; subgroup attributes: sex, race.
- **COMPAS** — ProPublica recidivism risk; subgroup attributes: race, sex.
- **Law School GPA** — admissions/outcome data binarized at the median first-year GPA; subgroup attributes: race, gender.

For each dataset the test split is intentionally shifted: a domain score selects the held-out test tail, and calibration is fit on the source region and evaluated on the shifted region. Subgroups come from protected attributes exposed by the dataset loaders.

Metrics:
- **`worst_group_ece`** — worst-subgroup expected calibration error (lower is better).
- **`brier`** — Brier score on test (lower is better).
- **`max_subgroup_gap`** — max over subgroups of `|accuracy − mean confidence|` (lower is better).
- **`subgroup_auroc`** — subgroup-level AUROC (higher is better; reported diagnostically).
