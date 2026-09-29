# Selective Deferral Under Subgroup Shift

## Research Question
Design a selective-prediction / deferral policy for high-stakes tabular decisions. The base classifier and train/calibration/test pipeline are fixed; the contribution is the *acceptance rule* that decides — given a target coverage — which test examples are accepted and which are deferred to a downstream reviewer or backup process.

A good policy should:
- keep selective risk low at the target coverage,
- avoid concentrating deferrals on one subgroup,
- preserve AUROC of its acceptance score as a confidence-ranking signal,
- be simple enough to fit and apply offline on modest compute.

## Background
Selective prediction lets a fixed classifier abstain when its prediction is unreliable. Under subgroup shift, naive confidence thresholds can defer disproportionately on one group while leaving others under-covered.

Reference baselines (provided as `edit_ops`):
- **`confidence_thresholding`** — single global threshold on `max(softmax)` chosen so the overall acceptance rate matches the target coverage.
- **`conformal_abstention`** — split-conformal: compute non-conformity scores on calibration data and threshold so the achieved coverage is at least the target with high probability. Foundational reference: Vovk, Gammerman, Shafer, *Algorithmic Learning in a Random World*, 2005.
- **`learned_deferral`** — train a compact meta-classifier predicting whether the base model will be correct on a given example; defer those it predicts as "wrong". Related: Mozannar & Sontag, "Consistent Estimators for Learning to Defer", ICML 2020 ([arXiv:2006.01862](https://arxiv.org/abs/2006.01862)).
- **`groupwise_thresholding`** — per-subgroup thresholds, each tuned to hit the target coverage on its subgroup; a stronger fairness-aware reference.

## Implementation Contract
Implement `SelectivePolicy` in `scikit-learn/custom_selective.py`:

```python
class SelectivePolicy:
    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT,
                 random_state: int = 0):
        ...

    def fit(self, probs: np.ndarray, y_true: np.ndarray,
            groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        # probs: (n, n_classes) calibration-time base-model probabilities
        # y_true: (n,) calibration labels
        # groups: (n,) integer subgroup ids
        # X: optional raw features
        ...

    def acceptance_score(self, probs, groups, X=None) -> np.ndarray:
        # Higher score = more confident -> more likely to accept.
        ...

    def predict_accept(self, probs, groups, X=None) -> np.ndarray:
        # Boolean array: True = accept, False = defer.
        ...

    def calibration_summary(self) -> dict[str, float]:
        ...
```

You may implement a global threshold, a learned acceptance score, subgroup-specific thresholds, conformal mechanisms, or any compact policy fitting this interface. The base classifier and the train/calibration/test split are not editable.

## Fixed Pipeline & Evaluation
Datasets (cached high-stakes tabular data from AIF360):
- **Adult** — Census income prediction; subgroup attributes: sex, race.
- **COMPAS** — ProPublica recidivism risk; subgroup attributes: race, sex.
- **Law School GPA** — admissions/outcome data, binarized at the training-set median; subgroup attributes: race, gender.

Each dataset is split into train / calibration / test; the policy fits on calibration probabilities/labels/subgroups and is evaluated on test.

Every metric is computed at exactly 80% test coverage. The harness ranks the test examples by your policy's own decisions (examples that `predict_accept` accepts come first), then by `acceptance_score` (highest first), then by test-set order, and accepts exactly the top `round(0.8 * n)`. Accepting more or fewer than 80% therefore only changes which examples this ranking trims or fills; coverage itself is not scored.

Each dataset's score is also multiplied by two sanity gates. (1) The AUROC of `acceptance_score` must keep at least half of the weakest reference baseline's margin over 0.5; below that the multiplier decays exponentially, reaching e^-3 at chance level. AUROC is computed on the ranks of the scores, so ±inf scores are fine (e.g. -inf for "never accept"); a NaN score is an error. (2) No regression in selective risk: the accepted 80% may give up at most 15% of the weakest reference baseline's error reduction relative to the base model's no-deferral error (the expected error of a random 80%); beyond that the multiplier falls by e^-3 per further 10%. A low deferral gap therefore cannot be bought with a blind or partly random accept/defer split.

Metrics:
- **`selective_risk_at80`** — classification error on the accepted examples at exactly 80% coverage (lower is better).
- **`worst_group_selective_risk`** — worst-subgroup error among accepted examples (lower is better).
- **`deferral_rate_gap`** — max-subgroup deferral rate minus min-subgroup deferral rate (lower is better).
- **`auroc`** — AUROC of the acceptance score as a predictor of correctness (higher is better).
