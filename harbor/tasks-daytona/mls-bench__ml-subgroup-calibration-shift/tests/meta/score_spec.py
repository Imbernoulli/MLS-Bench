"""Score spec for ml-subgroup-calibration-shift."""
from mlsbench.scoring.dsl import *

# worst_group_ece: lower is better (calibration error, 0 is perfect)
# brier: lower is better (0 is perfect)
# subgroup_auroc: higher is better, bounded at 1.0
# max_subgroup_gap: lower is better (max - min per-subgroup ECE)
# auroc_retention / slope_retention: discrimination kept relative to the
#   fixed base classifier (constraints, see below)
# refs from best baseline means per dataset


def _add_setting(label):
    term(f"worst_group_ece_{label}",
        col(f"worst_group_ece_{label}").lower().id()
        .bounded_power(bound=0.0))
    term(f"brier_{label}",
        col(f"brier_{label}").lower().id()
        .bounded_power(bound=0.0))
    term(f"subgroup_auroc_{label}",
        col(f"subgroup_auroc_{label}").higher().id()
        .bounded_power(bound=1.0))
    term(f"max_subgroup_gap_{label}",
        col(f"max_subgroup_gap_{label}").lower().id()
        .bounded_power(bound=0.0))

    # The shifted test split keeps the calibration set's class balance, so a
    # constant base-rate predictor has no information yet beats every
    # baseline on all three calibration terms (Brier ~p(1-p) against
    # 0.32-0.43). A calibrator must therefore keep the base classifier's
    # discrimination. auroc_retention is the subgroup AUROC over the base
    # classifier's own; any monotonic map keeps it near 1 (baselines
    # 0.954-1.000), and a constant, random or per-group constant map gets
    # 0.52-0.67. slope_retention is Tjur's discrimination slope
    # E[p|y=1] - E[p|y=0] over the base classifier's; it catches a map that
    # keeps the ranking but compresses every probability onto the base rate
    # (0.0001). Baselines get 0.98-1.47, so both penalties are 1.0 for them.
    # Shrinking toward the base rate is still allowed while the calibrated
    # probabilities keep half the base slope.
    term(f"auroc_retention_{label}",
        penalty_lower(col(f"auroc_retention_{label}").higher().id(),
                      target=0.9, sharpness=10.0))
    term(f"slope_retention_{label}",
        penalty_lower(col(f"slope_retention_{label}").higher().id(),
                      target=0.5, sharpness=10.0))

    # subgroup_auroc depends only on the fixed base classifier and is
    # invariant across monotonic post-hoc calibrations, so it does not
    # discriminate between methods. Kept as a diagnostic term but excluded
    # from the scored objective; auroc_retention constrains it instead.
    setting(label, weighted_mean(
        (f"worst_group_ece_{label}", 1.0),
        (f"brier_{label}", 1.0),
        (f"max_subgroup_gap_{label}", 1.0),
    ), constraints=[f"auroc_retention_{label}", f"slope_retention_{label}"])


for _label in ("adult", "compas", "law_school"):
    _add_setting(_label)

task(gmean("adult", "compas", "law_school"))
