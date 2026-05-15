"""Score spec for ml-subgroup-calibration-shift."""
from mlsbench.scoring.dsl import *

# worst_group_ece: lower is better (calibration error, 0 is perfect)
# brier: lower is better (0 is perfect)
# subgroup_auroc: higher is better, bounded at 1.0
# max_subgroup_gap: lower is better (fairness — smaller gap between subgroups)
# refs from best baseline means per dataset

term("worst_group_ece_adult",
    col("worst_group_ece_adult").lower().id()
    .bounded_power(bound=0.0))

term("brier_adult",
    col("brier_adult").lower().id()
    .bounded_power(bound=0.0))

term("subgroup_auroc_adult",
    col("subgroup_auroc_adult").higher().id()
    .bounded_power(bound=1.0))

term("max_subgroup_gap_adult",
    col("max_subgroup_gap_adult").lower().id()
    .bounded_power(bound=0.0))

term("worst_group_ece_compas",
    col("worst_group_ece_compas").lower().id()
    .bounded_power(bound=0.0))

term("brier_compas",
    col("brier_compas").lower().id()
    .bounded_power(bound=0.0))

term("subgroup_auroc_compas",
    col("subgroup_auroc_compas").higher().id()
    .bounded_power(bound=1.0))

term("max_subgroup_gap_compas",
    col("max_subgroup_gap_compas").lower().id()
    .bounded_power(bound=0.0))

term("worst_group_ece_law_school",
    col("worst_group_ece_law_school").lower().id()
    .bounded_power(bound=0.0))

term("brier_law_school",
    col("brier_law_school").lower().id()
    .bounded_power(bound=0.0))

term("subgroup_auroc_law_school",
    col("subgroup_auroc_law_school").higher().id()
    .bounded_power(bound=1.0))

term("max_subgroup_gap_law_school",
    col("max_subgroup_gap_law_school").lower().id()
    .bounded_power(bound=0.0))

# subgroup_auroc depends only on the fixed base classifier and is invariant
# across monotonic post-hoc calibrations, so it does not discriminate between
# methods. Kept as a diagnostic term but excluded from the scored setting.
setting("adult", weighted_mean(
    ("worst_group_ece_adult", 1.0),
    ("brier_adult", 1.0),
    ("max_subgroup_gap_adult", 1.0),
))
setting("compas", weighted_mean(
    ("worst_group_ece_compas", 1.0),
    ("brier_compas", 1.0),
    ("max_subgroup_gap_compas", 1.0),
))
setting("law_school", weighted_mean(
    ("worst_group_ece_law_school", 1.0),
    ("brier_law_school", 1.0),
    ("max_subgroup_gap_law_school", 1.0),
))

task(gmean("adult", "compas", "law_school"))
