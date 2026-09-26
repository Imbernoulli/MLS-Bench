"""Score spec for ml-selective-deferral."""
from mlsbench.scoring.dsl import *

# Every metric is computed at exactly 80% test coverage: the harness accepts
# the top round(0.8 * n) test examples of the policy's own ranking (accepted
# first, then by acceptance_score, then by test order). A policy therefore
# cannot trade coverage for selective risk, and coverage_at80 is the same for
# every submission, so it is not scored.
# selective_risk_at80: lower is better (lower error on accepted samples)
# worst_group_selective_risk: lower is better (lower worst-group error)
# deferral_rate_gap: lower is better (smaller subgroup deferral gap)
# auroc: higher is better, bounded at 1.0

# Sanity gates. A policy that defers blindly (constant or random score, or a
# random accept/defer split) gets a near-zero deferral_rate_gap for free while
# the risk terms only fall to 0, so ungated such non-solutions scored like, or
# above, the reference baselines. Both the acceptance score and the accepted
# 80% must keep at least half of the weakest baseline's information over
# chance; a chance-level policy has its setting multiplied by e^-3.
#   auroc gate: AUROC >= 0.5 + 0.5 * (min baseline AUROC - 0.5)
#   risk gate:  selective risk <= e - 0.5 * (e - max baseline selective risk),
#               e = the fixed base model's no-deferral test error, which is
#               the expected selective risk of a random 80% subset
# Constants measured with seed 42 (leaderboard.csv baseline rows; e is
# 1238/9045, 342/1055, 1773/4469 misclassified test examples).
_MIN_BASELINE_AUROC = {"adult": 0.851883, "compas": 0.629582, "law_school": 0.614418}
_MAX_BASELINE_RISK = {"adult": 0.072968, "compas": 0.28436, "law_school": 0.377343}
_NO_DEFERRAL_RISK = {"adult": 0.136871, "compas": 0.324171, "law_school": 0.396733}


def _add_setting(label):
    term(f"selective_risk_at80_{label}",
        col(f"selective_risk_at80_{label}").lower().id()
        .bounded_power(bound=0.0))
    term(f"worst_group_selective_risk_{label}",
        col(f"worst_group_selective_risk_{label}").lower().id()
        .bounded_power(bound=0.0))
    term(f"deferral_rate_gap_{label}",
        col(f"deferral_rate_gap_{label}").lower().id()
        .bounded_power(bound=0.0))
    term(f"auroc_{label}",
        col(f"auroc_{label}").higher().id()
        .bounded_power(bound=1.0))

    auroc_margin = 0.5 * (_MIN_BASELINE_AUROC[label] - 0.5)
    term(f"auroc_gate_{label}",
        penalty_lower(col(f"auroc_{label}").higher().id(),
                      target=0.5 + auroc_margin, sharpness=3.0 / auroc_margin))
    risk_margin = 0.5 * (_NO_DEFERRAL_RISK[label] - _MAX_BASELINE_RISK[label])
    term(f"risk_gate_{label}",
        penalty_upper(col(f"selective_risk_at80_{label}").lower().id(),
                      target=_NO_DEFERRAL_RISK[label] - risk_margin,
                      sharpness=3.0 / risk_margin))

    setting(label, weighted_mean(
        (f"selective_risk_at80_{label}", 1.0),
        (f"worst_group_selective_risk_{label}", 1.0),
        (f"deferral_rate_gap_{label}", 1.0),
        (f"auroc_{label}", 1.0),
    ), constraints=[f"auroc_gate_{label}", f"risk_gate_{label}"])


for _label in ("adult", "compas", "law_school"):
    _add_setting(_label)

task(gmean("adult", "compas", "law_school"))
