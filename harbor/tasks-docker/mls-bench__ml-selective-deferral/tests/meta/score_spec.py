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

# Sanity gates. The objective terms stop at 0 at the weakest baseline, so on
# their own they never charge for errors beyond it, while deferral_rate_gap
# pays up to 1.0 for any subgroup-balanced split -- including a blind or
# partly random one. Ungated, such non-solutions outscored every baseline.
#   auroc gate: the acceptance score must keep at least half of the weakest
#     baseline's AUROC margin over chance; chance level multiplies the
#     setting by e^-3.
#   risk gate (no regression): the selective risk may give up at most 15% of
#     the weakest baseline's error reduction over a random 80% (whose expected
#     risk is e, the fixed base model's no-deferral test error); beyond that
#     the setting is multiplied by e^-3 per further 10%. A subgroup-exact
#     confidence policy gives up <= 9%; half-random splits give up 23-44%.
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
    reduction = _NO_DEFERRAL_RISK[label] - _MAX_BASELINE_RISK[label]
    term(f"risk_gate_{label}",
        penalty_upper(col(f"selective_risk_at80_{label}").lower().id(),
                      target=_MAX_BASELINE_RISK[label] + 0.15 * reduction,
                      sharpness=3.0 / (0.1 * reduction)))

    setting(label, weighted_mean(
        (f"selective_risk_at80_{label}", 1.0),
        (f"worst_group_selective_risk_{label}", 1.0),
        (f"deferral_rate_gap_{label}", 1.0),
        (f"auroc_{label}", 1.0),
    ), constraints=[f"auroc_gate_{label}", f"risk_gate_{label}"])


for _label in ("adult", "compas", "law_school"):
    _add_setting(_label)

task(gmean("adult", "compas", "law_school"))
