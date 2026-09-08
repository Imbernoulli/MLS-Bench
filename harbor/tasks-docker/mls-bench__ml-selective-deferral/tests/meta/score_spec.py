"""Score spec for ml-selective-deferral."""
from mlsbench.scoring.dsl import *

# selective_risk_at80: lower is better (lower error on accepted samples)
# coverage_at80: higher is better (closer to the target acceptance budget)
# worst_group_selective_risk: lower is better (lower worst-group error)
# deferral_rate_gap: lower is better (smaller subgroup deferral gap)
# auroc: higher is better, bounded at 1.0


def _add_setting(label):
    term(f"selective_risk_at80_{label}",
        col(f"selective_risk_at80_{label}").lower().id()
        .bounded_power(bound=0.0))
    term(f"coverage_at80_{label}",
        col(f"coverage_at80_{label}").higher().id()
        .bounded_power(bound=1.0))
    term(f"worst_group_selective_risk_{label}",
        col(f"worst_group_selective_risk_{label}").lower().id()
        .bounded_power(bound=0.0))
    term(f"deferral_rate_gap_{label}",
        col(f"deferral_rate_gap_{label}").lower().id()
        .bounded_power(bound=0.0))
    term(f"auroc_{label}",
        col(f"auroc_{label}").higher().id()
        .bounded_power(bound=1.0))

    # coverage_at80 is included in scoring because broken implementations can
    # silently land far from the 0.80 target (e.g. a method that accepts only
    # 20% of samples gets an unearned tiny selective_risk on the 20% it kept).
    # All correct baselines hit ~0.795-0.811 so the coverage term has near-zero
    # spread between them; any agent that drifts to 0.20 / 1.00 takes a real hit.
    setting(label, weighted_mean(
        (f"selective_risk_at80_{label}", 1.0),
        (f"coverage_at80_{label}", 1.0),
        (f"worst_group_selective_risk_{label}", 1.0),
        (f"deferral_rate_gap_{label}", 1.0),
        (f"auroc_{label}", 1.0),
    ))


for _label in ("adult", "compas", "law_school"):
    _add_setting(_label)

task(gmean("adult", "compas", "law_school"))
