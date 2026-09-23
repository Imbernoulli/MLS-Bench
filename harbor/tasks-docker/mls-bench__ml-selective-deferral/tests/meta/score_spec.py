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

    # The 0.80 coverage is a hard budget, enforced multiplicatively. As one
    # term of the mean it was not: both risk terms are measured on accepted
    # samples only, so accepting just the most confident 20% gained more on
    # risk than it lost on coverage and outscored every correct method.
    # All baselines land at 0.77-0.81, inside [0.75, 0.85], so the penalty is
    # 1.0 for them; coverage 0.20 gets x0.004 and 1.00 gets x0.22.
    term(f"coverage_min_{label}",
        penalty_lower(col(f"coverage_at80_{label}").higher().id(),
                      target=0.75, sharpness=10.0))
    term(f"coverage_max_{label}",
        penalty_upper(col(f"coverage_at80_{label}").higher().id(),
                      target=0.85, sharpness=10.0))

    setting(label, weighted_mean(
        (f"selective_risk_at80_{label}", 1.0),
        (f"coverage_at80_{label}", 1.0),
        (f"worst_group_selective_risk_{label}", 1.0),
        (f"deferral_rate_gap_{label}", 1.0),
        (f"auroc_{label}", 1.0),
    ), constraints=[f"coverage_min_{label}", f"coverage_max_{label}"])


for _label in ("adult", "compas", "law_school"):
    _add_setting(_label)

task(gmean("adult", "compas", "law_school"))
