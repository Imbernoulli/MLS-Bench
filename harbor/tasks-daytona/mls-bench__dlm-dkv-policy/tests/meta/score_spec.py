"""Score spec for dlm-dkv-policy.

DLM cache policies are ranked by quality-preserving cache efficiency, not by
raw final-task quality alone. Each workload uses:

- benchmark-native final task quality as a near-lossless soft constraint
- cache reuse as the main policy-efficiency term
- decode throughput as a secondary efficiency term

Once a policy satisfies the workload quality threshold, small benchmark noise
or one-point accuracy differences are not rewarded. Throughput is normalized
against the visible baseline envelope rather than a hard hardware range.
"""
from mlsbench.scoring.dsl import *


def add_setting(
    name: str,
    quality_target: float,
    reuse_ref: float = 0.75,
    quality_sharpness: float = 0.35,
) -> None:
    slug = name.replace("-", "_")
    term(
        f"quality_gate_{slug}",
        penalty_lower(
            col(f"final_score_{name}").higher().id(),
            target=quality_target,
            sharpness=quality_sharpness,
        ),
    )
    term(
        f"reuse_{slug}",
        col(f"reuse_ratio_{name}").higher().id().bounded_power(
            bound=1.0,
            ref=reuse_ref,
            ref_score=0.5,
        ),
    )
    term(
        f"speed_{slug}",
        col(f"tokens_per_s_{name}").higher().id().sigmoid(
            ref=bl_best(f"tokens_per_s_{name}"),
            ref_score=0.5,
        ),
    )
    setting(
        name,
        weighted_mean(
            (f"reuse_{slug}", 0.75),
            (f"speed_{slug}", 0.25),
        ),
        constraints=[f"quality_gate_{slug}"],
    )


# Quality gates are calibrated from the current upstream-aligned rerun, not
# stale pre-alignment rows. The math gate intentionally stays near the passing
# baseline cluster; d2Cache's fixed-setting MATH drop is penalized rather than
# hidden by lowering the threshold.
add_setting(
    "math",
    quality_target=35.0,
)
add_setting(
    "humaneval",
    quality_target=40.0,
)
add_setting(
    "lm-eval",
    quality_target=84.0,
)

task(gmean("math", "humaneval", "lm-eval"))
