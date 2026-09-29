"""Score spec for mlsys-moe-load-balance.

Four metrics per MoE config (deepseek-v3, qwen3-moe, deepseek-v2, stress-skew),
all measured by the fixed harness in eplb/custom_eplb.py:
  balance       -- mean/max per-GPU load, higher better, bounded at 1.0
  balance_node  -- mean/max per-node load, higher better, bounded at 1.0
  locality      -- traffic-weighted 1/nodes_per_expert, 1.0 when every
                   expert's replicas sit on a single node
  runtime_ms    -- wall time of rebalance_experts() per call

Load balance is the objective. Each balance term is linear between the
`static` baseline (a load-oblivious placement: contiguous expert blocks per
node, round-robin spare replicas) at 0 and perfect balance (1.0) at 1, i.e.
the fraction of the naive placement's imbalance that a method removes.
(ref=1.0 puts the reference at the bound, which makes the power curve
linear, gamma = 1.) The floor is the worst baseline, which is `static`.

Locality and runtime are constraints that multiply the per-config score:
  locality   -- target 1.0 (hierarchical placement); the score decays as
                exp(-3.0 * (1 - locality)), e.g. x0.77 at locality 0.915.
  runtime_ms -- a 100 ms budget per call (EPLB reruns periodically, so any
                placement computed well within it is equally usable); the
                score decays as exp(-0.01 * (runtime_ms - 100)) above it.

Per-config score = mean(balance, balance_node) * locality_penalty *
runtime_penalty. Task score = geometric mean across the four configs (three
real-model deployments plus the hidden stress-skew stress test). *_std
columns are within-run variance and ignored.
"""
from mlsbench.scoring.dsl import *

CONFIGS = ["deepseek-v3", "qwen3-moe", "deepseek-v2", "stress-skew"]

for cfg in CONFIGS:
    slug = cfg.replace("-", "_")
    term(f"balance_{slug}",
        col(f"balance_{cfg}").higher().id()
        .bounded_power(bound=1.0, ref=const(1.0)))
    term(f"balance_node_{slug}",
        col(f"balance_node_{cfg}").higher().id()
        .bounded_power(bound=1.0, ref=const(1.0)))
    term(f"locality_{slug}",
        penalty_lower(col(f"locality_{cfg}").higher().id(),
                      target=1.0, sharpness=3.0))
    term(f"runtime_ms_{slug}",
        penalty_upper(col(f"runtime_ms_{cfg}").lower().id(),
                      target=100.0, sharpness=0.01))
    setting(cfg, weighted_mean(
        (f"balance_{slug}", 1.0),
        (f"balance_node_{slug}", 1.0),
    ), constraints=[f"locality_{slug}", f"runtime_ms_{slug}"])

task(gmean(*CONFIGS))

