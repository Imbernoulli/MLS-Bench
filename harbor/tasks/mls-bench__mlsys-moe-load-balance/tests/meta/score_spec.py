"""Score spec for mlsys-moe-load-balance.

Four metrics per MoE config (deepseek-v3, qwen3-moe, deepseek-v2, stress-skew):
  balance       — per-GPU load balance, higher better, bounded at 1.0
  balance_node  — per-node load balance, higher better, bounded at 1.0
  locality      — traffic-weighted node locality of expert replicas, higher
                  better, bounded at 1.0. Hierarchical placements that keep
                  every expert's replicas on a single node score ≈1.0; flat
                  placements that scatter replicas across all nodes score
                  ≈1/num_nodes. Captures the inter-node communication cost
                  pure load-balance metrics ignore.
  runtime_ms    — algorithm runtime (median over timing iters), lower better

Per-config score = weighted_mean of the four terms (equal weight). Including
both balance metrics and locality forces methods to balance load AND respect
node hierarchy: a flat scheme that ignores topology will saturate balance
but lose on locality, and vice versa. *_std columns are within-run variance
and ignored.

Task score = geometric mean across the four configs (three real-model
deployments plus the hidden stress-skew stress test).
"""
from mlsbench.scoring.dsl import *

# ---- per-config terms -------------------------------------------------------

term("balance_deepseek_v3",
    col("balance_deepseek-v3").higher().id()
    .bounded_power(bound=1.0))
term("balance_node_deepseek_v3",
    col("balance_node_deepseek-v3").higher().id()
    .bounded_power(bound=1.0))
term("locality_deepseek_v3",
    col("locality_deepseek-v3").higher().id()
    .bounded_power(bound=1.0))
term("runtime_ms_deepseek_v3",
    col("runtime_ms_deepseek-v3").lower().id()
    .sigmoid())

term("balance_qwen3_moe",
    col("balance_qwen3-moe").higher().id()
    .bounded_power(bound=1.0))
term("balance_node_qwen3_moe",
    col("balance_node_qwen3-moe").higher().id()
    .bounded_power(bound=1.0))
term("locality_qwen3_moe",
    col("locality_qwen3-moe").higher().id()
    .bounded_power(bound=1.0))
term("runtime_ms_qwen3_moe",
    col("runtime_ms_qwen3-moe").lower().id()
    .sigmoid())

term("balance_deepseek_v2",
    col("balance_deepseek-v2").higher().id()
    .bounded_power(bound=1.0))
term("balance_node_deepseek_v2",
    col("balance_node_deepseek-v2").higher().id()
    .bounded_power(bound=1.0))
term("locality_deepseek_v2",
    col("locality_deepseek-v2").higher().id()
    .bounded_power(bound=1.0))
term("runtime_ms_deepseek_v2",
    col("runtime_ms_deepseek-v2").lower().id()
    .sigmoid())

term("balance_stress_skew",
    col("balance_stress-skew").higher().id()
    .bounded_power(bound=1.0))
term("balance_node_stress_skew",
    col("balance_node_stress-skew").higher().id()
    .bounded_power(bound=1.0))
term("locality_stress_skew",
    col("locality_stress-skew").higher().id()
    .bounded_power(bound=1.0))
term("runtime_ms_stress_skew",
    col("runtime_ms_stress-skew").lower().id()
    .sigmoid())

# ---- per-config combined scores --------------------------------------------

setting("deepseek-v3", weighted_mean(
    ("balance_deepseek_v3", 1.0),
    ("balance_node_deepseek_v3", 1.0),
    ("locality_deepseek_v3", 1.0),
    ("runtime_ms_deepseek_v3", 1.0),
))
setting("qwen3-moe", weighted_mean(
    ("balance_qwen3_moe", 1.0),
    ("balance_node_qwen3_moe", 1.0),
    ("locality_qwen3_moe", 1.0),
    ("runtime_ms_qwen3_moe", 1.0),
))
setting("deepseek-v2", weighted_mean(
    ("balance_deepseek_v2", 1.0),
    ("balance_node_deepseek_v2", 1.0),
    ("locality_deepseek_v2", 1.0),
    ("runtime_ms_deepseek_v2", 1.0),
))
setting("stress-skew", weighted_mean(
    ("balance_stress_skew", 1.0),
    ("balance_node_stress_skew", 1.0),
    ("locality_stress_skew", 1.0),
    ("runtime_ms_stress_skew", 1.0),
))

task(gmean("deepseek-v3", "qwen3-moe", "deepseek-v2", "stress-skew"))
