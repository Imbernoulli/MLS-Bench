# MoE Expert Parallelism Load Balancing

## Research Question

Design an efficient expert placement algorithm for Mixture-of-Experts
(MoE) inference that assigns expert replicas to GPUs to minimize load
imbalance — at both the GPU and node level — while preserving inter-node
locality of replicas and keeping the rebalancing algorithm runtime low.

## Background

In MoE models (e.g., DeepSeek-V2/V3, Qwen3-MoE), different experts
receive different amounts of traffic depending on the input distribution.
During inference, experts are distributed across GPUs, and load imbalance
causes some GPUs to become bottlenecks. The Expert Parallelism Load
Balancer (EPLB), introduced in DeepSeek's open-source release
(`deepseek-ai/EPLB`), runs periodically to rebalance expert placement as
workload patterns change.

The standard three-stage hierarchical algorithm is:

1. Group-to-node packing: distribute expert groups across server nodes to
   balance inter-node load.
2. Expert replication: create additional replicas of popular (hot)
   experts within each node.
3. Replica-to-GPU packing: assign physical expert replicas to GPUs within
   each node.

The reference greedy bin-packing approach uses Python for-loops to find
optimal assignments, which is correct but slow. Vectorized tensor
operations can achieve equivalent balance quality with substantially
faster runtime, provided they preserve the node hierarchy.

## Task

Modify the editable section of `custom_eplb.py` to implement an expert
placement algorithm. You must implement:

- `balanced_packing(weight, num_packs)` — pack weighted items into
  balanced packs
- `replicate_experts(weight, num_phy)` — decide expert replication counts
  and assign physical IDs
- `rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)`
  — main entry point combining all three stages

## Interface

```python
def rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus):
    """
    Args:
        weight: [L, E] tensor — token load per expert per layer
        num_replicas: total physical expert slots (multiple of num_gpus)
        num_groups: number of expert groups (divisor of E)
        num_nodes: number of server nodes
        num_gpus: total GPUs (multiple of num_nodes)

    Returns:
        phy2log: [L, num_replicas] — logical expert ID for each physical slot
        log2phy: [L, E, max_rep] — physical IDs per expert (-1 = unused)
        logcnt: [L, E] — number of physical replicas per logical expert
    """
```

Constraints:

- `E % num_groups == 0`, `num_groups % num_nodes == 0`
- `num_gpus % num_nodes == 0`, `num_replicas % num_gpus == 0`
- Each GPU must receive exactly `num_replicas // num_gpus` physical
  experts
- Every logical expert must have at least one replica
- `logcnt.sum(-1)` must equal `num_replicas` for every layer

## Evaluation

Four MoE deployments derived from real architectures plus one stress
configuration:

| Config | E (experts) | G (groups) | N (nodes) | D (GPUs) | R (replicas) | zipf · skew |
|---|---|---|---|---|---|---|
| `deepseek-v3`  | 256 | 8  | 8  | 64  | 320 | 0.7 · 0.85 |
| `qwen3-moe`    | 128 | 8  | 4  | 32  | 160 | 0.5 · 0.70 |
| `deepseek-v2`  | 160 | 8  | 4  | 32  | 192 | 0.6 · 0.75 |
| `stress-skew`  | 256 | 32 | 16 | 128 | 384 | 1.0 · 0.95 |

`stress-skew` is a synthetic stress test: 16-node hierarchy is the
largest in the suite, the replication budget is tighter (1.5x rather than
2x), `groups_per_node = 2` makes Stage 1 group-to-node packing
non-trivial, and the workload follows a long-tail Zipf distribution.

Per configuration, four metrics are reported:

- `balance` — `mean_gpu_load / max_gpu_load`, averaged over layers and
  trials. Higher is better, capped at 1.0 (perfect per-GPU balance).
- `balance_node` — `mean_node_load / max_node_load`, the same ratio at
  node granularity. Higher is better, capped at 1.0.
- `locality` — traffic-weighted node locality of replicas. For each
  (layer, expert) pair the harness counts how many distinct nodes hold a
  replica; the score is `1 / nodes_per_expert` averaged over experts
  (weighted by per-layer expert traffic) and over layers. A hierarchical
  scheme that keeps every expert's replicas on a single node scores 1.0;
  a flat scheme that scatters replicas across all nodes uniformly scores
  `1 / num_nodes`. This metric directly captures the inter-node
  communication cost that pure load-balance metrics ignore.
- `runtime_ms` — median wall time of the placement algorithm over 20
  timed iterations, averaged across 10 workload trials. Lower is better.

The combined per-config score weights all four terms equally; the task
score is the geometric mean across the four configs. All three
balance/locality metrics are required: a flat scheme that scatters
replicas to maximize per-GPU balance will lose `locality`; a method that
co-locates replicas without addressing skew will lose `balance`.

## Reference baselines

### greedy
Greedy bin-packing using Python for-loops: the original EPLB reference
algorithm. Correct hierarchical placement but slow due to sequential
Python iteration.

### zigzag
Vectorized zigzag (snake) pattern (Cheng et al., 2025): items sorted by
weight are assigned to packs in alternating order (0,1,…,P-1,P-1,…,0),
interleaving heavy and light items. All three stages use this zigzag
tensor pattern; replication remains sequential. ~20 ms on medium configs.

### flat_zigzag
Flat (non-hierarchical) zigzag: skips the group-to-node Stage 1 entirely
and does a single global zigzag assignment of all replicas to all GPUs
directly. Faster than the hierarchical approach but may lose `locality`
in multi-node settings because it ignores inter-node topology.
