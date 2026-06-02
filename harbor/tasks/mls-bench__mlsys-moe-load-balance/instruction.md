# MLS-Bench: mlsys-moe-load-balance

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


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/eplb/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `eplb/custom_eplb.py`
- editable lines **62–209**




## Readable Context


### `eplb/custom_eplb.py`  [EDITABLE — lines 62–209 only]

```python
     1: """
     2: MoE Expert Parallelism Load Balancing (EPLB) Benchmark
     3: ======================================================
     4: 
     5: Design an efficient expert placement algorithm for Mixture-of-Experts (MoE)
     6: inference that assigns expert replicas to GPUs to minimize load imbalance
     7: while keeping the rebalancing algorithm runtime low.
     8: 
     9: Metrics:
    10:   - balance: avg_tokens_per_gpu / max_tokens_per_gpu (higher is better, 1.0 = perfect)
    11:   - runtime_ms: time to run the placement algorithm (lower is better)
    12: 
    13: Available libraries: torch, numpy
    14: """
    15: 
    16: import time
    17: import os
    18: import sys
    19: import argparse
    20: from typing import Tuple
    21: 
    22: import torch
    23: import numpy as np
    24: 
    25: # ================================================================
    26: # MoE model configurations (benchmark profiles based on real architectures)
    27: # ================================================================
    28: CONFIGS = {
    29:     # DeepSeek-V3/R1-style: 256 routed experts, 8 expert groups, top-8 routing
    30:     # Deployment and replica counts are benchmark modeling assumptions.
    31:     "deepseek-v3": {
    32:         "num_layers": 61, "num_experts": 256, "num_groups": 8,
    33:         "num_nodes": 8, "num_gpus": 64, "num_replicas": 320,
    34:         "zipf_alpha": 0.7, "skew_ratio": 0.85,
    35:     },
    36:     # Qwen3-MoE-style: 128 experts, 8 groups, top-8 routing
    37:     # Deployment and replica counts are benchmark modeling assumptions.
    38:     "qwen3-moe": {
    39:         "num_layers": 48, "num_experts": 128, "num_groups": 8,
    40:         "num_nodes": 4, "num_gpus": 32, "num_replicas": 160,
    41:         "zipf_alpha": 0.5, "skew_ratio": 0.70,
    42:     },
    43:     # DeepSeek-V2-style: 160 routed experts, 8 expert groups, top-6 routing
    44:     # Deployment and replica counts are benchmark modeling assumptions.
    45:     "deepseek-v2": {
    46:         "num_layers": 60, "num_experts": 160, "num_groups": 8,
    47:         "num_nodes": 4, "num_gpus": 32, "num_replicas": 192,
    48:         "zipf_alpha": 0.6, "skew_ratio": 0.75,
    49:     },
    50:     # Stress: 16-node deployment with pathological long-tail traffic, large
    51:     # group hierarchy (groups_per_node=2 makes Stage 1 group-to-node
    52:     # packing non-trivial), and tight replication budget (1.5x). Hidden
    53:     # config designed to keep headroom above the real-model configs.
    54:     "stress-skew": {
    55:         "num_layers": 48, "num_experts": 256, "num_groups": 32,
    56:         "num_nodes": 16, "num_gpus": 128, "num_replicas": 384,
    57:         "zipf_alpha": 1.0, "skew_ratio": 0.95,
    58:     },
    59: }
    60: 
    61: # ================================================================
    62: # EDITABLE SECTION (lines 62-209)
    63: # Implement your expert placement algorithm below.
    64: # You may define helper functions and modify the three core functions.
    65: # ================================================================
    66: 
    67: def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    68:     """
    69:     Pack n weighted items into num_packs balanced packs.
    70: 
    71:     Args:
    72:         weight: [B, n] — weight of each item across B batches
    73:         num_packs: number of packs
    74: 
    75:     Returns:
    76:         pack_index: [B, n] — which pack (0..num_packs-1) each item goes to
    77:         rank_in_pack: [B, n] — position (0..items_per_pack-1) within the pack
    78: 
    79:     Constraint: each pack must contain exactly n // num_packs items.
    80:     """
    81:     B, n = weight.shape
    82:     assert n % num_packs == 0
    83:     items_per_pack = n // num_packs
    84: 
    85:     if items_per_pack == 1:
    86:         idx = torch.arange(n, dtype=torch.int64, device=weight.device).expand(B, -1)
    87:         return idx, torch.zeros_like(idx)
    88: 
    89:     sorted_idx = weight.float().sort(-1, descending=True).indices.cpu()
    90:     pack_index = torch.full((B, n), -1, dtype=torch.int64)
    91:     rank_in_pack = torch.full((B, n), -1, dtype=torch.int64)
    92:     for b in range(B):
    93:         loads = [0.0] * num_packs
    94:         counts = [0] * num_packs
    95:         for j in range(n):
    96:             item = sorted_idx[b, j].item()
    97:             best = min(
    98:                 (p for p in range(num_packs) if counts[p] < items_per_pack),
    99:                 key=lambda p: loads[p],
   100:             )
   101:             pack_index[b, item] = best
   102:             rank_in_pack[b, item] = counts[best]
   103:             loads[best] += weight[b, item].item()
   104:             counts[best] += 1
   105:     return pack_index, rank_in_pack
   106: 
   107: 
   108: def replicate_experts(
   109:     weight: torch.Tensor, num_phy: int
   110: ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
   111:     """
   112:     Replicate num_log logical experts into num_phy physical slots
   113:     to minimize the maximum per-replica load.
   114: 
   115:     Args:
   116:         weight: [B, num_log] — load per logical expert
   117:         num_phy: total physical expert slots (>= num_log)
   118: 
   119:     Returns:
   120:         phy2log: [B, num_phy] — logical expert ID for each physical slot
   121:         rank: [B, num_phy] — replica rank (0 = original, 1+ = copies)
   122:         logcnt: [B, num_log] — number of replicas per logical expert
   123:     """
   124:     B, num_log = weight.shape
   125:     device = weight.device
   126:     phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(B, 1)
   127:     rank = torch.zeros(B, num_phy, dtype=torch.int64, device=device)
   128:     logcnt = torch.ones(B, num_log, dtype=torch.int64, device=device)
   129:     idx_b = torch.arange(B, dtype=torch.int64, device=device)
   130:     for i in range(num_log, num_phy):
   131:         eff = weight / logcnt.float()
   132:         top = eff.argmax(dim=-1)
   133:         phy2log[:, i] = top
   134:         rank[:, i] = logcnt[idx_b, top]
   135:         logcnt[idx_b, top] += 1
   136:     return phy2log, rank, logcnt
   137: 
   138: 
   139: def rebalance_experts(
   140:     weight: torch.Tensor,
   141:     num_replicas: int,
   142:     num_groups: int,
   143:     num_nodes: int,
   144:     num_gpus: int,
   145: ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
   146:     """
   147:     Main entry point: hierarchical expert placement across GPUs.
   148: 
   149:     Stage 1: Pack expert groups across nodes (inter-node balancing)
   150:     Stage 2: Create replicas for popular experts within each node
   151:     Stage 3: Pack physical replicas to GPUs (intra-node balancing)
   152: 
   153:     Args:
   154:         weight: [L, E] — token load per expert per layer
   155:         num_replicas: total physical expert slots (multiple of num_gpus)
   156:         num_groups: number of expert groups
   157:         num_nodes: number of server nodes
   158:         num_gpus: total GPUs (multiple of num_nodes)
   159: 
   160:     Returns:
   161:         phy2log: [L, num_replicas] — logical expert for each physical slot
   162:         log2phy: [L, E, max_rep] — physical IDs per expert (-1 = unused)
   163:         logcnt: [L, E] — replica count per expert
   164:     """
   165:     L, E = weight.shape
   166:     weight = weight.float().cpu()
   167:     group_size = E // num_groups
   168:     gpus_per_node = num_gpus // num_nodes
   169:     phy_per_gpu = num_replicas // num_gpus
   170:     groups_per_node = num_groups // num_nodes
   171:     experts_per_node = E // num_nodes
   172:     replicas_per_node = num_replicas // num_nodes
   173: 
   174:     def inv(perm):
   175:         out = torch.empty_like(perm)
   176:         out.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
   177:         return out
   178: 
   179:     # Stage 1
   180:     tpg = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
   181:     gpi, grk = balanced_packing(tpg, num_nodes)
   182:     log2mlog = (((gpi * groups_per_node + grk) * group_size).unsqueeze(-1)
   183:                 + torch.arange(group_size)).flatten(-2)
   184:     mlog2log = inv(log2mlog)
   185: 
   186:     # Stage 2
   187:     tpm = weight.gather(-1, mlog2log).view(-1, experts_per_node)
   188:     p2m, prk, mcnt = replicate_experts(tpm, replicas_per_node)
   189: 
   190:     # Stage 3
   191:     tpp = (tpm / mcnt.float()).gather(-1, p2m)
   192:     pi, ri = balanced_packing(tpp, gpus_per_node)
   193:     p2pp = pi * phy_per_gpu + ri
   194:     pp2p = inv(p2pp)
   195: 
   196:     pp2m = p2m.gather(-1, pp2p)
   197:     pp2m = (pp2m.view(L, num_nodes, -1)
   198:             + torch.arange(0, E, experts_per_node).view(1, -1, 1)).flatten(-2)
   199:     pp2log = mlog2log.gather(-1, pp2m)
   200:     pprank = prk.gather(-1, pp2p).view(L, -1)
   201:     logcnt = mcnt.view(L, -1).gather(-1, log2mlog)
   202: 
   203:     mx = logcnt.max().item()
   204:     log2phy = torch.full((L, E, mx), -1, dtype=torch.int64)
   205:     log2phy.view(L, -1).scatter_(
   206:         -1, pp2log * mx + pprank,
   207:         torch.arange(num_replicas).expand(L, -1),
   208:     )
   209:     return pp2log, log2phy, logcnt
   210: 
   211: # ================================================================
   212: # FIXED SECTION — Workload generation and evaluation harness
   213: # Do not modify below this line
   214: # ================================================================
   215: 
   216: 
   217: def generate_workload(num_layers: int, num_experts: int, seed: int,
   218:                       zipf_alpha: float = 1.5, skew_ratio: float = 0.8) -> torch.Tensor:
   219:     """Generate synthetic MoE expert load distributions.
   220: 
   221:     Creates realistic workloads mixing uniform and skewed (Zipf) patterns
   222:     to simulate real expert utilization during inference.
   223:     """
   224:     rng = np.random.default_rng(seed)
   225:     weight = np.zeros((num_layers, num_experts), dtype=np.float32)
   226: 
   227:     for layer in range(num_layers):
   228:         layer_seed = seed * 1000 + layer
   229:         layer_rng = np.random.default_rng(layer_seed)
   230: 
   231:         # Base uniform load
   232:         base = layer_rng.uniform(100, 500, size=num_experts).astype(np.float32)
   233: 
   234:         # Zipf-like skew: some experts are much more popular
   235:         ranks = np.arange(1, num_experts + 1, dtype=np.float32)
   236:         zipf = 1.0 / np.power(ranks, zipf_alpha)
   237:         perm = layer_rng.permutation(num_experts)
   238:         zipf_load = zipf[perm] * layer_rng.uniform(5000, 20000)
   239: 
   240:         # Mix uniform and skewed
   241:         weight[layer] = base * (1 - skew_ratio) + zipf_load * skew_ratio
   242: 
   243:     return torch.from_numpy(weight)
   244: 
   245: 
   246: def compute_balance(
   247:     weight: torch.Tensor,
   248:     phy2log: torch.Tensor,
   249:     logcnt: torch.Tensor,
   250:     num_gpus: int,
   251:     num_nodes: int,
   252:     num_replicas: int,
   253: ) -> Tuple[float, float]:
   254:     """Compute load balance at GPU and node level.
   255: 
   256:     Returns
   257:     -------
   258:     balance_gpu : mean_gpu_load / max_gpu_load (higher better, 1.0 = perfect)
   259:     balance_node : mean_node_load / max_node_load (higher better)
   260: 
   261:     Both are reported because the placement algorithm must balance load at
   262:     BOTH levels. A globally-balanced flat scheme can score well on
   263:     balance_gpu while leaving inter-node load uneven (or vice versa). The
   264:     score combines them so methods must respect node hierarchy.
   265:     """
   266:     L = weight.shape[0]
   267:     phy_per_gpu = num_replicas // num_gpus
   268:     gpus_per_node = num_gpus // num_nodes
   269:     tokens_per_phy = (weight / logcnt.float()).gather(-1, phy2log)
   270:     tokens_per_gpu = tokens_per_phy.view(L, num_gpus, phy_per_gpu).sum(-1)
   271:     bal_gpu = tokens_per_gpu.mean(-1) / tokens_per_gpu.max(-1).values.clamp(min=1e-8)
   272:     tokens_per_node = tokens_per_gpu.view(L, num_nodes, gpus_per_node).sum(-1)
   273:     bal_node = tokens_per_node.mean(-1) / tokens_per_node.max(-1).values.clamp(min=1e-8)
   274:     return bal_gpu.mean().item(), bal_node.mean().item()
   275: 
   276: 
   277: def compute_locality(
   278:     weight: torch.Tensor,
   279:     phy2log: torch.Tensor,
   280:     num_gpus: int,
   281:     num_nodes: int,
   282:     num_replicas: int,
   283: ) -> float:
   284:     """Traffic-weighted node locality.
   285: 
   286:     For each (layer, logical expert), counts the number of distinct nodes
   287:     holding a replica and computes 1 / nodes_per_expert. Averaged over
   288:     experts weighted by traffic (per-layer expert weight) and over layers.
   289:     Returns a value in [1/num_nodes, 1.0]. Higher is better.
   290: 
   291:     Captures inter-node communication cost — when a token routes to expert
   292:     e, locality_e is the probability that a chosen replica is on the same
   293:     node the token already lives on (roughly: it is the expected fraction
   294:     of replica options reachable without crossing a node boundary).
   295: 
   296:     A hierarchical scheme that keeps every expert's replicas co-located on
   297:     one node scores 1.0. A flat scheme that scatters replicas across all
   298:     nodes uniformly scores 1 / num_nodes. Pure load-balance metrics cannot
   299:     distinguish these two regimes; locality does.
   300:     """
   301:     L, E = weight.shape
   302:     phy_per_gpu = num_replicas // num_gpus
   303:     gpus_per_node = num_gpus // num_nodes
   304:     phy_per_node = phy_per_gpu * gpus_per_node
   305: 
   306:     slot_node = (torch.arange(num_replicas, dtype=torch.int64) // phy_per_node)  # [R]
   307:     combo = phy2log.long() * num_nodes + slot_node.unsqueeze(0)  # [L, R]
   308:     presence = torch.zeros(L, E * num_nodes, dtype=torch.int64)
   309:     presence.scatter_(1, combo, torch.ones_like(combo, dtype=torch.int64))
   310:     nodes_per_expert = (presence.view(L, E, num_nodes) > 0).sum(-1).float().clamp(min=1.0)  # [L, E]
   311: 
   312:     w = weight.float()
   313:     locality = (w / nodes_per_expert).sum(-1) / w.sum(-1).clamp(min=1e-8)  # [L]
   314:     return locality.mean().item()
   315: 
   316: 
   317: def verify_placement(
   318:     phy2log: torch.Tensor,
   319:     log2phy: torch.Tensor,
   320:     logcnt: torch.Tensor,
   321:     num_replicas: int,
   322:     num_experts: int,
   323:     num_gpus: int,
   324: ) -> bool:
   325:     """Verify that the placement is valid."""
   326:     L = phy2log.shape[0]
   327: 
   328:     if phy2log.shape != (L, num_replicas):
   329:         return False
   330:     if logcnt.shape != (L, num_experts):
   331:         return False
   332:     if (phy2log < 0).any() or (phy2log >= num_experts).any():
   333:         return False
   334: 
   335:     for layer in range(L):
   336:         for e in range(num_experts):
   337:             actual = (phy2log[layer] == e).sum().item()
   338:             if actual != logcnt[layer, e].item():
   339:                 return False
   340: 
   341:     if logcnt.sum(-1).ne(num_replicas).any():
   342:         return False
   343: 
   344:     return True
   345: 
   346: 
   347: def evaluate(config_name: str, seed: int, num_trials: int = 10, num_timing: int = 20):
   348:     """Run evaluation for a given MoE model configuration."""
   349:     cfg = CONFIGS[config_name]
   350:     L = cfg["num_layers"]
   351:     E = cfg["num_experts"]
   352:     G = cfg["num_groups"]
   353:     N = cfg["num_nodes"]
   354:     D = cfg["num_gpus"]
   355:     R = cfg["num_replicas"]
   356:     za = cfg["zipf_alpha"]
   357:     sr = cfg["skew_ratio"]
   358: 
   359:     print(f"Config: {config_name} (L={L}, E={E}, G={G}, N={N}, D={D}, R={R})")
   360:     print(f"Seed: {seed}, Trials: {num_trials}, Timing iters: {num_timing}")
   361: 
   362:     balances_gpu = []
   363:     balances_node = []
   364:     localities = []
   365:     runtimes = []
   366: 
   367:     for trial in range(num_trials):
   368:         trial_seed = seed * 10000 + trial
   369:         weight = generate_workload(L, E, trial_seed, za, sr)
   370: 
   371:         # Warm up
   372:         for _ in range(3):
   373:             rebalance_experts(weight.clone(), R, G, N, D)
   374: 
   375:         # Time the algorithm
   376:         times = []
   377:         for _ in range(num_timing):
   378:             w = weight.clone()
   379:             t0 = time.perf_counter()
   380:             phy2log, log2phy, logcnt = rebalance_experts(w, R, G, N, D)
   381:             t1 = time.perf_counter()
   382:             times.append((t1 - t0) * 1000)
   383: 
   384:         runtime_ms = np.median(times)
   385: 
   386:         # Verify correctness
   387:         valid = verify_placement(phy2log, log2phy, logcnt, R, E, D)
   388:         if not valid:
   389:             print(f"  Trial {trial}: INVALID placement!", flush=True)
   390:             balances_gpu.append(0.0)
   391:             balances_node.append(0.0)
   392:             localities.append(1.0 / N)
   393:             runtimes.append(runtime_ms)
   394:             continue
   395: 
   396:         # Compute balance at GPU and node level + locality
   397:         bal_gpu, bal_node = compute_balance(weight, phy2log, logcnt, D, N, R)
   398:         loc = compute_locality(weight, phy2log, D, N, R)
   399:         balances_gpu.append(bal_gpu)
   400:         balances_node.append(bal_node)
   401:         localities.append(loc)
   402:         runtimes.append(runtime_ms)
   403: 
   404:         if trial % 3 == 0:
   405:             print(
   406:                 f"TRAIN_METRICS trial={trial} balance={bal_gpu:.4f} "
   407:                 f"balance_node={bal_node:.4f} locality={loc:.4f} "
   408:                 f"runtime_ms={runtime_ms:.3f} valid={int(valid)}",
   409:                 flush=True,
   410:             )
   411: 
   412:     mean_balance = float(np.mean(balances_gpu))
   413:     mean_balance_node = float(np.mean(balances_node))
   414:     mean_locality = float(np.mean(localities))
   415:     mean_runtime = float(np.mean(runtimes))
   416:     std_balance = float(np.std(balances_gpu))
   417:     std_balance_node = float(np.std(balances_node))
   418:     std_locality = float(np.std(localities))
   419:     std_runtime = float(np.std(runtimes))
   420: 
   421:     print(
   422:         f"TEST_METRICS balance={mean_balance:.6f} "
   423:         f"balance_node={mean_balance_node:.6f} "
   424:         f"locality={mean_locality:.6f} "
   425:         f"runtime_ms={mean_runtime:.4f} "
   426:         f"balance_std={std_balance:.6f} "
   427:         f"balance_node_std={std_balance_node:.6f} "
   428:         f"locality_std={std_locality:.6f} "
   429:         f"runtime_std={std_runtime:.4f}",
   430:         flush=True,
   431:     )
   432: 
   433: 
   434: def main():
   435:     parser = argparse.ArgumentParser()
   436:     parser.add_argument("--config", type=str, required=True, choices=list(CONFIGS.keys()))
   437:     parser.add_argument("--seed", type=int, default=42)
   438:     parser.add_argument("--output-dir", type=str, default=".")
   439:     parser.add_argument("--num-trials", type=int, default=10)
   440:     parser.add_argument("--num-timing", type=int, default=20)
   441:     args = parser.parse_args()
   442: 
   443:     os.makedirs(args.output_dir, exist_ok=True)
   444:     evaluate(args.config, args.seed, args.num_trials, args.num_timing)
   445: 
   446: 
   447: if __name__ == "__main__":
   448:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `greedy` baseline — editable region  [READ-ONLY — reference implementation]

In `eplb/custom_eplb.py`:

```python
Lines 62–160:
    59: }
    60: 
    61: # ================================================================
    62: 
    63: def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    64:     B, n = weight.shape
    65:     assert n % num_packs == 0
    66:     items_per_pack = n // num_packs
    67: 
    68:     if items_per_pack == 1:
    69:         idx = torch.arange(n, dtype=torch.int64, device=weight.device).expand(B, -1)
    70:         return idx, torch.zeros_like(idx)
    71: 
    72:     sorted_idx = weight.float().sort(-1, descending=True).indices.cpu()
    73:     pack_index = torch.full((B, n), -1, dtype=torch.int64)
    74:     rank_in_pack = torch.full((B, n), -1, dtype=torch.int64)
    75:     for b in range(B):
    76:         loads = [0.0] * num_packs
    77:         counts = [0] * num_packs
    78:         for j in range(n):
    79:             item = sorted_idx[b, j].item()
    80:             best = min(
    81:                 (p for p in range(num_packs) if counts[p] < items_per_pack),
    82:                 key=lambda p: loads[p],
    83:             )
    84:             pack_index[b, item] = best
    85:             rank_in_pack[b, item] = counts[best]
    86:             loads[best] += weight[b, item].item()
    87:             counts[best] += 1
    88:     return pack_index, rank_in_pack
    89: 
    90: 
    91: def replicate_experts(
    92:     weight: torch.Tensor, num_phy: int
    93: ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    94:     B, num_log = weight.shape
    95:     device = weight.device
    96:     phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(B, 1)
    97:     rank = torch.zeros(B, num_phy, dtype=torch.int64, device=device)
    98:     logcnt = torch.ones(B, num_log, dtype=torch.int64, device=device)
    99:     idx_b = torch.arange(B, dtype=torch.int64, device=device)
   100:     for i in range(num_log, num_phy):
   101:         eff = weight / logcnt.float()
   102:         top = eff.argmax(dim=-1)
   103:         phy2log[:, i] = top
   104:         rank[:, i] = logcnt[idx_b, top]
   105:         logcnt[idx_b, top] += 1
   106:     return phy2log, rank, logcnt
   107: 
   108: 
   109: def rebalance_experts(
   110:     weight: torch.Tensor,
   111:     num_replicas: int,
   112:     num_groups: int,
   113:     num_nodes: int,
   114:     num_gpus: int,
   115: ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
   116:     L, E = weight.shape
   117:     weight = weight.float().cpu()
   118:     group_size = E // num_groups
   119:     gpus_per_node = num_gpus // num_nodes
   120:     phy_per_gpu = num_replicas // num_gpus
   121:     groups_per_node = num_groups // num_nodes
   122:     experts_per_node = E // num_nodes
   123:     replicas_per_node = num_replicas // num_nodes
   124: 
   125:     def inv(perm):
   126:         out = torch.empty_like(perm)
   127:         out.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
   128:         return out
   129: 
   130:     # Stage 1
   131:     tpg = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
   132:     gpi, grk = balanced_packing(tpg, num_nodes)
   133:     log2mlog = (((gpi * groups_per_node + grk) * group_size).unsqueeze(-1)
   134:                 + torch.arange(group_size)).flatten(-2)
   135:     mlog2log = inv(log2mlog)
   136: 
   137:     # Stage 2
   138:     tpm = weight.gather(-1, mlog2log).view(-1, experts_per_node)
   139:     p2m, prk, mcnt = replicate_experts(tpm, replicas_per_node)
   140: 
   141:     # Stage 3
   142:     tpp = (tpm / mcnt.float()).gather(-1, p2m)
   143:     pi, ri = balanced_packing(tpp, gpus_per_node)
   144:     p2pp = pi * phy_per_gpu + ri
   145:     pp2p = inv(p2pp)
   146: 
   147:     pp2m = p2m.gather(-1, pp2p)
   148:     pp2m = (pp2m.view(L, num_nodes, -1)
   149:             + torch.arange(0, E, experts_per_node).view(1, -1, 1)).flatten(-2)
   150:     pp2log = mlog2log.gather(-1, pp2m)
   151:     pprank = prk.gather(-1, pp2p).view(L, -1)
   152:     logcnt = mcnt.view(L, -1).gather(-1, log2mlog)
   153: 
   154:     mx = logcnt.max().item()
   155:     log2phy = torch.full((L, E, mx), -1, dtype=torch.int64)
   156:     log2phy.view(L, -1).scatter_(
   157:         -1, pp2log * mx + pprank,
   158:         torch.arange(num_replicas).expand(L, -1),
   159:     )
   160:     return pp2log, log2phy, logcnt
   161: 
   162: # ================================================================
   163: # FIXED SECTION — Workload generation and evaluation harness
```

### `zigzag` baseline — editable region  [READ-ONLY — reference implementation]

In `eplb/custom_eplb.py`:

```python
Lines 62–162:
    59: }
    60: 
    61: # ================================================================
    62: 
    63: def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    64:     B, n = weight.shape
    65:     assert n % num_packs == 0
    66: 
    67:     if n // num_packs == 1:
    68:         idx = torch.arange(n, dtype=torch.int64, device=weight.device).expand(B, -1)
    69:         return idx, torch.zeros_like(idx)
    70: 
    71:     # Sort items by weight descending
    72:     sorted_idx = weight.float().sort(-1, descending=True).indices
    73: 
    74:     # Zigzag assignment: even blocks go 0..P-1, odd blocks go P-1..0
    75:     positions = torch.arange(n, device=weight.device)
    76:     block_id = positions // num_packs
    77:     pos_in_block = positions % num_packs
    78:     is_even = block_id % 2 == 0
    79:     pack_assign = torch.where(is_even, pos_in_block, num_packs - 1 - pos_in_block)
    80:     rank_assign = block_id
    81: 
    82:     # Scatter back to original item order
    83:     pack_expanded = pack_assign.unsqueeze(0).expand(B, -1)
    84:     rank_expanded = rank_assign.unsqueeze(0).expand(B, -1)
    85:     pack_index = torch.zeros(B, n, dtype=torch.int64, device=weight.device)
    86:     rank_in_pack = torch.zeros(B, n, dtype=torch.int64, device=weight.device)
    87:     pack_index.scatter_(-1, sorted_idx, pack_expanded)
    88:     rank_in_pack.scatter_(-1, sorted_idx, rank_expanded)
    89: 
    90:     return pack_index.cpu(), rank_in_pack.cpu()
    91: 
    92: 
    93: def replicate_experts(
    94:     weight: torch.Tensor, num_phy: int
    95: ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    96:     B, num_log = weight.shape
    97:     device = weight.device
    98:     phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(B, 1)
    99:     rank = torch.zeros(B, num_phy, dtype=torch.int64, device=device)
   100:     logcnt = torch.ones(B, num_log, dtype=torch.int64, device=device)
   101:     idx_b = torch.arange(B, dtype=torch.int64, device=device)
   102:     for i in range(num_log, num_phy):
   103:         eff = weight / logcnt.float()
   104:         top = eff.argmax(dim=-1)
   105:         phy2log[:, i] = top
   106:         rank[:, i] = logcnt[idx_b, top]
   107:         logcnt[idx_b, top] += 1
   108:     return phy2log, rank, logcnt
   109: 
   110: 
   111: def rebalance_experts(
   112:     weight: torch.Tensor,
   113:     num_replicas: int,
   114:     num_groups: int,
   115:     num_nodes: int,
   116:     num_gpus: int,
   117: ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
   118:     L, E = weight.shape
   119:     weight = weight.float().cpu()
   120:     group_size = E // num_groups
   121:     gpus_per_node = num_gpus // num_nodes
   122:     phy_per_gpu = num_replicas // num_gpus
   123:     groups_per_node = num_groups // num_nodes
   124:     experts_per_node = E // num_nodes
   125:     replicas_per_node = num_replicas // num_nodes
   126: 
   127:     def inv(perm):
   128:         out = torch.empty_like(perm)
   129:         out.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
   130:         return out
   131: 
   132:     # Stage 1: zigzag packing of groups to nodes
   133:     tpg = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
   134:     gpi, grk = balanced_packing(tpg, num_nodes)
   135:     log2mlog = (((gpi * groups_per_node + grk) * group_size).unsqueeze(-1)
   136:                 + torch.arange(group_size)).flatten(-2)
   137:     mlog2log = inv(log2mlog)
   138: 
   139:     # Stage 2: greedy replication
   140:     tpm = weight.gather(-1, mlog2log).view(-1, experts_per_node)
   141:     p2m, prk, mcnt = replicate_experts(tpm, replicas_per_node)
   142: 
   143:     # Stage 3: zigzag packing of replicas to GPUs
   144:     tpp = (tpm / mcnt.float()).gather(-1, p2m)
   145:     pi, ri = balanced_packing(tpp, gpus_per_node)
   146:     p2pp = pi * phy_per_gpu + ri
   147:     pp2p = inv(p2pp)
   148: 
   149:     pp2m = p2m.gather(-1, pp2p)
   150:     pp2m = (pp2m.view(L, num_nodes, -1)
   151:             + torch.arange(0, E, experts_per_node).view(1, -1, 1)).flatten(-2)
   152:     pp2log = mlog2log.gather(-1, pp2m)
   153:     pprank = prk.gather(-1, pp2p).view(L, -1)
   154:     logcnt = mcnt.view(L, -1).gather(-1, log2mlog)
   155: 
   156:     mx = logcnt.max().item()
   157:     log2phy = torch.full((L, E, mx), -1, dtype=torch.int64)
   158:     log2phy.view(L, -1).scatter_(
   159:         -1, pp2log * mx + pprank,
   160:         torch.arange(num_replicas).expand(L, -1),
   161:     )
   162:     return pp2log, log2phy, logcnt
   163: 
   164: # ================================================================
   165: # FIXED SECTION — Workload generation and evaluation harness
```

### `flat_zigzag` baseline — editable region  [READ-ONLY — reference implementation]

In `eplb/custom_eplb.py`:

```python
Lines 62–144:
    59: }
    60: 
    61: # ================================================================
    62: 
    63: def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    64:     B, n = weight.shape
    65:     assert n % num_packs == 0
    66: 
    67:     if n // num_packs == 1:
    68:         idx = torch.arange(n, dtype=torch.int64, device=weight.device).expand(B, -1)
    69:         return idx, torch.zeros_like(idx)
    70: 
    71:     sorted_idx = weight.float().sort(-1, descending=True).indices
    72: 
    73:     positions = torch.arange(n, device=weight.device)
    74:     block_id = positions // num_packs
    75:     pos_in_block = positions % num_packs
    76:     is_even = block_id % 2 == 0
    77:     pack_assign = torch.where(is_even, pos_in_block, num_packs - 1 - pos_in_block)
    78:     rank_assign = block_id
    79: 
    80:     pack_expanded = pack_assign.unsqueeze(0).expand(B, -1)
    81:     rank_expanded = rank_assign.unsqueeze(0).expand(B, -1)
    82:     pack_index = torch.zeros(B, n, dtype=torch.int64, device=weight.device)
    83:     rank_in_pack = torch.zeros(B, n, dtype=torch.int64, device=weight.device)
    84:     pack_index.scatter_(-1, sorted_idx, pack_expanded)
    85:     rank_in_pack.scatter_(-1, sorted_idx, rank_expanded)
    86: 
    87:     return pack_index.cpu(), rank_in_pack.cpu()
    88: 
    89: 
    90: def replicate_experts(
    91:     weight: torch.Tensor, num_phy: int
    92: ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    93:     B, num_log = weight.shape
    94:     device = weight.device
    95:     phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(B, 1)
    96:     rank = torch.zeros(B, num_phy, dtype=torch.int64, device=device)
    97:     logcnt = torch.ones(B, num_log, dtype=torch.int64, device=device)
    98:     idx_b = torch.arange(B, dtype=torch.int64, device=device)
    99:     for i in range(num_log, num_phy):
   100:         eff = weight / logcnt.float()
   101:         top = eff.argmax(dim=-1)
   102:         phy2log[:, i] = top
   103:         rank[:, i] = logcnt[idx_b, top]
   104:         logcnt[idx_b, top] += 1
   105:     return phy2log, rank, logcnt
   106: 
   107: 
   108: def rebalance_experts(
   109:     weight: torch.Tensor,
   110:     num_replicas: int,
   111:     num_groups: int,
   112:     num_nodes: int,
   113:     num_gpus: int,
   114: ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
   115:     # Flat (non-hierarchical) approach: skip group-to-node, go directly to global
   116:     L, E = weight.shape
   117:     weight = weight.float().cpu()
   118:     phy_per_gpu = num_replicas // num_gpus
   119: 
   120:     # Step 1: Replicate experts globally
   121:     phy2log, phyrank, logcnt = replicate_experts(weight, num_replicas)
   122: 
   123:     # Step 2: Pack all replicas to GPUs directly using zigzag
   124:     tokens_per_phy = (weight / logcnt.float()).gather(-1, phy2log)
   125:     pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus)
   126: 
   127:     def inv(perm):
   128:         out = torch.empty_like(perm)
   129:         out.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
   130:         return out
   131: 
   132:     phy2pphy = pack_index * phy_per_gpu + rank_in_pack
   133:     pphy2phy = inv(phy2pphy)
   134: 
   135:     final_phy2log = phy2log.gather(-1, pphy2phy)
   136:     final_rank = phyrank.gather(-1, pphy2phy)
   137: 
   138:     mx = logcnt.max().item()
   139:     log2phy = torch.full((L, E, mx), -1, dtype=torch.int64)
   140:     log2phy.view(L, -1).scatter_(
   141:         -1, final_phy2log * mx + final_rank,
   142:         torch.arange(num_replicas).expand(L, -1),
   143:     )
   144:     return final_phy2log, log2phy, logcnt
   145: 
   146: # ================================================================
   147: # FIXED SECTION — Workload generation and evaluation harness
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
