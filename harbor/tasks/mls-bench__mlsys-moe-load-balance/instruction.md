# MLS-Bench: mlsys-moe-load-balance

# MoE Expert Parallelism Load Balancing

## Research Question

Design an efficient expert placement algorithm for Mixture-of-Experts
(MoE) inference that assigns expert replicas to GPUs to minimize load
imbalance — at both the GPU and node level — while preserving inter-node
locality of replicas and keeping the rebalancing algorithm runtime low.

## Background

In MoE models, different experts receive different amounts of traffic
depending on the input distribution. During inference, experts are
distributed across GPUs, and load imbalance causes some GPUs to become
bottlenecks. An Expert Parallelism Load Balancer runs periodically to
rebalance expert placement as workload patterns change.

A standard three-stage hierarchical algorithm is:

1. Group-to-node packing: distribute expert groups across server nodes to
   balance inter-node load.
2. Expert replication: create additional replicas of popular (hot)
   experts within each node.
3. Replica-to-GPU packing: assign physical expert replicas to GPUs within
   each node.

Algorithms must be both effective (low imbalance and good replica
locality) and fast at runtime.

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

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/eplb/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — are not permitted.

- `eplb/custom_eplb.py`
- editable lines **62–209**

## Readable Context

### `eplb/custom_eplb.py`  [EDITABLE — lines 62–209 only]

```python
     1: """
     2: MoE Expert Parallelism Load Balancing
     3: =====================================
     4:
     5: Design an efficient expert placement algorithm for Mixture-of-Experts (MoE)
     6: inference that assigns expert replicas to GPUs to minimize load imbalance
     7: while keeping the rebalancing algorithm runtime low.
     8:
     9: Available libraries: torch, numpy
    10: """
    11:
    12: import time
    13: import os
    14: import sys
    15: import argparse
    16: from typing import Tuple
    17:
    18: import torch
    19: import numpy as np
    20:
    21: # ================================================================
    22: # Deployment configurations (contents withheld)
    23: # Configurations supply num_layers, num_experts, num_groups,
    24: # num_nodes, num_gpus, num_replicas, and workload-skew knobs.
    25: # ================================================================
    26: CONFIGS = { ... }
    27:
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
   165:     # Default skeleton implementation provided below; replace as desired.
   166:     L, E = weight.shape
   167:     weight = weight.float().cpu()
   168:     group_size = E // num_groups
   169:     gpus_per_node = num_gpus // num_nodes
   170:     phy_per_gpu = num_replicas // num_gpus
   171:     groups_per_node = num_groups // num_nodes
   172:     experts_per_node = E // num_nodes
   173:     replicas_per_node = num_replicas // num_nodes
   174:
   175:     def inv(perm):
   176:         out = torch.empty_like(perm)
   177:         out.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
   178:         return out
   179:
   180:     # Stage 1
   181:     tpg = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
   182:     gpi, grk = balanced_packing(tpg, num_nodes)
   183:     log2mlog = (((gpi * groups_per_node + grk) * group_size).unsqueeze(-1)
   184:                 + torch.arange(group_size)).flatten(-2)
   185:     mlog2log = inv(log2mlog)
   186:
   187:     # Stage 2
   188:     tpm = weight.gather(-1, mlog2log).view(-1, experts_per_node)
   189:     p2m, prk, mcnt = replicate_experts(tpm, replicas_per_node)
   190:
   191:     # Stage 3
   192:     tpp = (tpm / mcnt.float()).gather(-1, p2m)
   193:     pi, ri = balanced_packing(tpp, gpus_per_node)
   194:     p2pp = pi * phy_per_gpu + ri
   195:     pp2p = inv(p2pp)
   196:
   197:     pp2m = p2m.gather(-1, pp2p)
   198:     pp2m = (pp2m.view(L, num_nodes, -1)
   199:             + torch.arange(0, E, experts_per_node).view(1, -1, 1)).flatten(-2)
   200:     pp2log = mlog2log.gather(-1, pp2m)
   201:     pprank = prk.gather(-1, pp2p).view(L, -1)
   202:     logcnt = mcnt.view(L, -1).gather(-1, log2mlog)
   203:
   204:     mx = logcnt.max().item()
   205:     log2phy = torch.full((L, E, mx), -1, dtype=torch.int64)
   206:     log2phy.view(L, -1).scatter_(
   207:         -1, pp2log * mx + pprank,
   208:         torch.arange(num_replicas).expand(L, -1),
   209:     )
   210:     return pp2log, log2phy, logcnt
   211:
   212: # ================================================================
   213: # FIXED SECTION — Workload generation and harness (withheld)
   214: # Do not modify below this line
   215: # ================================================================
```

## Tips

- Keep the function/class signatures of the editable regions identical.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.

Good luck.
