"""
MoE Expert Parallelism Load Balancing (EPLB) Benchmark
======================================================

Design an efficient expert placement algorithm for Mixture-of-Experts (MoE)
inference that assigns expert replicas to GPUs to minimize load imbalance
while keeping the rebalancing algorithm runtime low.

Metrics:
  - balance: avg_tokens_per_gpu / max_tokens_per_gpu (higher is better, 1.0 = perfect)
  - runtime_ms: time to run the placement algorithm (lower is better)

Available libraries: torch, numpy
"""

import time
import os
import sys
import argparse
from typing import Tuple

import torch
import numpy as np

# ================================================================
# MoE model configurations (benchmark profiles based on real architectures)
# ================================================================
CONFIGS = {
    # DeepSeek-V3/R1-style: 256 routed experts, 8 expert groups, top-8 routing
    # Deployment and replica counts are benchmark modeling assumptions.
    "deepseek-v3": {
        "num_layers": 61, "num_experts": 256, "num_groups": 8,
        "num_nodes": 8, "num_gpus": 64, "num_replicas": 320,
        "zipf_alpha": 0.7, "skew_ratio": 0.85,
    },
    # Qwen3-MoE-style: 128 experts, 8 groups, top-8 routing
    # Deployment and replica counts are benchmark modeling assumptions.
    "qwen3-moe": {
        "num_layers": 48, "num_experts": 128, "num_groups": 8,
        "num_nodes": 4, "num_gpus": 32, "num_replicas": 160,
        "zipf_alpha": 0.5, "skew_ratio": 0.70,
    },
    # DeepSeek-V2-style: 160 routed experts, 8 expert groups, top-6 routing
    # Deployment and replica counts are benchmark modeling assumptions.
    "deepseek-v2": {
        "num_layers": 60, "num_experts": 160, "num_groups": 8,
        "num_nodes": 4, "num_gpus": 32, "num_replicas": 192,
        "zipf_alpha": 0.6, "skew_ratio": 0.75,
    },
    # Stress: 16-node deployment with pathological long-tail traffic, large
    # group hierarchy (groups_per_node=2 makes Stage 1 group-to-node
    # packing non-trivial), and tight replication budget (1.5x). Hidden
    # config designed to keep headroom above the real-model configs.
    "stress-skew": {
        "num_layers": 48, "num_experts": 256, "num_groups": 32,
        "num_nodes": 16, "num_gpus": 128, "num_replicas": 384,
        "zipf_alpha": 1.0, "skew_ratio": 0.95,
    },
}

# ================================================================
# EDITABLE SECTION (lines 62-209)
# Implement your expert placement algorithm below.
# You may define helper functions and modify the three core functions.
# ================================================================

def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted items into num_packs balanced packs.

    Args:
        weight: [B, n] — weight of each item across B batches
        num_packs: number of packs

    Returns:
        pack_index: [B, n] — which pack (0..num_packs-1) each item goes to
        rank_in_pack: [B, n] — position (0..items_per_pack-1) within the pack

    Constraint: each pack must contain exactly n // num_packs items.
    """
    B, n = weight.shape
    assert n % num_packs == 0
    items_per_pack = n // num_packs

    if items_per_pack == 1:
        idx = torch.arange(n, dtype=torch.int64, device=weight.device).expand(B, -1)
        return idx, torch.zeros_like(idx)

    sorted_idx = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full((B, n), -1, dtype=torch.int64)
    rank_in_pack = torch.full((B, n), -1, dtype=torch.int64)
    for b in range(B):
        loads = [0.0] * num_packs
        counts = [0] * num_packs
        for j in range(n):
            item = sorted_idx[b, j].item()
            best = min(
                (p for p in range(num_packs) if counts[p] < items_per_pack),
                key=lambda p: loads[p],
            )
            pack_index[b, item] = best
            rank_in_pack[b, item] = counts[best]
            loads[best] += weight[b, item].item()
            counts[best] += 1
    return pack_index, rank_in_pack


def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate num_log logical experts into num_phy physical slots
    to minimize the maximum per-replica load.

    Args:
        weight: [B, num_log] — load per logical expert
        num_phy: total physical expert slots (>= num_log)

    Returns:
        phy2log: [B, num_phy] — logical expert ID for each physical slot
        rank: [B, num_phy] — replica rank (0 = original, 1+ = copies)
        logcnt: [B, num_log] — number of replicas per logical expert
    """
    B, num_log = weight.shape
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(B, 1)
    rank = torch.zeros(B, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(B, num_log, dtype=torch.int64, device=device)
    idx_b = torch.arange(B, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        eff = weight / logcnt.float()
        top = eff.argmax(dim=-1)
        phy2log[:, i] = top
        rank[:, i] = logcnt[idx_b, top]
        logcnt[idx_b, top] += 1
    return phy2log, rank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Main entry point: hierarchical expert placement across GPUs.

    Stage 1: Pack expert groups across nodes (inter-node balancing)
    Stage 2: Create replicas for popular experts within each node
    Stage 3: Pack physical replicas to GPUs (intra-node balancing)

    Args:
        weight: [L, E] — token load per expert per layer
        num_replicas: total physical expert slots (multiple of num_gpus)
        num_groups: number of expert groups
        num_nodes: number of server nodes
        num_gpus: total GPUs (multiple of num_nodes)

    Returns:
        phy2log: [L, num_replicas] — logical expert for each physical slot
        log2phy: [L, E, max_rep] — physical IDs per expert (-1 = unused)
        logcnt: [L, E] — replica count per expert
    """
    L, E = weight.shape
    weight = weight.float().cpu()
    group_size = E // num_groups
    gpus_per_node = num_gpus // num_nodes
    phy_per_gpu = num_replicas // num_gpus
    groups_per_node = num_groups // num_nodes
    experts_per_node = E // num_nodes
    replicas_per_node = num_replicas // num_nodes

    def inv(perm):
        out = torch.empty_like(perm)
        out.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
        return out

    # Stage 1
    tpg = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    gpi, grk = balanced_packing(tpg, num_nodes)
    log2mlog = (((gpi * groups_per_node + grk) * group_size).unsqueeze(-1)
                + torch.arange(group_size)).flatten(-2)
    mlog2log = inv(log2mlog)

    # Stage 2
    tpm = weight.gather(-1, mlog2log).view(-1, experts_per_node)
    p2m, prk, mcnt = replicate_experts(tpm, replicas_per_node)

    # Stage 3
    tpp = (tpm / mcnt.float()).gather(-1, p2m)
    pi, ri = balanced_packing(tpp, gpus_per_node)
    p2pp = pi * phy_per_gpu + ri
    pp2p = inv(p2pp)

    pp2m = p2m.gather(-1, pp2p)
    pp2m = (pp2m.view(L, num_nodes, -1)
            + torch.arange(0, E, experts_per_node).view(1, -1, 1)).flatten(-2)
    pp2log = mlog2log.gather(-1, pp2m)
    pprank = prk.gather(-1, pp2p).view(L, -1)
    logcnt = mcnt.view(L, -1).gather(-1, log2mlog)

    mx = logcnt.max().item()
    log2phy = torch.full((L, E, mx), -1, dtype=torch.int64)
    log2phy.view(L, -1).scatter_(
        -1, pp2log * mx + pprank,
        torch.arange(num_replicas).expand(L, -1),
    )
    return pp2log, log2phy, logcnt

# ================================================================
# FIXED SECTION — Workload generation and evaluation harness
# Do not modify below this line
# ================================================================


def generate_workload(num_layers: int, num_experts: int, seed: int,
                      zipf_alpha: float = 1.5, skew_ratio: float = 0.8) -> torch.Tensor:
    """Generate synthetic MoE expert load distributions.

    Creates realistic workloads mixing uniform and skewed (Zipf) patterns
    to simulate real expert utilization during inference.
    """
    rng = np.random.default_rng(seed)
    weight = np.zeros((num_layers, num_experts), dtype=np.float32)

    for layer in range(num_layers):
        layer_seed = seed * 1000 + layer
        layer_rng = np.random.default_rng(layer_seed)

        # Base uniform load
        base = layer_rng.uniform(100, 500, size=num_experts).astype(np.float32)

        # Zipf-like skew: some experts are much more popular
        ranks = np.arange(1, num_experts + 1, dtype=np.float32)
        zipf = 1.0 / np.power(ranks, zipf_alpha)
        perm = layer_rng.permutation(num_experts)
        zipf_load = zipf[perm] * layer_rng.uniform(5000, 20000)

        # Mix uniform and skewed
        weight[layer] = base * (1 - skew_ratio) + zipf_load * skew_ratio

    return torch.from_numpy(weight)


def compute_balance(
    weight: torch.Tensor,
    phy2log: torch.Tensor,
    logcnt: torch.Tensor,
    num_gpus: int,
    num_nodes: int,
    num_replicas: int,
) -> Tuple[float, float]:
    """Compute load balance at GPU and node level.

    Returns
    -------
    balance_gpu : mean_gpu_load / max_gpu_load (higher better, 1.0 = perfect)
    balance_node : mean_node_load / max_node_load (higher better)

    Both are reported because the placement algorithm must balance load at
    BOTH levels. A globally-balanced flat scheme can score well on
    balance_gpu while leaving inter-node load uneven (or vice versa). The
    score combines them so methods must respect node hierarchy.
    """
    L = weight.shape[0]
    phy_per_gpu = num_replicas // num_gpus
    gpus_per_node = num_gpus // num_nodes
    tokens_per_phy = (weight / logcnt.float()).gather(-1, phy2log)
    tokens_per_gpu = tokens_per_phy.view(L, num_gpus, phy_per_gpu).sum(-1)
    bal_gpu = tokens_per_gpu.mean(-1) / tokens_per_gpu.max(-1).values.clamp(min=1e-8)
    tokens_per_node = tokens_per_gpu.view(L, num_nodes, gpus_per_node).sum(-1)
    bal_node = tokens_per_node.mean(-1) / tokens_per_node.max(-1).values.clamp(min=1e-8)
    return bal_gpu.mean().item(), bal_node.mean().item()


def compute_locality(
    weight: torch.Tensor,
    phy2log: torch.Tensor,
    num_gpus: int,
    num_nodes: int,
    num_replicas: int,
) -> float:
    """Traffic-weighted node locality.

    For each (layer, logical expert), counts the number of distinct nodes
    holding a replica and computes 1 / nodes_per_expert. Averaged over
    experts weighted by traffic (per-layer expert weight) and over layers.
    Returns a value in [1/num_nodes, 1.0]. Higher is better.

    Captures inter-node communication cost — when a token routes to expert
    e, locality_e is the probability that a chosen replica is on the same
    node the token already lives on (roughly: it is the expected fraction
    of replica options reachable without crossing a node boundary).

    A hierarchical scheme that keeps every expert's replicas co-located on
    one node scores 1.0. A flat scheme that scatters replicas across all
    nodes uniformly scores 1 / num_nodes. Pure load-balance metrics cannot
    distinguish these two regimes; locality does.
    """
    L, E = weight.shape
    phy_per_gpu = num_replicas // num_gpus
    gpus_per_node = num_gpus // num_nodes
    phy_per_node = phy_per_gpu * gpus_per_node

    slot_node = (torch.arange(num_replicas, dtype=torch.int64) // phy_per_node)  # [R]
    combo = phy2log.long() * num_nodes + slot_node.unsqueeze(0)  # [L, R]
    presence = torch.zeros(L, E * num_nodes, dtype=torch.int64)
    presence.scatter_(1, combo, torch.ones_like(combo, dtype=torch.int64))
    nodes_per_expert = (presence.view(L, E, num_nodes) > 0).sum(-1).float().clamp(min=1.0)  # [L, E]

    w = weight.float()
    locality = (w / nodes_per_expert).sum(-1) / w.sum(-1).clamp(min=1e-8)  # [L]
    return locality.mean().item()


def verify_placement(
    phy2log: torch.Tensor,
    log2phy: torch.Tensor,
    logcnt: torch.Tensor,
    num_replicas: int,
    num_experts: int,
    num_gpus: int,
) -> bool:
    """Verify that the placement is valid."""
    L = phy2log.shape[0]

    if phy2log.shape != (L, num_replicas):
        return False
    if logcnt.shape != (L, num_experts):
        return False
    if (phy2log < 0).any() or (phy2log >= num_experts).any():
        return False

    for layer in range(L):
        for e in range(num_experts):
            actual = (phy2log[layer] == e).sum().item()
            if actual != logcnt[layer, e].item():
                return False

    if logcnt.sum(-1).ne(num_replicas).any():
        return False

    return True


def evaluate(config_name: str, seed: int, num_trials: int = 10, num_timing: int = 20):
    """Run evaluation for a given MoE model configuration."""
    cfg = CONFIGS[config_name]
    L = cfg["num_layers"]
    E = cfg["num_experts"]
    G = cfg["num_groups"]
    N = cfg["num_nodes"]
    D = cfg["num_gpus"]
    R = cfg["num_replicas"]
    za = cfg["zipf_alpha"]
    sr = cfg["skew_ratio"]

    print(f"Config: {config_name} (L={L}, E={E}, G={G}, N={N}, D={D}, R={R})")
    print(f"Seed: {seed}, Trials: {num_trials}, Timing iters: {num_timing}")

    balances_gpu = []
    balances_node = []
    localities = []
    runtimes = []

    for trial in range(num_trials):
        trial_seed = seed * 10000 + trial
        weight = generate_workload(L, E, trial_seed, za, sr)

        # Warm up
        for _ in range(3):
            rebalance_experts(weight.clone(), R, G, N, D)

        # Time the algorithm
        times = []
        for _ in range(num_timing):
            w = weight.clone()
            t0 = time.perf_counter()
            phy2log, log2phy, logcnt = rebalance_experts(w, R, G, N, D)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        runtime_ms = np.median(times)

        # Verify correctness
        valid = verify_placement(phy2log, log2phy, logcnt, R, E, D)
        if not valid:
            print(f"  Trial {trial}: INVALID placement!", flush=True)
            balances_gpu.append(0.0)
            balances_node.append(0.0)
            localities.append(1.0 / N)
            runtimes.append(runtime_ms)
            continue

        # Compute balance at GPU and node level + locality
        bal_gpu, bal_node = compute_balance(weight, phy2log, logcnt, D, N, R)
        loc = compute_locality(weight, phy2log, D, N, R)
        balances_gpu.append(bal_gpu)
        balances_node.append(bal_node)
        localities.append(loc)
        runtimes.append(runtime_ms)

        if trial % 3 == 0:
            print(
                f"TRAIN_METRICS trial={trial} balance={bal_gpu:.4f} "
                f"balance_node={bal_node:.4f} locality={loc:.4f} "
                f"runtime_ms={runtime_ms:.3f} valid={int(valid)}",
                flush=True,
            )

    mean_balance = float(np.mean(balances_gpu))
    mean_balance_node = float(np.mean(balances_node))
    mean_locality = float(np.mean(localities))
    mean_runtime = float(np.mean(runtimes))
    std_balance = float(np.std(balances_gpu))
    std_balance_node = float(np.std(balances_node))
    std_locality = float(np.std(localities))
    std_runtime = float(np.std(runtimes))

    print(
        f"TEST_METRICS balance={mean_balance:.6f} "
        f"balance_node={mean_balance_node:.6f} "
        f"locality={mean_locality:.6f} "
        f"runtime_ms={mean_runtime:.4f} "
        f"balance_std={std_balance:.6f} "
        f"balance_node_std={std_balance_node:.6f} "
        f"locality_std={std_locality:.6f} "
        f"runtime_std={std_runtime:.4f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--num-timing", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(args.config, args.seed, args.num_trials, args.num_timing)


if __name__ == "__main__":
    main()
