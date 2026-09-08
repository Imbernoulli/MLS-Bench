"""Flat zigzag baseline for mlsys-moe-load-balance.

Skips hierarchical node decomposition and does a single-level
global zigzag packing: replicate experts, then assign all replicas
to all GPUs directly using zigzag tensor pattern.
Faster than hierarchical approaches (no Stage 1 overhead) but may
produce worse balance in multi-node settings since it ignores
inter-node topology.
"""

_FILE = "eplb/custom_eplb.py"

_CONTENT = """\

def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B, n = weight.shape
    assert n % num_packs == 0

    if n // num_packs == 1:
        idx = torch.arange(n, dtype=torch.int64, device=weight.device).expand(B, -1)
        return idx, torch.zeros_like(idx)

    sorted_idx = weight.float().sort(-1, descending=True).indices

    positions = torch.arange(n, device=weight.device)
    block_id = positions // num_packs
    pos_in_block = positions % num_packs
    is_even = block_id % 2 == 0
    pack_assign = torch.where(is_even, pos_in_block, num_packs - 1 - pos_in_block)
    rank_assign = block_id

    pack_expanded = pack_assign.unsqueeze(0).expand(B, -1)
    rank_expanded = rank_assign.unsqueeze(0).expand(B, -1)
    pack_index = torch.zeros(B, n, dtype=torch.int64, device=weight.device)
    rank_in_pack = torch.zeros(B, n, dtype=torch.int64, device=weight.device)
    pack_index.scatter_(-1, sorted_idx, pack_expanded)
    rank_in_pack.scatter_(-1, sorted_idx, rank_expanded)

    return pack_index.cpu(), rank_in_pack.cpu()


def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    # Flat (non-hierarchical) approach: skip group-to-node, go directly to global
    L, E = weight.shape
    weight = weight.float().cpu()
    phy_per_gpu = num_replicas // num_gpus

    # Step 1: Replicate experts globally
    phy2log, phyrank, logcnt = replicate_experts(weight, num_replicas)

    # Step 2: Pack all replicas to GPUs directly using zigzag
    tokens_per_phy = (weight / logcnt.float()).gather(-1, phy2log)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus)

    def inv(perm):
        out = torch.empty_like(perm)
        out.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
        return out

    phy2pphy = pack_index * phy_per_gpu + rank_in_pack
    pphy2phy = inv(phy2pphy)

    final_phy2log = phy2log.gather(-1, pphy2phy)
    final_rank = phyrank.gather(-1, pphy2phy)

    mx = logcnt.max().item()
    log2phy = torch.full((L, E, mx), -1, dtype=torch.int64)
    log2phy.view(L, -1).scatter_(
        -1, final_phy2log * mx + final_rank,
        torch.arange(num_replicas).expand(L, -1),
    )
    return final_phy2log, log2phy, logcnt
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 62,
        "end_line": 209,
        "content": _CONTENT,
    },
]
