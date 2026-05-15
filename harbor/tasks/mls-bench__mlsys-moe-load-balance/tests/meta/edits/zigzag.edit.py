"""Zigzag/snake tensor pattern baseline for mlsys-moe-load-balance.

Reference: Section 5.2 of "Barbarians at the Gate" (Cheng et al., 2025)
           Internal frontier-lab implementation described in the paper.

Replaces greedy for-loop bin-packing with a vectorized zigzag (snake)
pattern that uses PyTorch tensor operations. Items sorted by weight
are assigned to packs in alternating order: 0,1,...,P-1,P-1,...,1,0,...
This naturally interleaves heavy and light items across packs.
Replication stage remains sequential (for-loop).
~20ms runtime on medium configs.
"""

_FILE = "eplb/custom_eplb.py"

_CONTENT = """\

def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B, n = weight.shape
    assert n % num_packs == 0

    if n // num_packs == 1:
        idx = torch.arange(n, dtype=torch.int64, device=weight.device).expand(B, -1)
        return idx, torch.zeros_like(idx)

    # Sort items by weight descending
    sorted_idx = weight.float().sort(-1, descending=True).indices

    # Zigzag assignment: even blocks go 0..P-1, odd blocks go P-1..0
    positions = torch.arange(n, device=weight.device)
    block_id = positions // num_packs
    pos_in_block = positions % num_packs
    is_even = block_id % 2 == 0
    pack_assign = torch.where(is_even, pos_in_block, num_packs - 1 - pos_in_block)
    rank_assign = block_id

    # Scatter back to original item order
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

    # Stage 1: zigzag packing of groups to nodes
    tpg = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    gpi, grk = balanced_packing(tpg, num_nodes)
    log2mlog = (((gpi * groups_per_node + grk) * group_size).unsqueeze(-1)
                + torch.arange(group_size)).flatten(-2)
    mlog2log = inv(log2mlog)

    # Stage 2: greedy replication
    tpm = weight.gather(-1, mlog2log).view(-1, experts_per_node)
    p2m, prk, mcnt = replicate_experts(tpm, replicas_per_node)

    # Stage 3: zigzag packing of replicas to GPUs
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
