"""Static (load-oblivious) placement baseline for mlsys-moe-load-balance.

A naive reference floor, not a load balancer: it ignores the per-expert
load entirely. Node n hosts the contiguous block of logical experts
[n * E/N, (n+1) * E/N) (i.e. whole expert groups, as in a default static
expert-parallel sharding), the node's spare replica slots are filled by
round-robin copies of its own experts, and slots are laid out on the node's
GPUs in order. It keeps every expert's replicas on one node (locality 1.0)
and costs almost nothing to compute, but leaves load imbalance untouched.
Its balance anchors the zero point of the balance terms.
"""

_FILE = "eplb/custom_eplb.py"

_CONTENT = """\

def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Load-oblivious: item i goes to pack i // (n // num_packs).
    B, n = weight.shape
    assert n % num_packs == 0
    per = n // num_packs
    idx = torch.arange(n, dtype=torch.int64)
    return (idx // per).expand(B, -1).clone(), (idx % per).expand(B, -1).clone()


def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Load-oblivious: spare slots are round-robin copies of experts 0, 1, ...
    B, num_log = weight.shape
    slot = torch.arange(num_phy, dtype=torch.int64)
    phy2log = (slot % num_log).expand(B, -1).clone()
    rank = (slot // num_log).expand(B, -1).clone()
    logcnt = torch.bincount(slot % num_log, minlength=num_log).expand(B, -1).clone()
    return phy2log, rank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L, E = weight.shape
    experts_per_node = E // num_nodes
    replicas_per_node = num_replicas // num_nodes

    # Within each node: replicate the node's own experts round-robin.
    p2m, prk, mcnt = replicate_experts(weight[:1, :experts_per_node], replicas_per_node)
    offset = torch.arange(0, E, experts_per_node, dtype=torch.int64).view(-1, 1)
    phy2log = (p2m + offset).flatten().expand(L, -1).clone()
    phyrank = prk.expand(num_nodes, -1).flatten().expand(L, -1).clone()
    logcnt = mcnt.expand(num_nodes, -1).flatten().expand(L, -1).clone()

    mx = logcnt.max().item()
    log2phy = torch.full((L, E, mx), -1, dtype=torch.int64)
    log2phy.view(L, -1).scatter_(
        -1, phy2log * mx + phyrank,
        torch.arange(num_replicas).expand(L, -1),
    )
    return phy2log, log2phy, logcnt
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
