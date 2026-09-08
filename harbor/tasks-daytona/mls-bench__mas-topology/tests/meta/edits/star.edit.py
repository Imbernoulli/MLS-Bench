"""Star topology baseline -- rigorous codebase edit ops.

Hub-and-spoke: node 0 broadcasts to all others.
0 -> 1, 0 -> 2, ..., 0 -> (N-1).
O(1) depth, O(N) edges. Parallel diverse solutions, no iterative refinement.
"""

_FILE = "chatdev-macnet/custom_topology.py"

_STAR_BODY = """\
    # Star topology: hub-and-spoke
    # Node 0 broadcasts to all other nodes in parallel.
    # Good for generating diverse solutions simultaneously.
    edges = []
    for i in range(1, node_num):
        edges.append((0, i))
    return edges
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 34,
        "end_line": 40,
        "content": _STAR_BODY,
    },
]
