"""Chain topology baseline -- rigorous codebase edit ops.

Sequential pipeline: 0 -> 1 -> 2 -> ... -> (N-1).
Each agent improves upon the previous agent's solution.
O(N) depth, O(N) edges. Deep iterative refinement, no parallelism.

This is the default implementation in custom_template.py, so this
edit is a no-op (replaces with identical content).
"""

_FILE = "chatdev-macnet/custom_topology.py"

_CHAIN_BODY = """\
    # Chain topology: sequential pipeline
    # Each agent improves upon the previous agent's solution.
    # 0 -> 1 -> 2 -> ... -> (node_num - 1)
    edges = []
    for i in range(node_num - 1):
        edges.append((i, i + 1))
    return edges
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 34,
        "end_line": 40,
        "content": _CHAIN_BODY,
    },
]
