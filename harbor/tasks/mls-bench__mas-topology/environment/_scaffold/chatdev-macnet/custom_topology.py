"""Custom multi-agent collaboration topology.

This module defines the DAG topology for multi-agent code generation.
The function generate_topology(node_num) returns a list of directed edges
that determine how LLM agents collaborate to produce code.

The system will automatically add:
  - An input sentinel node (-1) connecting to all source nodes (no predecessors)
  - An output sentinel node (-2) connecting from all sink nodes (no successors)
"""


# ── Editable topology function ───────────────────────────────────────
# EDITABLE REGION START

def generate_topology(node_num: int) -> list[tuple[int, int]]:
    """Design the multi-agent collaboration topology.

    Given N agent nodes (numbered 0 to node_num-1), return a list of
    directed edges (source, target) forming a DAG. The graph will
    automatically get input/output sentinel nodes (-1, -2) added.

    Constraints:
    - Must form a valid DAG (no cycles)
    - All nodes 0..node_num-1 should be reachable from at least one path
    - Edges must go from lower-indexed to higher-indexed nodes

    Args:
        node_num: Number of agent nodes (e.g., 4, 8, 16)

    Returns:
        edges: List of (source, target) tuples
    """
    # Default: chain topology (sequential pipeline)
    # Each agent improves upon the previous agent's solution.
    # 0 -> 1 -> 2 -> ... -> (node_num - 1)
    edges = []
    for i in range(node_num - 1):
        edges.append((i, i + 1))
    return edges

# EDITABLE REGION END
# ── End of editable region ───────────────────────────────────────────
