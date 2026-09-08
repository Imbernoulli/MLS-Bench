"""Layered (MLP-like) topology baseline -- rigorous codebase edit ops.

Splits nodes into layers of roughly equal size, fully connects adjacent
layers. Like a neural network's MLP structure.
Balances depth and width, inspired by topology comparisons in MacNet-style
multi-agent collaboration experiments.
"""

_FILE = "chatdev-macnet/custom_topology.py"

_LAYERED_BODY = """\
    # Layered (MLP-like) topology
    # Split nodes into layers, fully connect adjacent layers.
    # Balances depth (iterative refinement) and width (diversity).
    import math
    layer_num = max(2, int(math.log(node_num, 2)))
    layers = [node_num // layer_num] * layer_num
    # Distribute remainder to first layer
    layers[0] += node_num % layer_num

    # Compute start/end indices for each layer
    start_ids = []
    end_ids = []
    cur = 0
    for size in layers:
        start_ids.append(cur)
        end_ids.append(cur + size)
        cur += size

    # Fully connect adjacent layers
    edges = []
    for i in range(len(layers) - 1):
        for u in range(start_ids[i], end_ids[i]):
            for v in range(start_ids[i + 1], end_ids[i + 1]):
                edges.append((u, v))
    return edges
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 34,
        "end_line": 40,
        "content": _LAYERED_BODY,
    },
]
