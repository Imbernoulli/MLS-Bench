"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/layer_budget.py"

_CONTENT = 'def layer_sparsity(layer_names):\n    # SENSITIVITY-AWARE: prune the classifier head and first conv less (more\n    # accuracy-sensitive), the wide mid-layer convs more; mean normalized to target.\n    out = {}\n    for n in layer_names:\n        if n.startswith("fc") or ".fc." in n:\n            out[n] = 0.40\n        elif n.startswith("conv1"):\n            out[n] = 0.50\n        else:\n            out[n] = 0.80\n    return out\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 17, "content": _CONTENT},
]
