"""Square-root inverse frequency weighting baseline.

weight[c] = 1 / sqrt(count[c]), then normalized so weights sum to num_classes.
A common heuristic that provides gentler reweighting than full inverse frequency.

Reference: Common heuristic used in many imbalanced learning papers.

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_weighting.py"

_CONTENT = """\
def compute_class_weights(class_counts, num_classes, config):
    \"\"\"Square-root inverse frequency weighting.

    weight[c] = 1 / sqrt(count[c]), normalized so weights sum to num_classes.
    Gentler reweighting than full inverse frequency.
    \"\"\"
    weights = 1.0 / torch.sqrt(class_counts)
    # Normalize so weights sum to num_classes
    weights = weights / weights.sum() * num_classes
    return weights
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 164,
        "end_line": 195,
        "content": _CONTENT,
    },
]
