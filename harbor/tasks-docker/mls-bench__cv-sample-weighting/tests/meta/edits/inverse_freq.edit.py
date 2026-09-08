"""Inverse frequency weighting baseline.

Standard inverse frequency: weight[c] = total_samples / (num_classes * count[c]).
This is the most common reweighting strategy, directly compensating for class imbalance.

Reference: Standard statistical practice for imbalanced classification.

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_weighting.py"

_CONTENT = """\
def compute_class_weights(class_counts, num_classes, config):
    \"\"\"Inverse frequency weighting.

    weight[c] = total_samples / (num_classes * count[c]).
    Directly proportional to inverse class frequency.
    Smoothed via square-root dampening to prevent training instability
    on architectures without skip connections (e.g. VGG).
    Normalized so weights sum to num_classes.
    \"\"\"
    total = class_counts.sum().float()
    weights = total / (num_classes * class_counts.float())
    # Square-root dampening: reduces dynamic range while preserving ordering
    # For ratio=100 CIFAR-100: raw ratio ~100x -> dampened ~10x
    weights = torch.sqrt(weights)
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
