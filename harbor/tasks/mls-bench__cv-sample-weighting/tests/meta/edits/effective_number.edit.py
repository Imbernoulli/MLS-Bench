"""Effective number of samples weighting baseline.

Uses the effective number of samples: E_n = (1 - beta^n) / (1 - beta), where
beta = (N-1)/N ~ 0.9999. Weight[c] = 1 / E_n[c], then normalized.

Reference: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples"
(CVPR 2019).

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_weighting.py"

_CONTENT = """\
def compute_class_weights(class_counts, num_classes, config):
    \"\"\"Effective number of samples weighting (Cui et al., CVPR 2019).

    E_n = (1 - beta^n) / (1 - beta).
    weight[c] = (1 - beta) / (1 - beta^count[c]).
    Uses beta=0.9999, a task-local value explored in class-balanced losses.
    Smoothed via square-root dampening to prevent training instability
    on architectures without skip connections (e.g. VGG).
    \"\"\"
    beta = 0.9999
    effective_num = 1.0 - torch.pow(beta, class_counts.float())
    weights = (1.0 - beta) / effective_num
    # Square-root dampening: reduces dynamic range while preserving ordering
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
