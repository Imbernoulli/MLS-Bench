"""LDAM-style class-dependent weighting baseline.

Uses the class-dependent n^{-1/4} scaling from LDAM inside the task's
class-weighting interface.

Reference: Cao et al., "Learning Imbalanced Datasets with Label-Distribution-
Aware Margin Loss" (NeurIPS 2019)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_weighting.py"

_CONTENT = """\
def compute_class_weights(class_counts, num_classes, config):
    \"\"\"LDAM-inspired n^{-1/4} weighting (Cao et al., NeurIPS 2019).

    Weights each class by count^{-1/4}, using the LDAM class-dependent
    scaling in the task's class-weight API.
    Normalized so weights sum to num_classes.
    \"\"\"
    weights = class_counts.float().pow(-0.25)
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
