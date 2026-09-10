"""Label Smoothing baseline.

Replaces hard one-hot targets with smoothed distribution:
targets = (1 - eps) * one_hot + eps / C, with eps=0.1.

Reference: Szegedy et al., "Rethinking the Inception Architecture" (CVPR 2016)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_loss.py"

_CONTENT = """\
def compute_loss(logits, targets, config):
    \"\"\"Label Smoothing cross-entropy (eps=0.1).

    Softens hard targets to (1-eps)*one_hot + eps/C, preventing
    overconfident predictions and improving generalization.
    \"\"\"
    return F.cross_entropy(logits, targets, label_smoothing=0.1)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 246,
        "end_line": 266,
        "content": _CONTENT,
    },
]
