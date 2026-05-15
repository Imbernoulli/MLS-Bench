"""Cross-entropy baseline for security-poison-robust-learning."""

_FILE = "pytorch-vision/bench/poison/custom_robust_loss.py"

_CONTENT = """\
class RobustLoss:
    \"\"\"Standard cross-entropy on poisoned labels.\"\"\"

    def __init__(self):
        pass

    def compute_loss(self, logits, labels, epoch):
        return F.cross_entropy(logits, labels)
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 8, "end_line": 16, "content": _CONTENT}
]
