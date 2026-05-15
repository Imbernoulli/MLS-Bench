"""Generalized cross-entropy baseline for security-poison-robust-learning."""

_FILE = "pytorch-vision/bench/poison/custom_robust_loss.py"

_CONTENT = """\
class RobustLoss:
    \"\"\"Generalized cross-entropy for noisy labels.\"\"\"

    def __init__(self):
        self.q = 0.7

    def compute_loss(self, logits, labels, epoch):
        probs = torch.softmax(logits, dim=1)
        p = probs.gather(1, labels[:, None]).clamp_min(1e-8)
        return ((1.0 - p.pow(self.q)) / self.q).mean()
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 8, "end_line": 16, "content": _CONTENT}
]
