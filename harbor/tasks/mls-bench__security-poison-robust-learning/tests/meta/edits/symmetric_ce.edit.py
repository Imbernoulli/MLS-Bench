"""Symmetric cross-entropy baseline for security-poison-robust-learning."""

_FILE = "pytorch-vision/bench/poison/custom_robust_loss.py"

_CONTENT = """\
class RobustLoss:
    \"\"\"Cross-entropy plus reverse-CE penalty.\"\"\"

    def __init__(self):
        self.alpha = 1.0
        self.beta = 0.5

    def compute_loss(self, logits, labels, epoch):
        ce = F.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=1).clamp_min(1e-8)
        one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float().clamp_min(1e-4)
        rce = -(probs * torch.log(one_hot)).sum(dim=1).mean()
        return self.alpha * ce + self.beta * rce
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 8, "end_line": 16, "content": _CONTENT}
]
