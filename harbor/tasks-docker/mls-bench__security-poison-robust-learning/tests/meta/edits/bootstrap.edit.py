"""Bootstrap baseline for security-poison-robust-learning."""

_FILE = "pytorch-vision/bench/poison/custom_robust_loss.py"

_CONTENT = """\
class RobustLoss:
    \"\"\"Interpolate labels with model predictions.\"\"\"

    def __init__(self):
        self.beta = 0.8

    def compute_loss(self, logits, labels, epoch):
        hard = F.one_hot(labels, num_classes=logits.shape[1]).float()
        soft = torch.softmax(logits.detach(), dim=1)
        target = self.beta * hard + (1.0 - self.beta) * soft
        log_probs = F.log_softmax(logits, dim=1)
        return -(target * log_probs).sum(dim=1).mean()
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 8, "end_line": 16, "content": _CONTENT}
]
