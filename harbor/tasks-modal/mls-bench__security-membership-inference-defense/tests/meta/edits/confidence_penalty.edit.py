"""Confidence-penalty baseline for security-membership-inference-defense."""

_FILE = "pytorch-vision/custom_membership_defense.py"

_CONTENT = """\
class MembershipDefense:
    \"\"\"Cross-entropy minus predictive entropy bonus.\"\"\"

    def __init__(self):
        self.entropy_weight = 0.1

    def compute_loss(self, logits, labels, epoch):
        ce = F.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()
        return ce - self.entropy_weight * entropy
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 9, "end_line": 29, "content": _CONTENT}
]
