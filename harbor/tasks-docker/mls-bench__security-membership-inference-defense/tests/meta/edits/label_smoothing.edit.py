"""Label-smoothing baseline for security-membership-inference-defense."""

_FILE = "pytorch-vision/custom_membership_defense.py"

_CONTENT = """\
class MembershipDefense:
    \"\"\"Cross-entropy with fixed label smoothing.\"\"\"

    def __init__(self):
        self.label_smoothing = 0.1

    def compute_loss(self, logits, labels, epoch):
        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 9, "end_line": 29, "content": _CONTENT}
]
