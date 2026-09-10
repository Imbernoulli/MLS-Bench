"""ERM baseline for security-membership-inference-defense."""

_FILE = "pytorch-vision/custom_membership_defense.py"

_CONTENT = """\
class MembershipDefense:
    \"\"\"Standard cross-entropy training.\"\"\"

    def __init__(self):
        pass

    def compute_loss(self, logits, labels, epoch):
        return F.cross_entropy(logits, labels)
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 9, "end_line": 29, "content": _CONTENT}
]
