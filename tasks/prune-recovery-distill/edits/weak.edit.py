"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/recovery_distill.py"

_CONTENT = 'def recovery_loss(logits, targets, teacher_logits):\n    # Plain cross-entropy candidate that does not use the dense teacher.\n    return F.cross_entropy(logits, targets)\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 17, "content": _CONTENT},
]
