"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/recovery_distill.py"

_CONTENT = 'def recovery_loss(logits, targets, teacher_logits):\n    # KD-aware recovery: CE + KL(softmax(student/T) || softmax(teacher/T)), T=2.\n    T = 2.0\n    ce = F.cross_entropy(logits, targets)\n    log_p = F.log_softmax(logits / T, dim=1)\n    q = F.softmax(teacher_logits / T, dim=1)\n    kd = F.kl_div(log_p, q, reduction="batchmean") * (T * T)\n    return 0.5 * ce + 0.5 * kd\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 17, "content": _CONTENT},
]
