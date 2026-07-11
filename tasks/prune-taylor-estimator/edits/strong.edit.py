"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/taylor_estimator.py"

_CONTENT = 'def estimate_importance(model, batches, params):\n    # First-order TAYLOR importance |w * grad| accumulated over the batches.\n    import torch.nn.functional as F\n    dev = next(model.parameters()).device\n    scores = {name: torch.zeros_like(p) for name, p in params}\n    for x, y in batches:\n        x, y = x.to(dev), y.to(dev)\n        model.zero_grad()\n        F.cross_entropy(model(x), y).backward()\n        for name, p in params:\n            if p.grad is not None:\n                scores[name] += (p.detach() * p.grad.detach()).abs()\n    n = max(1, len(batches))\n    return {name: s / n for name, s in scores.items()}\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 17, "content": _CONTENT},
]
