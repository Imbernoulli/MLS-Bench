"""PolyLoss baseline.

Extends CE with a polynomial correction term: CE + epsilon * (1 - pt),
where pt is the predicted probability of the true class. Uses epsilon=2.0.

Reference: Leng et al., "PolyLoss: A Polynomial Expansion Perspective of
Classification Loss Functions" (ICLR 2022)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_loss.py"

_CONTENT = """\
def compute_loss(logits, targets, config):
    \"\"\"PolyLoss (epsilon=2.0).

    Adds polynomial correction to CE: CE + eps*(1-pt), where pt is the
    softmax probability assigned to the true class.
    \"\"\"
    ce = F.cross_entropy(logits, targets)
    pt = F.softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze()
    return ce + 2.0 * (1 - pt).mean()
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 246,
        "end_line": 266,
        "content": _CONTENT,
    },
]
