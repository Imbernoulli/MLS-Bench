"""Symmetric Cross-Entropy baseline.

Combines standard CE with a reverse cross-entropy term to improve robustness
to noisy labels and overconfident predictions. The reverse CE uses model
predictions as soft targets and the one-hot labels to weight them.

Total loss = alpha * CE(p, y) + beta * CE(y, p)

Reference: Wang et al., "Symmetric Cross Entropy for Robust Learning with
Noisy Labels" (ICCV 2019)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_loss.py"

_CONTENT = """\
def compute_loss(logits, targets, config):
    \"\"\"Symmetric Cross-Entropy (Wang et al., ICCV 2019).

    Combines standard CE with reverse CE for robust learning.
    L = alpha * CE(p, y) + beta * RCE(p, y)
    where RCE = -sum(p * log(y_smooth)).
    alpha=1.0, beta=0.1, clamp for numerical stability.
    \"\"\"
    C = config['num_classes']
    alpha = 1.0
    beta = 0.1

    # Standard CE
    ce = F.cross_entropy(logits, targets)

    # Reverse CE: -sum(p * log(y_smooth))
    # Clamp predictions for numerical stability
    pred = F.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
    # One-hot with label smoothing to avoid log(0) and stabilise RCE
    y_one_hot = F.one_hot(targets, C).float()
    y_smooth = y_one_hot * (1 - 0.1) + 0.1 / C  # label smoothing eps=0.1
    rce = -(pred * y_smooth.log()).sum(dim=1).mean()

    return alpha * ce + beta * rce
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
