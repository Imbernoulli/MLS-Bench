"""Confidence penalty regularization baseline.

Penalizes low-entropy (over-confident) output distributions by subtracting
the entropy of the softmax predictions, scaled by beta=0.1.

Reference: Pereyra et al., "Regularizing Neural Networks by Penalizing
Confident Output Distributions" (ICLR 2017)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_reg.py"

_CONTENT = """\
def compute_regularization(model, inputs, outputs, targets, config):
    \"\"\"Confidence penalty: penalize low-entropy predictions.

    Computes negative entropy of the softmax distribution and adds it
    as a penalty, encouraging the model to be less over-confident.
    Beta=0.1.
    \"\"\"
    probs = F.softmax(outputs, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
    return -0.1 * entropy  # penalize confident (low-entropy) predictions
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 246,
        "end_line": 273,
        "content": _CONTENT,
    },
]
