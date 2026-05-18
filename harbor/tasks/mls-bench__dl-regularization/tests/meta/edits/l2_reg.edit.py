"""L2 (weight decay) regularization baseline.

Explicitly computes the squared L2 norm of all trainable weight parameters
(excluding biases and BatchNorm) and returns it scaled by lambda=5e-4.
Equivalent to setting weight_decay=5e-4 in the optimizer.

Reference: Krogh & Hertz, "A Simple Weight Decay Can Improve
Generalization" (NeurIPS 1991)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_reg.py"

_CONTENT = """\
def compute_regularization(model, inputs, outputs, targets, config):
    \"\"\"L2 weight decay regularization.

    Computes sum of squared L2 norms over all weight parameters
    (excluding biases and BatchNorm parameters), scaled by 5e-4.
    Equivalent to optimizer weight_decay=5e-4.
    \"\"\"
    l2_lambda = 5e-4
    reg = torch.tensor(0.0, device=outputs.device)
    for name, p in model.named_parameters():
        if 'weight' in name and 'bn' not in name and p.requires_grad:
            reg = reg + (p ** 2).sum()
    return l2_lambda * reg
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
