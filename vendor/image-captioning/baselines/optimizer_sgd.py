"""Design the Optimizer and LR Schedule — weak baseline (sgd).

Reference implementation for the caption-optimizer surface (make_optimizer). See tasks/caption-optimizer/edits/sgd.edit.py.
"""
import torch


def make_optimizer(params):
    # Plain SGD with a high, fixed learning rate and no momentum/weight-decay:
    # a poor match for a tiny mapping fine-tune (unstable, slow to converge).
    return torch.optim.SGD(params, lr=1e-2)
