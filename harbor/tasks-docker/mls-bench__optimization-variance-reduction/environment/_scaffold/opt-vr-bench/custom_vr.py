"""Variance Reduction Benchmark for Finite-Sum Optimization (optimizer module).

This file holds the agent-editable variance-reduction optimizer together with
the FIXED gradient helpers it is allowed to call. The data loaders (which build
the synthetic ground-truth w_true), the test labels, the per-problem scoring
(``evaluate``) and the training driver (``train_problem`` / ``main``) live in a
separate FIXED module (``vr_driver.py``) that the agent's process cannot reach
from here. ``vr_driver.py`` loads the PRE-GENERATED observable data and drives
this optimizer, so the editable optimizer can neither name/recompute the eval
datasets, read X_test/y_test, nor score against the held-out labels.

Evaluates variance reduction strategies for stochastic gradient methods on
finite-sum problems:  min_x  F(x) = (1/n) * sum_{i=1}^{n} f_i(x)

Benchmarks:
  1. logistic  -- L2-regularized logistic regression on MNIST (convex)
  2. mlp       -- 2-layer MLP on CIFAR-10 (non-convex)
  3. conditioned -- L2-regularized linear regression on synthetic
                    ill-conditioned data (strongly convex)

The benchmark is run via:  python opt-vr-bench/vr_driver.py --problem <name> ...
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# FIXED -- Gradient helpers available to the optimizer
# ============================================================================

def compute_loss_on_batch(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                          loss_type: str, l2_reg: float) -> torch.Tensor:
    """Compute loss on a batch, including L2 regularization."""
    pred = model(X)
    if loss_type == "cross_entropy":
        loss = F.cross_entropy(pred, y)
    elif loss_type == "mse":
        loss = F.mse_loss(pred, y)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    # L2 regularization
    if l2_reg > 0:
        reg = sum(p.pow(2).sum() for p in model.parameters()) * l2_reg / 2
        loss = loss + reg
    return loss


def compute_full_gradient(model: nn.Module, X_train: torch.Tensor,
                          y_train: torch.Tensor, loss_type: str,
                          l2_reg: float, device: torch.device,
                          batch_size: int = 512) -> List[torch.Tensor]:
    """Compute the full gradient (1/n) * sum_i grad f_i(x) over all training data.

    Returns a list of gradient tensors, one per parameter (same order as
    model.parameters()).
    """
    model.zero_grad()
    n = X_train.size(0)
    # Accumulate gradient over mini-batches for memory efficiency
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Xb = X_train[start:end].to(device)
        yb = y_train[start:end].to(device)
        loss = compute_loss_on_batch(model, Xb, yb, loss_type, l2_reg)
        # Scale by fraction of data in this batch
        (loss * (end - start) / n).backward()
    full_grad = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    return full_grad


def compute_stochastic_gradient(model: nn.Module, X_batch: torch.Tensor,
                                y_batch: torch.Tensor, loss_type: str,
                                l2_reg: float) -> List[torch.Tensor]:
    """Compute stochastic gradient on a mini-batch.

    Returns a list of gradient tensors, one per parameter.
    """
    model.zero_grad()
    loss = compute_loss_on_batch(model, X_batch, y_batch, loss_type, l2_reg)
    loss.backward()
    sg = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    return sg


# ============================================================================
# EDITABLE -- Variance Reduction Strategy
# ============================================================================
# Design a variance reduction mechanism for stochastic gradient computation.
# You may modify ONLY this section.
#
# Interface contract:
#   - VarianceReductionOptimizer.__init__(model, lr, l2_reg, loss_type, n_train, batch_size, device)
#   - VarianceReductionOptimizer.train_one_epoch(X_train, y_train)
#     -> trains for one epoch, returns dict with 'avg_loss'
#
# Available helper functions (FIXED, defined above):
#   - compute_full_gradient(model, X_train, y_train, loss_type, l2_reg, device)
#     -> returns list of full gradient tensors
#   - compute_stochastic_gradient(model, X_batch, y_batch, loss_type, l2_reg)
#     -> returns list of stochastic gradient tensors on a mini-batch
#   - compute_loss_on_batch(model, X_batch, y_batch, loss_type, l2_reg)
#     -> returns scalar loss tensor
#
# Constraints:
#   - Must work across all problems with the shared hyperparameter config
#   - May use full gradient computation (compute_full_gradient) at most once
#     per epoch (to maintain sublinear per-epoch cost)
#   - Must respect the provided learning rate and L2 regularization
#   - The model parameters should be updated in-place (via param.data)

class VarianceReductionOptimizer:
    """Variance reduction strategy for finite-sum optimization.

    Default implementation: vanilla mini-batch SGD (no variance reduction).
    The agent should replace this with a variance-reduced method.
    """

    def __init__(self, model: nn.Module, lr: float, l2_reg: float,
                 loss_type: str, n_train: int, batch_size: int,
                 device: torch.device):
        self.model = model
        self.lr = lr
        self.l2_reg = l2_reg
        self.loss_type = loss_type
        self.n_train = n_train
        self.batch_size = batch_size
        self.device = device
        self.params = list(model.parameters())

    def train_one_epoch(self, X_train: torch.Tensor,
                        y_train: torch.Tensor) -> dict:
        """Train for one pass over the data.

        Args:
            X_train: full training features [n, ...]
            y_train: full training labels [n, ...]

        Returns:
            dict with at least 'avg_loss' key
        """
        self.model.train()
        n = X_train.size(0)
        indices = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            idx = indices[start:end]
            Xb = X_train[idx].to(self.device)
            yb = y_train[idx].to(self.device)

            # Standard SGD: compute stochastic gradient and update
            self.model.zero_grad()
            loss = compute_loss_on_batch(
                self.model, Xb, yb, self.loss_type, self.l2_reg
            )
            loss.backward()

            # SGD parameter update
            with torch.no_grad():
                for p in self.params:
                    if p.grad is not None:
                        p.data.add_(p.grad, alpha=-self.lr)

            total_loss += loss.item()
            n_batches += 1

        return {"avg_loss": total_loss / max(n_batches, 1)}
