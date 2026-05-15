"""Variance Reduction Benchmark for Finite-Sum Optimization

Evaluates variance reduction strategies for stochastic gradient methods on
finite-sum problems:  min_x  F(x) = (1/n) * sum_{i=1}^{n} f_i(x)

Benchmarks:
  1. logistic  -- L2-regularized logistic regression on MNIST (convex)
  2. mlp       -- 2-layer MLP on CIFAR-10 (non-convex)
  3. conditioned -- L2-regularized linear regression on synthetic
                    ill-conditioned data (strongly convex)

Usage:
  python opt-vr-bench/custom_vr.py --problem <name> \
      --seed $SEED --output-dir $OUTPUT_DIR
"""

import argparse
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


# ============================================================================
# FIXED -- Utilities
# ============================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# FIXED -- Model Definitions
# ============================================================================

class LogisticRegression(nn.Module):
    """Multinomial logistic regression for MNIST (convex with L2 reg)."""
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


class SmallMLP(nn.Module):
    """2-layer MLP for CIFAR-10 (non-convex)."""
    def __init__(self, input_dim=3072, hidden_dim=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class LinearModel(nn.Module):
    """Linear model for regression (strongly convex with L2 reg)."""
    def __init__(self, input_dim=50):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# ============================================================================
# FIXED -- Data Loading
# ============================================================================

def get_mnist_dataset(data_dir=os.environ.get("DATA_ROOT", "/data") + "/mnist"):
    """Return MNIST train/test as (X, y) tensors."""
    import torchvision
    import torchvision.transforms as T
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_ds = torchvision.datasets.MNIST(data_dir, train=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(data_dir, train=False, transform=transform)
    # Convert to tensors for finite-sum access
    X_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    y_train = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
    X_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    y_test = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
    return X_train, y_train, X_test, y_test


def get_cifar10_dataset(data_dir=os.environ.get("DATA_ROOT", "/data") + "/cifar"):
    """Return CIFAR-10 train/test as (X, y) tensors."""
    import torchvision
    import torchvision.transforms as T
    transform = T.Compose([T.ToTensor(), T.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_ds = torchvision.datasets.CIFAR10(data_dir, train=True, transform=transform)
    test_ds = torchvision.datasets.CIFAR10(data_dir, train=False, transform=transform)
    X_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    y_train = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
    X_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    y_test = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
    return X_train, y_train, X_test, y_test


def get_conditioned_dataset(n_train=10000, n_test=2000, dim=50,
                            condition_number=100, seed=0):
    """Return synthetic ill-conditioned regression data."""
    rng = np.random.RandomState(seed)
    # Create design matrix with specified condition number
    U, _, _ = np.linalg.svd(rng.randn(dim, dim), full_matrices=True)
    singular_values = np.logspace(0, np.log10(condition_number), dim)
    A = U @ np.diag(singular_values) @ U.T
    # Generate data: y = X @ w_true + noise
    w_true = rng.randn(dim, 1)
    X_all = rng.randn(n_train + n_test, dim) @ A
    y_all = X_all @ w_true + 0.1 * rng.randn(n_train + n_test, 1)
    X_train = torch.tensor(X_all[:n_train], dtype=torch.float32)
    y_train = torch.tensor(y_all[:n_train], dtype=torch.float32)
    X_test = torch.tensor(X_all[n_train:], dtype=torch.float32)
    y_test = torch.tensor(y_all[n_train:], dtype=torch.float32)
    return X_train, y_train, X_test, y_test


# ============================================================================
# FIXED -- Problem Configurations
# ============================================================================

PROBLEM_CONFIGS = {
    "logistic": {
        "lr": 0.1,
        "l2_reg": 1e-4,
        "n_epochs": 20,
        "batch_size": 128,
        "eval_interval": 1,
        "target_metric": "test_accuracy",
        "higher_is_better": True,
        "loss_type": "cross_entropy",
    },
    "mlp": {
        "lr": 0.05,
        "l2_reg": 1e-4,
        "n_epochs": 40,
        "batch_size": 128,
        "eval_interval": 2,
        "target_metric": "test_accuracy",
        "higher_is_better": True,
        "loss_type": "cross_entropy",
    },
    "conditioned": {
        "lr": 0.001,
        "l2_reg": 1e-3,
        "n_epochs": 30,
        "batch_size": 128,
        "eval_interval": 1,
        "target_metric": "test_mse",
        "higher_is_better": False,
        "loss_type": "mse",
    },
}


def build_model(problem: str, device: torch.device) -> nn.Module:
    """Instantiate the model for a given problem."""
    if problem == "logistic":
        return LogisticRegression(input_dim=784, num_classes=10).to(device)
    elif problem == "mlp":
        return SmallMLP(input_dim=3072, hidden_dim=256, num_classes=10).to(device)
    elif problem == "conditioned":
        return LinearModel(input_dim=50).to(device)
    else:
        raise ValueError(f"Unknown problem: {problem}")


def get_data(problem: str, seed: int):
    """Return (X_train, y_train, X_test, y_test) for a problem."""
    if problem == "logistic":
        return get_mnist_dataset()
    elif problem == "mlp":
        return get_cifar10_dataset()
    elif problem == "conditioned":
        return get_conditioned_dataset(seed=seed)
    else:
        raise ValueError(f"Unknown problem: {problem}")


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


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_type: str, l2_reg: float, device: torch.device,
             batch_size: int = 512) -> dict:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for start in range(0, X.size(0), batch_size):
        end = min(start + batch_size, X.size(0))
        Xb = X[start:end].to(device)
        yb = y[start:end].to(device)
        pred = model(Xb)
        if loss_type == "cross_entropy":
            total_loss += F.cross_entropy(pred, yb, reduction='sum').item()
            correct += (pred.argmax(dim=1) == yb).sum().item()
        elif loss_type == "mse":
            total_loss += F.mse_loss(pred, yb, reduction='sum').item()
        total += yb.size(0)
    model.train()
    result = {"test_loss": total_loss / total}
    if loss_type == "cross_entropy":
        result["test_accuracy"] = 100.0 * correct / total
    elif loss_type == "mse":
        result["test_mse"] = total_loss / total
    return result


# ============================================================================
# EDITABLE -- Variance Reduction Strategy (lines 286-370)
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


# ============================================================================
# FIXED -- Training Driver
# ============================================================================

def train_problem(problem: str, seed: int, output_dir: str):
    """Train on a single problem and report metrics."""
    cfg = PROBLEM_CONFIGS[problem]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Problem: {problem} | Seed: {seed} | Device: {device} ===",
          flush=True)

    # Build model
    model = build_model(problem, device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}", flush=True)

    # Load data
    X_train, y_train, X_test, y_test = get_data(problem, seed)
    n_train = X_train.size(0)
    print(f"Training samples: {n_train}", flush=True)

    # Create variance reduction optimizer
    optimizer = VarianceReductionOptimizer(
        model=model,
        lr=cfg["lr"],
        l2_reg=cfg["l2_reg"],
        loss_type=cfg["loss_type"],
        n_train=n_train,
        batch_size=cfg["batch_size"],
        device=device,
    )

    # Training loop (epoch-based)
    best_metric = None
    total_grad_comps = 0  # Track gradient computation cost

    for epoch in range(1, cfg["n_epochs"] + 1):
        t0 = time.time()
        train_info = optimizer.train_one_epoch(X_train, y_train)
        epoch_time = time.time() - t0

        # Count approximate gradient computations
        # One epoch of SGD: n/batch_size mini-batch gradient computations
        # Full gradient: equivalent to n/batch_size computations
        n_sgd_steps = math.ceil(n_train / cfg["batch_size"])
        total_grad_comps += n_sgd_steps

        avg_loss = train_info.get("avg_loss", 0.0)
        extra_full_grads = train_info.get("full_grad_count", 0)
        total_grad_comps += extra_full_grads * n_sgd_steps

        print(f"TRAIN_METRICS: epoch={epoch} avg_loss={avg_loss:.6f} "
              f"time={epoch_time:.2f}s grad_comps={total_grad_comps}",
              flush=True)

        # Evaluation
        if epoch % cfg["eval_interval"] == 0 or epoch == cfg["n_epochs"]:
            metrics = evaluate(model, X_test, y_test, cfg["loss_type"],
                               cfg["l2_reg"], device)
            metric_val = metrics[cfg["target_metric"]]

            if best_metric is None:
                best_metric = metric_val
            elif cfg["higher_is_better"]:
                best_metric = max(best_metric, metric_val)
            else:
                best_metric = min(best_metric, metric_val)

            metric_str = " ".join(f"{k}={v:.6f}" for k, v in metrics.items())
            print(f"EVAL_METRICS: epoch={epoch} {metric_str} "
                  f"best_{cfg['target_metric']}={best_metric:.6f}",
                  flush=True)

    # Final reporting
    final_metrics = evaluate(model, X_test, y_test, cfg["loss_type"],
                             cfg["l2_reg"], device)
    final_val = final_metrics[cfg["target_metric"]]

    print(f"TEST_METRICS: "
          f"best_{cfg['target_metric']}={best_metric:.6f} "
          f"final_{cfg['target_metric']}={final_val:.6f} "
          f"total_grad_comps={total_grad_comps}",
          flush=True)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "problem": problem,
        "seed": seed,
        "best_metric": best_metric,
        "final_metric": final_val,
        "target_metric": cfg["target_metric"],
        "total_grad_comps": total_grad_comps,
    }
    torch.save(result, os.path.join(output_dir, f"result_{problem}.pt"))


# ============================================================================
# FIXED -- Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Variance Reduction Benchmark for Finite-Sum Optimization")
    parser.add_argument("--problem", type=str, required=True,
                        choices=["logistic", "mlp", "conditioned"],
                        help="Problem to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Directory to save results")
    args = parser.parse_args()
    train_problem(args.problem, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
