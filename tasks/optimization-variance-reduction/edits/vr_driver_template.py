"""FIXED driver for the opt-variance-reduction benchmark (do not edit).

Holds the per-problem model definitions, configs, the scoring (``evaluate``)
and the training loop (``train_problem`` / ``main``). The data-generating
process — including the synthetic ground-truth ``w_true`` and the test labels —
lives in a host-only module the agent's process cannot import; this driver only
loads the PRE-GENERATED observable data (X_train, y_train, X_test, y_test)
written into opt-vr-bench/_inputs/ by the task scaffold.

The editable optimizer in ``custom_vr.py`` receives only X_train / y_train via
``train_one_epoch`` and the model it is asked to fit; it can neither name nor
recompute the eval datasets, read the test set, nor call ``evaluate``. The
best_/final_ metrics are trajectory quantities (the best test score over the
epoch schedule) and are reported by this driver — identical to the pre-fix
pipeline.
"""

import argparse
import base64
import io
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_vr import VarianceReductionOptimizer


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
# Public-image datasets (MNIST/CIFAR-10) are read from the baked /data dir; the
# synthetic 'conditioned' problem — whose generator builds the ground-truth
# w_true — is read from the PRE-GENERATED observable arrays in _inputs/, so its
# generator and w_true never enter the container. None of these functions are in
# the agent-editable custom_vr.py scope.

def _inputs_dir():
    d = os.environ.get("VR_INPUTS_DIR")
    if d:
        return d
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_inputs")


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


def _load_pregenerated(problem: str, seed: int):
    """Load PRE-GENERATED (X_train, y_train, X_test, y_test) for a problem.

    The generator (incl. the synthetic ground-truth w_true) lives host-only; here
    we only read the observable arrays written into _inputs/ by the scaffold.
    """
    path = os.path.join(_inputs_dir(), f"{problem}_seed{seed}.npz.b64")
    with open(path, "r") as f:
        raw = base64.b64decode(f.read())
    with np.load(io.BytesIO(raw)) as data:
        X_train = torch.from_numpy(np.ascontiguousarray(data["X_train"]))
        y_train = torch.from_numpy(np.ascontiguousarray(data["y_train"]))
        X_test = torch.from_numpy(np.ascontiguousarray(data["X_test"]))
        y_test = torch.from_numpy(np.ascontiguousarray(data["y_test"]))
    return X_train, y_train, X_test, y_test


def get_data(problem: str, seed: int):
    """Return (X_train, y_train, X_test, y_test) for a problem."""
    if problem == "logistic":
        return get_mnist_dataset()
    elif problem == "mlp":
        return get_cifar10_dataset()
    elif problem == "conditioned":
        return _load_pregenerated(problem, seed)
    else:
        raise ValueError(f"Unknown problem: {problem}")


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
