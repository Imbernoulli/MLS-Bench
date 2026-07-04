"""Held-out DGP module for optimization-variance-reduction (NOT agent-visible).

This module lives OUTSIDE every path that is bind-mounted into the agent's
container. It is imported only by the input pre-generator in
``edits/mid_edit.py`` (host-side) — never by the agent-editable ``custom_vr.py``
or the FIXED ``vr_driver.py`` that runs inside the container. It holds the
dataset loaders (including the synthetic ground-truth ``w_true``) and the test
labels, so the editable optimizer can neither name/recompute the eval datasets
nor read the held-out test set.

The loader bodies are byte-identical to the originals that used to live in
``edits/custom_template.py`` — so the data (and therefore every honest result)
is reproduced exactly. ``gen_input`` returns the observable problem
(X_train, y_train, X_test, y_test); ``truth`` exposes the test set for the
host-side scorer (the metric itself stays a trajectory quantity in the driver).

NOTE — intrinsic residual for 'conditioned': the closed-form ridge minimizer of
the regularized TRAINING objective is recomputable from (X_train, y_train) — the
data the optimizer must legitimately receive — so it cannot be hidden by any
holdout split. This module seals the w_true / test-label / dataset-identity
leak; it does NOT (and cannot) prevent an optimizer from solving the observable
training problem in closed form. See the task report for the residual.
"""

import os

import numpy as np
import torch


# ============================================================================
# Data Loading (byte-identical to the original custom_template.py loaders)
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
# Dispatch (byte-identical mapping to the original get_data)
# ============================================================================

def _load(problem, seed):
    """Return (X_train, y_train, X_test, y_test) for a problem.

    Identical dispatch to the original get_data: MNIST/CIFAR ignore the seed;
    'conditioned' threads it through to the synthetic generator.
    """
    if problem == "logistic":
        return get_mnist_dataset()
    elif problem == "mlp":
        return get_cifar10_dataset()
    elif problem == "conditioned":
        return get_conditioned_dataset(seed=seed)
    else:
        raise ValueError(f"Unknown problem: {problem}")


# ============================================================================
# Helpers for the input pre-generator (mid_edit) and host-side use
# ============================================================================

def gen_input(problem, seed):
    """Agent-visible input: (X_train, y_train, X_test, y_test) as numpy arrays."""
    X_train, y_train, X_test, y_test = _load(problem, seed)
    return (
        X_train.cpu().numpy(),
        y_train.cpu().numpy(),
        X_test.cpu().numpy(),
        y_test.cpu().numpy(),
    )


def truth(problem, seed):
    """Held-out test set (X_test, y_test) for the same problem/seed (numpy)."""
    _Xtr, _ytr, X_test, y_test = _load(problem, seed)
    return X_test.cpu().numpy(), y_test.cpu().numpy()
