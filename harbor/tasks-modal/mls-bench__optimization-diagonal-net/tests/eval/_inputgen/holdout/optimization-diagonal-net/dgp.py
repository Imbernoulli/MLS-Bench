"""Held-out DGP module for optimization-diagonal-net (NOT agent-visible).

This module lives OUTSIDE every path that is bind-mounted into the agent's
container (workspace package dir, task dir, data binds). It is imported only by
the input pre-generator in ``edits/mid_edit.py`` (host-side) — never by the
agent-editable ``custom_optimizer.py`` or the FIXED ``fixed_benchmark.py``
driver that runs inside the container. It holds the data-generating process
that also produces the true sparse vector ``w_star``, so the agent's optimizer
can never import the generator or reach the ground truth.

``generate_problem`` is byte-identical to the original that used to live in
``edits/fixed_benchmark.py`` — so the data (and therefore every honest result)
is reproduced exactly. ``gen_input`` returns ONLY the observable problem
(X_train, y_train, X_test, y_test) that the optimizer legitimately needs;
``truth`` returns the held-out ground truth ``w_star``.
"""

from __future__ import annotations

import numpy as np
import torch


# ===================================================================
# Data generation (byte-identical to the original fixed_benchmark.py)
# ===================================================================

def generate_problem(
    dim: int,
    sparsity: int,
    n_max_train: int,
    n_test: int,
    seed: int,
):
    """Generate a clean sparse-recovery dataset (no label noise).

    Labels are y = X @ w_star exactly.  Per-step Rademacher noise (±delta)
    is added during training, not here.

    The dataset is generated at maximum size ``n_max_train`` so that any
    n_train <= n_max_train can reuse the first n_train rows without
    re-generating data.

    Returns:
        (X_train, y_train, X_test, y_test, w_star) as torch.Tensor (float64)
    """
    rng = np.random.RandomState(seed)

    # --- ground-truth sparse vector ---
    support = rng.choice(dim, size=sparsity, replace=False)
    w_star_np = np.zeros(dim, dtype=np.float64)
    signs = rng.choice([-1.0, 1.0], size=sparsity)
    w_star_np[support] = signs

    # --- test data (generated first for stability) ---
    X_test_np = (2 * rng.randint(0, 2, size=(n_test, dim)) - 1).astype(np.float64)
    y_test_np = X_test_np @ w_star_np           # clean labels

    # --- training data (max size, clean labels) ---
    X_train_np = (2 * rng.randint(0, 2, size=(n_max_train, dim)) - 1).astype(np.float64)
    y_train_np = X_train_np @ w_star_np         # clean labels

    # Convert to torch tensors
    X_train = torch.from_numpy(X_train_np)
    y_train = torch.from_numpy(y_train_np)
    X_test = torch.from_numpy(X_test_np)
    y_test = torch.from_numpy(y_test_np)
    w_star = torch.from_numpy(w_star_np)

    return X_train, y_train, X_test, y_test, w_star


# ===================================================================
# Helpers for the input pre-generator (mid_edit)
# ===================================================================

def gen_input(dim, sparsity, n_max_train, n_test, seed):
    """Agent-visible input: the observable problem only.

    Returns (X_train, y_train, X_test, y_test) as numpy float64 arrays. The true
    sparse vector w_star is computed internally by the DGP but deliberately NOT
    returned here, so the pre-generator that writes the agent's input file never
    persists the answer.
    """
    X_train, y_train, X_test, y_test, _w_star = generate_problem(
        int(dim), int(sparsity), int(n_max_train), int(n_test), int(seed),
    )
    return (
        X_train.cpu().numpy(),
        y_train.cpu().numpy(),
        X_test.cpu().numpy(),
        y_test.cpu().numpy(),
    )


def truth(dim, sparsity, n_max_train, n_test, seed):
    """Held-out ground truth: the true sparse vector w_star (numpy float64)."""
    *_rest, w_star = generate_problem(
        int(dim), int(sparsity), int(n_max_train), int(n_test), int(seed),
    )
    return w_star.cpu().numpy()
