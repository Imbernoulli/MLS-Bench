"""Fixed benchmark harness for diagonal-net sparse recovery.

Evaluates custom optimizers on the problem of recovering a sparse linear
predictor through a diagonal-net parameterization (w_hat = u^L - v^L, L=2).
The benchmark measures the minimum training-set size n* required for
reliable recovery (test MSE < 1.0) using a coarse-to-fine search protocol.

Metric: -log2(n*) (higher is better — fewer training samples needed).

Uses PyTorch with autograd for gradient computation; the optimizer interface
receives torch.Tensor gradients directly.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================================================================
# Configuration
# ===================================================================

@dataclass(frozen=True)
class ProblemConfig:
    """Immutable descriptor for a sparse-recovery problem setting."""
    dim: int
    sparsity: int
    delta: float = 0.5
    n_test: int = 4096
    alpha_init: float = 1e-3
    eval_batch: int = 1000


@dataclass(frozen=True)
class StoppingConfig:
    """Parameters for the two-window plateau stopping rule."""
    eval_interval: int = 100
    plateau_window: int = 200          # in evaluations (not steps)
    min_steps_before_plateau_check: int = 50_000
    max_steps: int = 1_000_000
    improvement_tol_abs: float = 1e-6
    improvement_tol_rel: float = 1e-3
    band_tol_abs: float = 1e-6
    band_tol_rel: float = 1e-3


@dataclass(frozen=True)
class SearchConfig:
    """Parameters for the coarse-to-fine training-size search."""
    grid: tuple[int, ...] = (50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600)
    num_seeds: int = 5
    success_seeds_required: int = 4
    recovery_threshold: float = 1.0


# ===================================================================
# Diagonal-Net model (PyTorch)
# ===================================================================

class DiagonalNet(nn.Module):
    """Diagonal-net model: w_hat = u^2 - v^2."""

    def __init__(self, dim: int, alpha_init: float):
        super().__init__()
        init_val = (alpha_init / math.sqrt(2 * dim)) * torch.ones(dim, dtype=torch.float64)
        self.u = nn.Parameter(init_val.clone())
        self.v = nn.Parameter(init_val.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.u ** 2 - self.v ** 2
        return x @ w


# ===================================================================
# Data generation
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
# Two-window plateau stopping rule
# ===================================================================

def _check_plateau(
    history: list[float],
    window: int,
    tol_imp_abs: float,
    tol_imp_rel: float,
    tol_band_abs: float,
    tol_band_rel: float,
) -> bool:
    """Return True if the loss history shows plateau behaviour."""
    if len(history) < 2 * window:
        return False
    recent = history[-window:]
    prev = history[-2 * window : -window]
    recent_min, recent_max = min(recent), max(recent)
    prev_min = min(prev)

    # 1. Little improvement from previous to recent window
    improvement = prev_min - recent_min
    if improvement > tol_imp_abs + tol_imp_rel * max(prev_min, 1e-12):
        return False

    # 2. Small oscillation within the recent window
    band = recent_max - recent_min
    if band > tol_band_abs + tol_band_rel * max(recent_min, 1e-12):
        return False

    return True


def _should_stop(
    train_hist: list[float],
    test_hist: list[float],
    step: int,
    cfg: StoppingConfig,
) -> bool:
    """Decide whether training should stop."""
    if step >= cfg.max_steps:
        return True
    if step < cfg.min_steps_before_plateau_check:
        return False
    if len(train_hist) < 2 * cfg.plateau_window:
        return False

    train_ok = _check_plateau(
        train_hist, cfg.plateau_window,
        cfg.improvement_tol_abs, cfg.improvement_tol_rel,
        cfg.band_tol_abs, cfg.band_tol_rel,
    )
    test_ok = _check_plateau(
        test_hist, cfg.plateau_window,
        cfg.improvement_tol_abs, cfg.improvement_tol_rel,
        cfg.band_tol_abs, cfg.band_tol_rel,
    )
    return train_ok and test_ok


# ===================================================================
# Batched MSE evaluation
# ===================================================================

def _mse_batched(
    model: DiagonalNet,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 1000,
) -> float:
    """Compute 0.5 * mean((model(X) - y)^2) in batches to save memory."""
    n = X.shape[0]
    if batch_size <= 0 or batch_size >= n:
        return float(0.5 * torch.mean((model(X) - y) ** 2).item())
    total_sq = 0.0
    for i in range(0, n, batch_size):
        xb = X[i : i + batch_size]
        yb = y[i : i + batch_size]
        total_sq += torch.sum((model(xb) - yb) ** 2).item()
    return 0.5 * total_sq / n


# ===================================================================
# Single training run
# ===================================================================

def _train_single(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    w_star: torch.Tensor,
    dim: int,
    alpha_init: float,
    delta: float,
    init_state_fn,
    step_fn,
    hparams: dict[str, Any],
    model_seed: int,
    stop_cfg: StoppingConfig,
    eval_batch: int = 1000,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one complete training instance and return diagnostics."""
    torch.manual_seed(model_seed)

    model = DiagonalNet(dim, alpha_init).to(DEVICE)
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)

    state = init_state_fn(model.u.data.clone(), model.v.data.clone(), hparams)

    # Move any tensor values in optimizer state to the correct device
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(DEVICE)

    train_hist: list[float] = []
    test_hist: list[float] = []
    final_step = stop_cfg.max_steps

    n_train = X_train.shape[0]

    # RNG for Rademacher noise
    noise_rng = torch.Generator(device=DEVICE)
    noise_rng.manual_seed(model_seed + 2_000_000)

    for t in range(1, stop_cfg.max_steps + 1):
        # Forward pass with fresh per-step Rademacher noise on labels
        model.zero_grad()
        noise = delta * (2.0 * torch.randint(
            0, 2, (n_train,), generator=noise_rng, dtype=torch.float64,
            device=DEVICE,
        ) - 1.0)
        y_noisy = y_train + noise
        loss = 0.5 * torch.mean((model(X_train) - y_noisy) ** 2)
        loss.backward()

        # Optimizer step
        with torch.no_grad():
            u_new, v_new, state = step_fn(
                model.u.data.clone(), model.v.data.clone(),
                model.u.grad.clone(), model.v.grad.clone(),
                state, hparams,
            )
            model.u.data.copy_(u_new)
            model.v.data.copy_(v_new)

        if t % stop_cfg.eval_interval == 0:
            with torch.no_grad():
                # Evaluate on clean labels
                train_mse = _mse_batched(model, X_train, y_train, eval_batch)
                test_mse = _mse_batched(model, X_test, y_test, eval_batch)
            train_hist.append(train_mse)
            test_hist.append(test_mse)

            if verbose and t % 10_000 == 0:
                print(
                    f"    step={t} train_mse={train_mse:.6e} "
                    f"test_mse={test_mse:.6e}",
                    flush=True,
                )

            if _should_stop(train_hist, test_hist, t, stop_cfg):
                final_step = t
                break

            # Early exit: recovery clearly hopeless
            if (
                t >= stop_cfg.min_steps_before_plateau_check
                and test_mse > 5.0
                and train_mse < 1e-3
            ):
                final_step = t
                break

    # --- final metrics ---
    with torch.no_grad():
        final_test_mse = _mse_batched(model, X_test, y_test, eval_batch)
        final_train_mse = _mse_batched(model, X_train, y_train, eval_batch)
        w_hat = (model.u ** 2 - model.v ** 2).cpu().numpy()

    w_star_np = w_star.cpu().numpy()

    # Diagnostic: distance to ground truth
    w_diff = float(np.linalg.norm(w_hat - w_star_np))
    support_true = set(np.nonzero(w_star_np)[0])
    support_hat = set(np.where(np.abs(w_hat) > 0.5)[0])
    tp = len(support_true & support_hat)
    precision = tp / max(len(support_hat), 1)
    recall = tp / max(len(support_true), 1)

    return {
        "success": final_test_mse < 1.0,
        "final_test_mse": float(final_test_mse),
        "final_train_mse": float(final_train_mse),
        "steps": final_step,
        "w_diff": w_diff,
        "support_precision": precision,
        "support_recall": recall,
    }


# ===================================================================
# Training-size search protocol
# ===================================================================

def _evaluate_n_train(
    problem: ProblemConfig,
    n_train: int,
    datasets: dict[int, tuple],
    init_state_fn,
    step_fn,
    hparams: dict[str, Any],
    seeds: list[int],
    stop_cfg: StoppingConfig,
    success_threshold: int = 0,
    verbose: bool = False,
) -> tuple[int, list[dict[str, Any]]]:
    """Evaluate a candidate n_train across several seeds.

    Uses pre-generated datasets (sliced to n_train rows).

    Returns:
        (num_successes, per_seed_results)
    """
    results: list[dict[str, Any]] = []
    successes = 0
    num_seeds = len(seeds)
    threshold = success_threshold if success_threshold > 0 else (num_seeds + 1) // 2
    for seed in seeds:
        remaining = num_seeds - len(results)
        # Early termination: can't reach threshold even if all remaining succeed
        if successes + remaining < threshold:
            break
        X_tr_full, y_tr_full, X_te, y_te, w_star = datasets[seed]
        X_tr = X_tr_full[:n_train]
        y_tr = y_tr_full[:n_train]
        model_seed = seed + 1_000_000
        result = _train_single(
            X_tr, y_tr, X_te, y_te, w_star,
            problem.dim, problem.alpha_init, problem.delta,
            init_state_fn, step_fn, hparams,
            model_seed, stop_cfg, problem.eval_batch, verbose,
        )
        result["seed"] = seed
        results.append(result)
        if result["success"]:
            successes += 1
    return successes, results


def _coarse_to_fine_search(
    problem: ProblemConfig,
    init_state_fn,
    step_fn,
    get_hparams_fn,
    search_cfg: SearchConfig,
    stop_cfg: StoppingConfig,
    master_seed: int = 42,
    verbose: bool = False,
) -> tuple[int, dict[int, dict[str, Any]]]:
    """Find the smallest n_train for reliable recovery.

    Phase 1 — scan the coarse grid until a successful n_train is found.
    Phase 2 — binary search between the last failure and the first success.

    Data is pre-generated once per seed at the maximum grid size.

    Returns:
        (n_star, all_tested)  where all_tested maps n_train -> results.
    """
    grid = list(search_cfg.grid)
    seeds = list(range(master_seed, master_seed + search_cfg.num_seeds))
    threshold = search_cfg.success_seeds_required
    hparams = get_hparams_fn(problem.dim, problem.sparsity, problem.delta)

    # Pre-generate clean datasets for all seeds at max training size
    n_max_train = max(grid)
    datasets: dict[int, tuple] = {}
    print(f"Generating datasets (n_max_train={n_max_train}, n_test={problem.n_test}) ...",
          flush=True)
    for seed in seeds:
        datasets[seed] = generate_problem(
            problem.dim, problem.sparsity, n_max_train, problem.n_test, seed,
        )
    print("Datasets ready.", flush=True)

    all_tested: dict[int, dict[str, Any]] = {}
    first_success_idx: int | None = None

    # --- Phase 1: coarse grid ---
    for i, n_train in enumerate(grid):
        successes, results = _evaluate_n_train(
            problem, n_train, datasets, init_state_fn, step_fn, hparams,
            seeds, stop_cfg, threshold, verbose,
        )
        avg_test = float(np.mean([r["final_test_mse"] for r in results]))
        avg_steps = float(np.mean([r["steps"] for r in results]))
        all_tested[n_train] = {
            "successes": successes,
            "results": results,
        }
        print(
            f"SEARCH_METRICS n_train={n_train} "
            f"successes={successes}/{search_cfg.num_seeds} "
            f"avg_test_mse={avg_test:.4e} avg_steps={avg_steps:.0f}",
            flush=True,
        )
        if successes >= threshold:
            first_success_idx = i
            break

    if first_success_idx is None:
        return grid[-1], all_tested

    if first_success_idx == 0:
        return grid[0], all_tested

    # --- Phase 2: binary search ---
    lo = grid[first_success_idx - 1]
    hi = grid[first_success_idx]

    while hi - lo > max(5, int(lo * 0.05)):
        mid = (lo + hi) // 2
        if mid == lo or mid == hi:
            break
        successes, results = _evaluate_n_train(
            problem, mid, datasets, init_state_fn, step_fn, hparams,
            seeds, stop_cfg, threshold, verbose,
        )
        avg_test = float(np.mean([r["final_test_mse"] for r in results]))
        avg_steps = float(np.mean([r["steps"] for r in results]))
        all_tested[mid] = {
            "successes": successes,
            "results": results,
        }
        print(
            f"SEARCH_METRICS n_train={mid} "
            f"successes={successes}/{search_cfg.num_seeds} "
            f"avg_test_mse={avg_test:.4e} avg_steps={avg_steps:.0f}",
            flush=True,
        )
        if successes >= threshold:
            hi = mid
        else:
            lo = mid

    return hi, all_tested


# ===================================================================
# CLI entry point
# ===================================================================

def run_cli(get_hyperparameters, init_state, step):
    """Parse arguments and run the benchmark for one (d, k) setting."""
    parser = argparse.ArgumentParser(
        description="MLS-Bench opt-diagonal-net benchmark.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--label", type=str, default="eval")
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--sparsity", type=int, required=True)
    parser.add_argument("--sigma", type=float, default=0.0,
                        help="(deprecated, ignored) Use --delta instead.")
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--alpha-init", type=float, default=1e-3)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--eval-batch", type=int, default=1000)
    parser.add_argument("--grid-max", type=int, default=None,
                        help="Override maximum value in the search grid")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    problem = ProblemConfig(
        dim=args.dim,
        sparsity=args.sparsity,
        delta=args.delta,
        n_test=args.n_test,
        alpha_init=args.alpha_init,
        eval_batch=args.eval_batch,
    )

    if args.smoke:
        stop_cfg = StoppingConfig(
            eval_interval=50,
            plateau_window=20,
            min_steps_before_plateau_check=3_000,
            max_steps=10_000,
        )
        search_cfg = SearchConfig(
            grid=(100, 200, 400, 800),
            num_seeds=2,
            success_seeds_required=2,
        )
    else:
        stop_cfg = StoppingConfig()
        search_cfg = SearchConfig()

    # Override grid max if requested
    if args.grid_max is not None:
        base = list(search_cfg.grid)
        extended = [v for v in base if v <= args.grid_max]
        if not extended or extended[-1] < args.grid_max:
            extended.append(args.grid_max)
        search_cfg = SearchConfig(
            grid=tuple(extended),
            num_seeds=search_cfg.num_seeds,
            success_seeds_required=search_cfg.success_seeds_required,
            recovery_threshold=search_cfg.recovery_threshold,
        )

    print(
        f"TASK_CONFIG dim={args.dim} sparsity={args.sparsity} "
        f"delta={args.delta} alpha_init={args.alpha_init} "
        f"n_test={args.n_test} seed={args.seed} smoke={args.smoke}",
        flush=True,
    )

    n_star, all_tested = _coarse_to_fine_search(
        problem, init_state, step, get_hyperparameters,
        search_cfg, stop_cfg,
        master_seed=args.seed, verbose=True,
    )

    score = -math.log2(max(n_star, 1))

    print(
        f"FINAL_METRICS n_star={n_star} score={score:.6f}",
        flush=True,
    )

    # --- save detailed results ---
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, Any] = {
            "problem": asdict(problem),
            "n_star": int(n_star),
            "score": float(score),
            "tested": {},
        }
        for nt, info in sorted(all_tested.items()):
            summary["tested"][str(nt)] = {
                "successes": info["successes"],
                "per_seed": [
                    {
                        k: (float(v) if isinstance(v, (np.floating, float)) else v)
                        for k, v in r.items()
                    }
                    for r in info["results"]
                ],
            }
        out = args.output_dir / f"{args.label}_seed{args.seed}.json"
        out.write_text(json.dumps(summary, indent=2, default=str))
