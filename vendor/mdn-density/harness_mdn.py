#!/usr/bin/env python3
"""Shared MDN harness with task-specific, single-axis solution surfaces.

The solution returns one validated scalar or enum. Frozen code constructs the
complete model, preventing an activation task, for example, from also changing
the component count, covariance, width, or optimizer.

Emits a hash-bound completion followed by one terminal metric line.
"""
from __future__ import annotations

import argparse
import math
import time

import torch.nn as nn

import common
import mdn_blocks as mb


_SURFACES = (
    "activation",
    "covariance",
    "density_family",
    "initial_sigma",
    "learning_rate",
    "network_width",
    "num_components",
    "component_balance",
    "trunk_depth",
    "variance_floor",
)

_TASK_PROTOCOLS = {
    "mdn-activation": ("activation", {"spiral"}),
    "mdn-component-balance": ("component_balance", {"spiral"}),
    "mdn-covariance": ("covariance", {"rot_bimodal"}),
    "mdn-density-bench": (
        "density_family",
        {"inverse_sine", "two_branch", "spiral"},
    ),
    "mdn-initialization": ("initial_sigma", {"spiral"}),
    "mdn-learning-rate": ("learning_rate", {"inverse_sine"}),
    "mdn-network-width": ("network_width", {"spiral"}),
    "mdn-num-components": ("num_components", {"inverse_sine"}),
    "mdn-trunk-depth": ("trunk_depth", {"spiral"}),
    "mdn-variance-floor": ("variance_floor", {"spiral"}),
}


def _require_choice(value, *, label: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{label} must be one of {sorted(choices)}")
    return value


def _require_int(value, *, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{label} must be in [{low}, {high}]")
    return value


def _require_float(value, *, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{label} must be finite and in [{low}, {high}]")
    return result


def _require_keys(config: dict, expected: set[str]) -> None:
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError(f"surface_config requires exactly {sorted(expected)}")


def _build_from_surface(solution: str, surface: str):
    """Parse one literal design axis and construct all modules in trusted code."""
    config = common.load_surface_config(solution)

    selected_lr = None
    component_balance_weight = 0.0
    if surface == "activation":
        _require_keys(config, {"activation"})
        activation = _require_choice(
            config["activation"], label="activation",
            choices={"elu", "gelu", "relu", "sigmoid", "tanh"},
        )
        model = mb.mdn(k=6, hidden=32, act=activation, var_mode="exp", sigma_init=0.3)
    elif surface == "covariance":
        _require_keys(config, {"covariance"})
        covariance = _require_choice(
            config["covariance"], label="covariance", choices={"diag", "full"},
        )
        model = mb.mdn2d(k=2, cov=covariance, hidden=64)
    elif surface == "density_family":
        _require_keys(config, {"density_family"})
        family = _require_choice(
            config["density_family"], label="density family",
            choices={"point", "single_gaussian", "mixture"},
        )
        if family == "point":
            model = mb.PointRegressor(hidden=64, sigma=0.3)
        elif family == "single_gaussian":
            model = mb.single_gaussian(hidden=64, var_mode="exp")
        else:
            model = mb.mdn(k=5, hidden=64, var_mode="softplus", sigma_init=0.3)
    elif surface == "initial_sigma":
        _require_keys(config, {"initial_sigma"})
        sigma = _require_float(
            config["initial_sigma"], label="initial sigma", low=1e-3, high=10.0,
        )
        model = mb.mdn(k=6, hidden=64, var_mode="exp", sigma_init=sigma)
    elif surface == "learning_rate":
        _require_keys(config, {"learning_rate"})
        selected_lr = _require_float(
            config["learning_rate"], label="learning rate", low=1e-5, high=1e-1,
        )
        model = mb.mdn(k=5, hidden=64, var_mode="exp", sigma_init=0.3)
    elif surface == "network_width":
        _require_keys(config, {"network_width"})
        width = _require_int(
            config["network_width"], label="network width", low=2, high=128,
        )
        model = mb.mdn(k=6, hidden=width, var_mode="exp", sigma_init=0.3)
    elif surface == "num_components":
        _require_keys(config, {"num_components"})
        components = _require_int(
            config["num_components"], label="number of components", low=1, high=16,
        )
        model = mb.mdn(k=components, hidden=64, var_mode="exp", sigma_init=0.3)
    elif surface == "component_balance":
        _require_keys(config, {"component_balance_weight"})
        component_balance_weight = _require_float(
            config["component_balance_weight"],
            label="component balance weight",
            low=0.0,
            high=1.0,
        )
        model = mb.mdn(
            k=6,
            hidden=64,
            var_mode="exp",
            sigma_init=0.3,
        )
    elif surface == "trunk_depth":
        _require_keys(config, {"trunk_depth"})
        depth = _require_int(
            config["trunk_depth"], label="trunk depth", low=1, high=4,
        )
        model = mb.mdn(
            k=6, hidden=64, depth=depth, var_mode="exp", sigma_init=0.3,
        )
    elif surface == "variance_floor":
        _require_keys(config, {"variance_floor"})
        floor = _require_float(
            config["variance_floor"], label="variance floor", low=0.0, high=1.0,
        )
        model = mb.mdn(
            k=6, hidden=64, var_mode="exp", sigma_eps=floor, sigma_init=0.3,
        )
    else:  # argparse prevents this; direct callers also fail closed.
        raise ValueError(f"unknown MDN surface {surface!r}")

    if not isinstance(model, nn.Module):
        raise TypeError("frozen MDN builder produced a non-module")
    return model, selected_lr, component_balance_weight


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=tuple(_TASK_PROTOCOLS))
    ap.add_argument("--solution", required=True)
    ap.add_argument("--surface", required=True, choices=_SURFACES)
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=20000)
    args = ap.parse_args()

    expected_surface, expected_targets = _TASK_PROTOCOLS[args.task]
    if args.surface != expected_surface or args.target not in expected_targets:
        raise SystemExit(
            f"task {args.task!r} requires surface={expected_surface!r} and "
            f"target in {sorted(expected_targets)!r}"
        )
    is_2d = args.target in common.TARGETS_2D
    if not is_2d and args.target not in common.TARGETS:
        raise SystemExit(
            f"unknown target {args.target!r}; choose from "
            f"{list(common.TARGETS) + list(common.TARGETS_2D)}")
    if (
        args.seed != 42
        or args.steps != 4000
        or args.batch_size != 512
        or args.n_train != 20000
        or args.n_test != 20000
    ):
        raise SystemExit(
            "MDN full protocol requires seed=42, steps=4000, batch_size=512, "
            "n_train=20000, and n_test=20000"
        )

    common.require_cuda_device()
    common.set_seeds(args.seed)
    t0 = time.time()

    model, selected_lr, component_balance_weight = _build_from_surface(
        args.solution, args.surface
    )
    # Sanity: must be a real torch.nn.Module, not a duck-typed stand-in that
    # merely exposes .to()/.parameters()/.train()/.eval()/__call__. Closes a
    # bypass where a hand-built object could report NLL without a real forward pass
    # without ever running a real forward/backward pass through trainable
    # tensors (e.g. a plain class holding a decorative, unused nn.Parameter
    # purely so Adam/backward() don't error out).
    if not isinstance(model, nn.Module):
        raise TypeError("build_mdn must return a real torch.nn.Module instance")
    n_par = common.n_params(model)
    k = getattr(model, "k", "?")

    lr = args.lr if selected_lr is None else selected_lr
    print(
        f"MDN_DESIGN target={args.target} K={k} params={n_par} lr={lr:g} "
        f"component_balance_weight={component_balance_weight:g}",
        flush=True,
    )

    train_sha256, test_sha256 = common.data_proof(args.target, args.seed)
    if is_2d:
        x_tr, y_tr, x_te, y_te = common.make_dataset_2d(
            args.target, args.n_train, args.n_test, args.seed)
        print(
            f"MDN_PROTOCOL protocol=mdn_full_v3 task={args.task} "
            f"surface={args.surface} target={args.target} seed={args.seed} "
            f"steps={args.steps} batch_size={args.batch_size} "
            f"n_train={x_tr.shape[0]} n_test={x_te.shape[0]} "
            f"train_sha256={train_sha256} test_sha256={test_sha256} "
            f"device={common.DEVICE}",
            flush=True,
        )
        nll = common.train_and_eval_2d(
            model, x_tr, y_tr, x_te, y_te,
            steps=args.steps, batch_size=args.batch_size, lr=lr, seed=args.seed)
    else:
        x_tr, y_tr, x_te, y_te = common.make_dataset(
            args.target, args.n_train, args.n_test, args.seed)
        print(
            f"MDN_PROTOCOL protocol=mdn_full_v3 task={args.task} "
            f"surface={args.surface} target={args.target} seed={args.seed} "
            f"steps={args.steps} batch_size={args.batch_size} "
            f"n_train={x_tr.shape[0]} n_test={x_te.shape[0]} "
            f"train_sha256={train_sha256} test_sha256={test_sha256} "
            f"device={common.DEVICE}",
            flush=True,
        )
        nll = common.train_and_eval(
            model, x_tr, y_tr, x_te, y_te,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=lr,
            seed=args.seed,
            component_balance_weight=component_balance_weight,
        )

    dt = time.time() - t0
    if not math.isfinite(nll) or not math.isfinite(dt) or dt < 0:
        raise RuntimeError("MDN evaluation produced invalid final metrics")
    print(
        f"MDN_COMPLETE protocol=mdn_full_v3 task={args.task} "
        f"surface={args.surface} target={args.target} seed={args.seed} "
        f"steps={args.steps} final_step={args.steps - 1} "
        f"batch_size={args.batch_size} n_train={x_tr.shape[0]} n_test={x_te.shape[0]} "
        f"train_sha256={train_sha256} test_sha256={test_sha256}",
        flush=True,
    )
    print(
        f"MDN_METRICS protocol=mdn_full_v3 task={args.task} "
        f"surface={args.surface} target={args.target} seed={args.seed} "
        f"steps={args.steps} train_sha256={train_sha256} test_sha256={test_sha256} "
        f"nll={nll:.6f} params={n_par} elapsed={dt:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
