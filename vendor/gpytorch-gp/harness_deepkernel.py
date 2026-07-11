#!/usr/bin/env python3
"""gp-deep-kernel harness (fixed pipeline).

Trains a Deep Kernel Learning GP on a FIXED medium regression split (kin8nm,
8192x8): a trusted, literal-plan-selected feature extractor maps inputs to a small
feature space, then a fixed ScaleKernel(RBFKernel(ard)) GP is placed on top and
trained jointly under the exact marginal log-likelihood. The editable file contains
only a bounded architecture plan; agent Python never executes. Everything else is fixed.

Invalid feature-extractor or width surfaces terminate the command.

Scores held-out test NLL (original y scale, lower better) + RMSE.

Emits one metric line:
    GP_METRICS protocol=openml_full_v2 ... nll=<N> rmse=<R> elapsed=<S>
"""
from __future__ import annotations

import argparse
import time

import gpytorch
import torch
import torch.nn as nn

import common


def _fixed_width_extractor(d: int, p: int) -> nn.Module:
    if isinstance(p, bool) or not isinstance(p, int) or not 1 <= p <= 32:
        raise ValueError("select_width must return an integer in [1, 32]")
    net = nn.Sequential(
        nn.Linear(d, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, p),
    )
    net.out_features = p
    return net


class DKLModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor, p):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=p)
        )

    def forward(self, x):
        z = self.feature_extractor(x)
        z = self.scale_to_bounds(z)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(z), self.covar_module(z)
        )


def _resolve_extractor(obj, d):
    """Accept an nn.Module (with .out_features) or a (module, p) tuple."""
    if isinstance(obj, tuple) and len(obj) == 2:
        module, p = obj
    else:
        module = obj
        p = getattr(module, "out_features", None)
        if p is None:
            # infer by a dry forward on a dummy batch
            with torch.no_grad():
                dummy = torch.zeros(2, d)
                p = module(dummy).size(-1)
    if not isinstance(module, nn.Module):
        raise TypeError("feature extractor must be an nn.Module")
    if isinstance(p, bool) or not isinstance(p, int) or p < 1:
        raise ValueError("feature extractor output dimension must be a positive integer")
    with torch.no_grad():
        probe = module(torch.zeros(2, d))
    common.require_finite_tensor(probe, "feature extractor probe output")
    if probe.ndim != 2 or probe.shape != (2, p):
        raise RuntimeError(f"feature extractor must map [2, {d}] to [2, {p}]")
    common.require_finite_module(module, "feature extractor")
    return module, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--dataset", default="kin8nm")
    ap.add_argument("--solution", required=True)
    ap.add_argument(
        "--surface",
        required=True,
        choices=["deep_kernel", "deep_kernel_width"],
    )
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    common.validate_run_contract(
        args.task, args.surface, args.dataset, "iterations", args.iters, args.seed
    )
    if args.lr != 0.01:
        raise ValueError("the deep-kernel protocol requires Adam lr=0.01")
    common.install_strict_numerical_warnings()
    common.set_seeds(args.seed)
    t0 = time.time()

    train_x, train_y, test_x, test_y, std, d = common.load_split(args.dataset)
    dev = common.device()

    config = common.load_surface_config(args.solution)
    if args.surface == "deep_kernel_width":
        common.validate_surface_config("deep_kernel_width", config)
        fe, p = _resolve_extractor(_fixed_width_extractor(d, config["width"]), d)
    else:
        fe, p = _resolve_extractor(common.build_feature_extractor(config, d), d)
    fe = fe.to(dev)
    print(f"FEATURES p={p}", flush=True)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dev)
    model = DKLModel(train_x, train_y, likelihood, fe, p).to(dev)

    with common.exact_gp_numerics(args.dataset):
        model.train(); likelihood.train()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        completed_updates = 0
        for i in range(args.iters):
            opt.zero_grad()
            out = model(train_x)
            loss = -mll(out, train_y)
            if loss.numel() != 1 or not torch.isfinite(loss).item():
                raise RuntimeError(f"training loss is non-finite at iteration {i + 1}")
            loss.backward()
            common.require_finite_gradients(model)
            common.require_finite_gradients(likelihood)
            opt.step()
            completed_updates = i + 1
            common.require_finite_module(model, "DKL model")
            common.require_finite_module(likelihood, "DKL likelihood")
            if (i + 1) % 30 == 0:
                print(f"TRAIN it={i+1} loss={loss.item():.4f}", flush=True)

        model.eval(); likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(test_x))
            nll, rmse = common.score(pred, test_y, std)

    dt = time.time() - t0
    print(
        common.format_completion_line(
            args.task, args.surface, args.dataset,
            train_x.size(0), test_x.size(0),
            "iterations", args.iters, train_x.size(0), completed_updates, args.seed,
        ),
        flush=True,
    )
    print(
        common.format_metric_line(
            args.task, args.surface, args.dataset,
            train_x.size(0), test_x.size(0),
            "iterations", args.iters, train_x.size(0), completed_updates,
            args.seed, nll, rmse, dt,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
