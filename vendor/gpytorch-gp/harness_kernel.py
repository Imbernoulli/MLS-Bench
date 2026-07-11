#!/usr/bin/env python3
"""gp-kernel-design harness (fixed pipeline).

Builds an ExactGP on a FIXED regression split (concrete, 1030x8), using the
trusted covariance and mean modules selected by a literal plan, trains Type-II MLE for a fixed budget,
and scores held-out test NLL (per point, original y scale, lower better) + RMSE.

Depending on ``--surface``, the literal plan controls a covariance/mean family or
one validated kernel-axis choice. The ExactGP wrapper, optimizer, iteration count,
likelihood, dataset, split, standardization, and NLL/RMSE path are fixed here.
Agent-authored Python never executes. Invalid plans terminate the command.

Emits one metric line:
    GP_METRICS protocol=openml_full_v2 ... nll=<N> rmse=<R> elapsed=<S>
"""
from __future__ import annotations

import argparse
import time

import gpytorch
import torch

import common


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def _default_covar(train_x, train_y, d):
    return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())


def _default_mean(d):
    return gpytorch.means.ConstantMean()


def _axis_modules(surface, solution, train_x, train_y, d):
    config = common.load_surface_config(solution)
    if surface == "kernel_design":
        covar, mean = common.build_kernel_design(config, train_x, train_y, d)
    elif surface == "ard":
        common.validate_surface_config("ard", config)
        use_ard = config["ard"]
        ard_dims = d if use_ard else None
        covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)
        )
        mean = _default_mean(d)
    elif surface == "smoothness":
        common.validate_surface_config("smoothness", config)
        choice = config["kernel"]
        if choice == "rbf":
            base = gpytorch.kernels.RBFKernel(ard_num_dims=d)
        elif choice == "matern12":
            base = gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=d)
        elif choice == "matern52":
            base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)
        else:
            raise ValueError("select_kernel must return rbf, matern12, or matern52")
        covar = gpytorch.kernels.ScaleKernel(base)
        mean = _default_mean(d)
    else:
        raise ValueError(f"unsupported kernel surface: {surface!r}")
    if not isinstance(covar, gpytorch.kernels.Kernel):
        raise TypeError("kernel surface must produce a gpytorch Kernel")
    if not isinstance(mean, gpytorch.means.Mean):
        raise TypeError("mean surface must produce a gpytorch Mean")
    common.require_finite_module(covar, "covariance")
    common.require_finite_module(mean, "mean")
    return covar, mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--dataset", default="concrete")
    ap.add_argument("--solution", required=True)
    ap.add_argument(
        "--surface",
        required=True,
        choices=["kernel_design", "ard", "smoothness"],
    )
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    common.validate_run_contract(
        args.task, args.surface, args.dataset, "iterations", args.iters, args.seed
    )
    if args.lr != 0.1:
        raise ValueError("the kernel protocol requires Adam lr=0.1")
    common.install_strict_numerical_warnings()
    common.set_seeds(args.seed)
    t0 = time.time()

    train_x, train_y, test_x, test_y, std, d = common.load_split(args.dataset)

    covar, mean = _axis_modules(
        args.surface, args.solution, train_x, train_y, d
    )

    dev = common.device()
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dev)
    model = ExactGPModel(train_x, train_y, likelihood, mean, covar).to(dev)

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
            common.require_finite_module(model, "GP model")
            common.require_finite_module(likelihood, "GP likelihood")
            if (i + 1) % 50 == 0:
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
