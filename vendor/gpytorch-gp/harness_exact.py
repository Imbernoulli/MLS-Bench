#!/usr/bin/env python3
"""Generic ExactGP harness for the gp-* MLS-Bench tasks that edit ONE surface of an
otherwise-fixed exact Gaussian-process regressor.

Reused by several tasks; each task fixes a different `--surface` and points at a
different agent-editable solution file. The FIXED skeleton is:

    ExactGP( mean_module, covar_module, likelihood ), Adam(lr, iters) on the exact
    marginal log-likelihood, trained on a standardized fixed regression split, then
    scored on held-out test NLL (per point, original y scale, lower better) + RMSE.

The literal plan controls exactly one of mean family, observation-noise policy, or
learning rate. Trusted builders construct all modules and the optimizer; agent
Python never executes. Invalid plans terminate the command.

Emits one metric line:
    GP_METRICS protocol=openml_full_v2 ... nll=<N> rmse=<R> elapsed=<S>
"""
from __future__ import annotations

import argparse
import time

import gpytorch
import torch

import common


def _default_covar(train_x, train_y, d):
    return gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d))


def _default_mean(train_x, train_y, d):
    return gpytorch.means.ConstantMean()


def _default_likelihood(train_x, train_y, d):
    return gpytorch.likelihoods.GaussianLikelihood()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--dataset", default="concrete")
    ap.add_argument("--solution", required=True)
    ap.add_argument(
        "--surface",
        required=True,
        choices=["mean_function", "likelihood_noise", "exact_lr"],
        help="which module the agent's solution controls; the rest are fixed defaults",
    )
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    common.validate_run_contract(
        args.task, args.surface, args.dataset, "iterations", args.iters, args.seed
    )
    if args.lr != 0.1:
        raise ValueError("the ExactGP protocol requires the fixed CLI lr=0.1")
    common.install_strict_numerical_warnings()
    common.set_seeds(args.seed)
    t0 = time.time()

    train_x, train_y, test_x, test_y, std, d = common.load_split(args.dataset)
    dev = common.device()

    config = common.load_surface_config(args.solution)
    covar = _default_covar(train_x, train_y, d)
    if args.surface == "mean_function":
        common.validate_surface_config("mean_function", config)
        mean = common.build_mean_from_config(config, d)
    else:
        mean = _default_mean(train_x, train_y, d)
    if args.surface == "likelihood_noise":
        likelihood = common.build_likelihood_from_config(config)
    else:
        likelihood = _default_likelihood(train_x, train_y, d)
    common.require_finite_module(covar, "covariance")
    common.require_finite_module(mean, "mean")
    common.require_finite_module(likelihood, "likelihood")

    likelihood = likelihood.to(dev)
    model = ExactGPModel(train_x, train_y, likelihood, mean, covar).to(dev)

    # Learning-rate surface: the agent controls the Adam lr; everything else fixed.
    lr = args.lr
    if args.surface == "exact_lr":
        common.validate_surface_config("exact_lr", config)
        lr = common.validate_learning_rate(config["learning_rate"])
    else:
        lr = common.validate_learning_rate(lr)
    print(f"LR lr={lr}", flush=True)

    with common.exact_gp_numerics(args.dataset):
        model.train(); likelihood.train()

        opt = torch.optim.Adam(model.parameters(), lr=lr)
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
