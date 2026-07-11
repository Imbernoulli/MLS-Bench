#!/usr/bin/env python3
"""gp-sparse-inducing harness (fixed pipeline).

Trains a Stochastic Variational GP (SVGP) on a FIXED medium regression split
(elevators, ~16.6k x 18), where a literal plan controls only the inducing-point
method and count. Trusted code selects the locations; agent Python never executes.
Everything else (ApproximateGP + VariationalStrategy +
CholeskyVariationalDistribution, RBF kernel, ELBO, minibatch loop, epochs,
optimizer) is fixed. Scores held-out test NLL (original y scale, lower better) +
RMSE.

Invalid inducing-point surfaces terminate the command.

Emits one metric line:
    GP_METRICS protocol=openml_full_v2 ... nll=<N> rmse=<R> elapsed=<S>
"""
from __future__ import annotations

import argparse
import time

import gpytorch
import torch
from torch.utils.data import DataLoader, TensorDataset

import common


class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        vd = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        vs = gpytorch.variational.VariationalStrategy(
            self, inducing_points, vd, learn_inducing_locations=True
        )
        super().__init__(vs)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(1))
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--surface", required=True, choices=["inducing"])
    ap.add_argument("--dataset", default="elevators")
    ap.add_argument("--solution", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--max-inducing", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    common.validate_run_contract(
        args.task, args.surface, args.dataset, "epochs", args.epochs, args.seed
    )
    if args.batch_size != 1024 or args.lr != 0.01 or args.max_inducing != 2048:
        raise ValueError(
            "the sparse-GP protocol requires batch_size=1024, lr=0.01, "
            "and max_inducing=2048"
        )
    common.install_strict_numerical_warnings()
    common.set_seeds(args.seed)
    t0 = time.time()

    train_x, train_y, test_x, test_y, std, d = common.load_split(args.dataset)

    config = common.load_surface_config(args.solution)
    ip = common.select_inducing_from_config(config, train_x)
    if not torch.is_tensor(ip) or ip.dim() != 2 or ip.size(1) != d:
        raise TypeError("select_inducing_points must return a [M, d] tensor")
    if not 1 <= ip.size(0) <= args.max_inducing:
        raise ValueError(f"inducing-point count must be in [1, {args.max_inducing}]")
    common.require_finite_tensor(ip, "inducing points")

    ip = ip.to(common.device()).float().contiguous()
    print(f"INDUCING M={ip.size(0)}", flush=True)

    dev = common.device()
    model = SVGPModel(ip).to(dev)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dev)

    model.train(); likelihood.train()
    opt = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}],
        lr=args.lr,
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True
    )
    completed_updates = 0
    for epoch in range(args.epochs):
        last = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = -mll(out, yb)
            if loss.numel() != 1 or not torch.isfinite(loss).item():
                raise RuntimeError(f"training loss is non-finite at epoch {epoch + 1}")
            loss.backward()
            common.require_finite_gradients(model)
            common.require_finite_gradients(likelihood)
            opt.step()
            completed_updates += 1
            common.require_finite_module(model, "SVGP model")
            common.require_finite_module(likelihood, "SVGP likelihood")
            last = loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"TRAIN epoch={epoch+1} loss={last:.4f}", flush=True)

    model.eval(); likelihood.eval()
    means, variances = [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        te_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=4096)
        for xb, _ in te_loader:
            pred = likelihood(model(xb))
            means.append(pred.mean)
            variances.append(pred.variance)

    class _P:  # minimal predictive-dist shim for common.score
        pass

    p = _P()
    p.mean = torch.cat(means)
    p.variance = torch.cat(variances)
    nll, rmse = common.score(p, test_y, std)

    dt = time.time() - t0
    print(
        common.format_completion_line(
            args.task, args.surface, args.dataset,
            train_x.size(0), test_x.size(0),
            "epochs", args.epochs, args.batch_size, completed_updates, args.seed,
        ),
        flush=True,
    )
    print(
        common.format_metric_line(
            args.task, args.surface, args.dataset,
            train_x.size(0), test_x.size(0),
            "epochs", args.epochs, args.batch_size, completed_updates,
            args.seed, nll, rmse, dt,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
