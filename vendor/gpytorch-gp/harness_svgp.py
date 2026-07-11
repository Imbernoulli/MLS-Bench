#!/usr/bin/env python3
"""Generic SVGP harness for the gp-* tasks that edit ONE surface of an otherwise
fixed Stochastic Variational GP regressor.

Reused by several tasks; each fixes a different `--surface` and points at a
different agent-editable solution file. The FIXED skeleton is:

    ApproximateGP( VariationalStrategy, <variational dist>, ConstantMean,
    ScaleKernel(RBFKernel(ard)) ), VariationalELBO, minibatch Adam loop for a fixed
    epoch budget, trained on a standardized fixed regression split, then scored on
    held-out test NLL (per point, original y scale, lower better) + RMSE.

The only editable axis is a literal learning-rate value. The inducing set (M=256
k-means), variational distribution, and batch size are fixed. The plan is parsed
without executing agent Python; invalid values terminate the command.

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


def _kmeans_inducing(train_x, m):
    from sklearn.cluster import KMeans

    m = min(m, train_x.size(0))
    x = train_x.detach().cpu().numpy()
    km = KMeans(n_clusters=m, n_init=3, random_state=42).fit(x)
    return torch.as_tensor(km.cluster_centers_, dtype=train_x.dtype, device=train_x.device)


def _default_variational(m):
    return gpytorch.variational.CholeskyVariationalDistribution(m)


def _default_batch(n):
    return 1024


class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, var_dist):
        vs = gpytorch.variational.VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=True
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
    ap.add_argument("--dataset", default="elevators")
    ap.add_argument("--solution", required=True)
    ap.add_argument(
        "--surface",
        required=True,
        choices=["svgp_lr"],
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--fixed-inducing", type=int, default=256,
                    help="M for the fixed k-means inducing set (variational/minibatch tasks)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    common.validate_run_contract(
        args.task, args.surface, args.dataset, "epochs", args.epochs, args.seed
    )
    if args.batch_size != 1024 or args.fixed_inducing != 256:
        raise ValueError(
            "the SVGP learning-rate protocol requires batch_size=1024 and "
            "fixed_inducing=256"
        )
    common.install_strict_numerical_warnings()
    common.set_seeds(args.seed)
    t0 = time.time()

    train_x, train_y, test_x, test_y, std, d = common.load_split(args.dataset)
    dev = common.device()

    config = common.load_surface_config(args.solution)
    common.validate_surface_config("svgp_lr", config)

    # The only editable axis is the learning rate. All structural choices stay fixed.
    ip = _kmeans_inducing(train_x, args.fixed_inducing)
    ip = ip.to(dev).float().contiguous()
    print(f"INDUCING M={ip.size(0)}", flush=True)

    vd = _default_variational(ip.size(0))
    print(f"VARIATIONAL {type(vd).__name__}", flush=True)

    bs = args.batch_size
    print(f"BATCH bs={bs}", flush=True)

    # --- learning rate ---
    lr = common.validate_learning_rate(config["learning_rate"])
    print(f"LR lr={lr}", flush=True)

    model = SVGPModel(ip, vd).to(dev)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dev)

    model.train(); likelihood.train()
    opt = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}],
        lr=lr,
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=bs, shuffle=True)
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

    class _P:
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
