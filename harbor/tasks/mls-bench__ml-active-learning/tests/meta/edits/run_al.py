"""Active learning evaluation runner for ml-active-learning task.

Runs a pool-based active learning loop with the specified query strategy
on an OpenML tabular dataset using an MLP model. Reports accuracy at each
round and computes final metrics: accuracy at fixed label budget and AUC
of the learning curve.

Usage:
    python run_al.py --did <openml_id> --alg custom --seed 42
"""

import argparse
import gc
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms

# Ensure badge package is importable
sys.path.insert(0, "/workspace/badge")

from dataset import get_handler
from query_strategies import (
    RandomSampling,
    LeastConfidence,
    EntropySampling,
    BadgeSampling,
    BaitSampling,
    BALDDropout,
)
from query_strategies.custom_sampling import CustomSampling


# ── MLP model (same as badge repo) ─────────────────────────────────────────
class mlpMod(nn.Module):
    def __init__(self, dim, embSize=128, nClasses=10, useNonLin=True):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.linear = nn.Linear(embSize, nClasses, bias=False)
        self.useNonLin = useNonLin

    def forward(self, x):
        x = x.view(-1, self.dim)
        if self.useNonLin:
            emb = F.relu(self.lm1(x))
        else:
            emb = self.lm1(x)
        out = self.linear(emb)
        return out, emb

    def get_embedding_dim(self):
        return self.embSize


def main():
    parser = argparse.ArgumentParser(description="Active Learning Evaluation")
    parser.add_argument("--did", type=int, required=True, help="OpenML dataset ID")
    parser.add_argument("--alg", type=str, default="custom", help="Algorithm name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--nStart", type=int, default=100, help="Initial labeled pool size")
    parser.add_argument("--nQuery", type=int, default=100, help="Batch query size per round")
    parser.add_argument("--nRounds", type=int, default=20, help="Number of AL rounds")
    parser.add_argument("--nEmb", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load OpenML dataset ─────────────────────────────────────────────
    # Resolve badge pkg dir: container /workspace/badge or $MLSBENCH_PKG_DIR
    # (local mode) or the cwd if the script is run from the badge root.
    data_dir = os.environ.get("BADGE_DATA_DIR")
    if data_dir:
        data_path = os.path.join(data_dir, f"data_{args.did}.pk")
    else:
        pkg_dir = os.environ.get("MLSBENCH_PKG_DIR", "/workspace/badge")
        data_path = os.path.join(pkg_dir, "oml", f"data_{args.did}.pk")
        if not os.path.exists(data_path):
            # Fallback: relative path (script often runs from the badge dir)
            data_path = os.path.join("oml", f"data_{args.did}.pk")
    data = pickle.load(open(data_path, "rb"))["data"]
    raw_X = np.asarray(data[0])
    # Encode string/categorical columns ordinally (e.g. splice dataset has amino-acid strings)
    if raw_X.dtype.kind in ('U', 'S', 'O'):
        from sklearn.preprocessing import OrdinalEncoder
        raw_X = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(raw_X)
    X = raw_X.astype(np.float32)
    y = np.asarray(data[1])

    # Handle NaN values
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    y = LabelEncoder().fit(y).transform(y)
    nClasses = int(max(y) + 1)
    nSamps, dim = X.shape

    # Train/test split
    testSplit = 0.1
    inds = np.random.permutation(nSamps)
    X = X[inds]
    y = y[inds]

    split = int((1.0 - testSplit) * nSamps)
    while True:
        inds = np.random.permutation(split)
        if len(inds) > 50000:
            inds = inds[:50000]
        X_tr = X[:split][inds]
        X_tr = torch.Tensor(X_tr)
        y_tr = y[:split][inds]
        Y_tr = torch.Tensor(y_tr).long()

        X_te = torch.Tensor(X[split:])
        Y_te = torch.Tensor(y[split:]).long()

        if len(np.unique(Y_tr.numpy())) == nClasses:
            break

    handler = get_handler("other")
    dim_shape = (dim,)

    train_args = {
        "transform": transforms.Compose([transforms.ToTensor()]),
        "n_epoch": 10,
        "loader_tr_args": {"batch_size": 128, "num_workers": 0},
        "loader_te_args": {"batch_size": 1000, "num_workers": 0},
        "optimizer_args": {"lr": args.lr, "momentum": 0},
        "transformTest": transforms.Compose([transforms.ToTensor()]),
        "lr": args.lr,
        "modelType": "mlp",
    }

    # BAIT regularization
    train_args["lamb"] = 1

    # ── Create network ──────────────────────────────────────────────────
    opts_nClasses = nClasses
    net = mlpMod(dim_shape, embSize=args.nEmb, nClasses=opts_nClasses)

    # ── Initialize labeled pool ─────────────────────────────────────────
    n_pool = len(Y_tr)
    idxs_lb = np.zeros(n_pool, dtype=bool)
    idxs_tmp = np.arange(n_pool)
    np.random.shuffle(idxs_tmp)
    idxs_lb[idxs_tmp[: args.nStart]] = True

    print(f"Dataset: OpenML-{args.did}", flush=True)
    print(f"Samples: {n_pool} train, {len(Y_te)} test, {nClasses} classes", flush=True)
    print(f"Initial labeled: {args.nStart}, Query size: {args.nQuery}, Rounds: {args.nRounds}", flush=True)
    print(f"Algorithm: {args.alg}", flush=True)

    # ── Select strategy ─────────────────────────────────────────────────
    X_tr_np = X_tr.numpy() if isinstance(X_tr, torch.Tensor) else X_tr
    if args.alg == "random":
        strategy = RandomSampling(X_tr_np, Y_tr, idxs_lb, net, handler, train_args)
    elif args.alg == "least_confidence":
        strategy = LeastConfidence(X_tr_np, Y_tr, idxs_lb, net, handler, train_args)
    elif args.alg == "entropy":
        strategy = EntropySampling(X_tr_np, Y_tr, idxs_lb, net, handler, train_args)
    elif args.alg == "badge":
        strategy = BadgeSampling(X_tr_np, Y_tr, idxs_lb, net, handler, train_args)
    elif args.alg == "bait":
        strategy = BaitSampling(X_tr_np, Y_tr, idxs_lb, net, handler, train_args)
    elif args.alg == "bald":
        strategy = BALDDropout(X_tr_np, Y_tr, idxs_lb, net, handler, train_args, n_drop=10)
    elif args.alg == "custom":
        strategy = CustomSampling(X_tr_np, Y_tr, idxs_lb, net, handler, train_args)
    else:
        raise ValueError(f"Unknown algorithm: {args.alg}")

    # ── Active learning loop ────────────────────────────────────────────
    X_te_np = X_te.numpy() if isinstance(X_te, torch.Tensor) else X_te
    accs = []

    # Round 0: train on initial labeled set
    strategy.train()
    P = strategy.predict(X_te_np, Y_te)
    acc0 = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    accs.append(acc0)
    n_labeled = int(sum(idxs_lb))
    print(f"TRAIN_METRICS round=0 n_labeled={n_labeled} accuracy={acc0:.6f}", flush=True)

    for rd in range(1, args.nRounds + 1):
        gc.collect()

        # Query
        q_idxs = strategy.query(args.nQuery)
        idxs_lb[q_idxs] = True

        # Update and retrain
        strategy.update(idxs_lb)
        strategy.train(verbose=False)

        # Evaluate
        P = strategy.predict(X_te_np, Y_te)
        acc = 1.0 * (Y_te == P).sum().item() / len(Y_te)
        accs.append(acc)
        n_labeled = int(sum(idxs_lb))
        print(f"TRAIN_METRICS round={rd} n_labeled={n_labeled} accuracy={acc:.6f}", flush=True)

        if sum(~strategy.idxs_lb) < args.nQuery:
            print("Unlabeled pool exhausted, stopping early.", flush=True)
            break

    # ── Compute final metrics ───────────────────────────────────────────
    final_accuracy = accs[-1]

    # AUC of learning curve (trapezoidal rule, normalized to [0,1] range)
    n_points = len(accs)
    if n_points > 1:
        auc = np.trapz(accs, dx=1.0 / (n_points - 1))
    else:
        auc = accs[0]

    print(f"TEST_METRICS accuracy={final_accuracy:.6f} auc={auc:.6f}", flush=True)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(
        os.path.join(args.output_dir, "results.npz"),
        accs=np.array(accs),
        final_accuracy=final_accuracy,
        auc=auc,
    )


if __name__ == "__main__":
    main()
