"""Evaluation harness for the causal-observational-nonlinear task.

FIXED driver (do not edit). It loads a PRE-GENERATED observational matrix X
(written into bench/_inputs/ by the task scaffold), calls the agent-editable
``run_causal_discovery(X)``, serializes the returned estimated adjacency matrix
B (n x n float64, convention B[i, j] != 0 means j -> i), and prints a single
base64 line:

    CAUSAL_PRED <args...> n=<n> adj=<base64 of the n x n float64 matrix>

The nonlinear ANM data-generating process (which also produces the true
adjacency matrix) and the metrics live in a host-only module the agent's process
cannot import. The host-side parser regenerates the true B, reconstructs the
estimated B from the payload, and scores SHD + F1 + precision + recall. This
driver never imports data_gen / metrics and never holds the true adjacency.
"""
import argparse
import base64
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from custom_algorithm import run_causal_discovery


def _inputs_dir():
    d = os.environ.get("CAUSAL_INPUTS_DIR")
    if d:
        return d
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_inputs")


def _input_key(args):
    return (
        f"{args.graph_type}_n{args.n_nodes}_s{args.n_samples}"
        f"_{args.noise_type}_{args.fn_type}_p{args.er_prob}_m{args.sf_m}_seed{args.seed}"
    )


def _load_input(args):
    path = os.path.join(_inputs_dir(), f"{_input_key(args)}.npy.b64")
    with open(path, "r") as f:
        raw = base64.b64decode(f.read())
    return np.frombuffer(raw, dtype=np.float64).reshape(args.n_samples, args.n_nodes).copy()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a causal discovery algorithm on synthetic nonlinear ANM data."
    )
    parser.add_argument(
        "--graph_type", required=True, choices=["er", "sf"],
        help="DAG topology: 'er' (Erdos-Renyi) or 'sf' (Scale-Free / BA)",
    )
    parser.add_argument("--n_nodes",   type=int,   required=True, help="Number of variables")
    parser.add_argument("--n_samples", type=int,   required=True, help="Number of observations")
    parser.add_argument(
        "--noise_type", required=True, choices=["exp", "laplace", "uniform", "gaussian"],
        help="Exogenous noise distribution",
    )
    parser.add_argument(
        "--fn_type", default="mixed", choices=["gp", "mlp", "poly", "sigmoid", "mixed"],
        help="Nonlinear function type (default: mixed)",
    )
    parser.add_argument("--er_prob", type=float, default=0.3,
                        help="Edge probability for ER graphs (default: 0.3)")
    parser.add_argument("--sf_m",    type=int,   default=2,
                        help="Edges per new node for BA/SF graphs (default: 2)")
    parser.add_argument("--seed",    type=int,   default=42, help="Random seed")
    args = parser.parse_args()

    X = _load_input(args)

    B_est = run_causal_discovery(X)
    B_est = np.asarray(B_est, dtype=np.float64)
    if B_est.shape != (args.n_nodes, args.n_nodes):
        raise ValueError(
            f"run_causal_discovery returned shape {B_est.shape}, expected "
            f"{(args.n_nodes, args.n_nodes)}."
        )
    payload = base64.b64encode(
        np.ascontiguousarray(B_est, dtype=np.float64).tobytes()
    ).decode("ascii")

    print(
        f"CAUSAL_PRED "
        f"graph_type={args.graph_type} "
        f"n_nodes={args.n_nodes} "
        f"n_samples={args.n_samples} "
        f"noise_type={args.noise_type} "
        f"fn_type={args.fn_type} "
        f"er_prob={args.er_prob} "
        f"sf_m={args.sf_m} "
        f"seed={args.seed} "
        f"n={args.n_nodes} "
        f"adj={payload}",
        flush=True,
    )


if __name__ == "__main__":
    main()
