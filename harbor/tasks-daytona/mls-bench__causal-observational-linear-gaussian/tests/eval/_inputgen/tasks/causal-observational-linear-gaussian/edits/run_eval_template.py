"""Evaluation harness for the causal-observational-linear-gaussian task.

FIXED driver (do not edit). It loads a PRE-GENERATED observational matrix X
(written into bench/_inputs/ by the task scaffold), calls the agent-editable
``run_causal_discovery(X)``, serializes the returned estimated CPDAG to its
causallearn endpoint matrix, and prints a single base64 line:

    CAUSAL_PRED <args...> n=<n> adj=<base64 of the n x n int endpoint matrix>

The data-generating process (which also produces the true DAG) and the metrics
live in a host-only module the agent's process cannot import. The host-side
parser regenerates the true DAG, reconstructs the estimated graph from the
endpoint matrix, and scores SHD + adjacency/arrow precision/recall. This driver
never imports data_gen / metrics and never holds the true DAG.
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
        f"_p{args.er_prob}_m{args.sf_m}_ns{args.noise_scale}_seed{args.seed}"
    )


def _load_input(args):
    path = os.path.join(_inputs_dir(), f"{_input_key(args)}.npy.b64")
    with open(path, "r") as f:
        raw = base64.b64decode(f.read())
    return np.frombuffer(raw, dtype=np.float64).reshape(args.n_samples, args.n_nodes).copy()


def _to_endpoint_matrix(est_graph, n_nodes):
    """Extract the causallearn endpoint matrix (n x n ints) from the output.

    Accepts a GeneralGraph/Graph, a wrapper with ``.G``, or ``dict['G']`` — the
    same shapes the metric accepts — and returns ``graph.graph``. Lossless: this
    is exactly how causallearn stores a graph internally.
    """
    g = est_graph
    if isinstance(g, dict) and "G" in g:
        g = g["G"]
    elif hasattr(g, "G") and not hasattr(g, "graph"):
        g = g.G
    mat = np.asarray(g.graph, dtype=np.int64)
    if mat.shape != (n_nodes, n_nodes):
        raise ValueError(
            f"Estimated graph matrix has shape {mat.shape}, expected "
            f"{(n_nodes, n_nodes)}."
        )
    return mat


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CPDAG recovery on synthetic linear Gaussian data."
    )
    parser.add_argument(
        "--graph_type", required=True, choices=["er", "sf"],
        help="DAG topology: 'er' (Erdos-Renyi) or 'sf' (Scale-Free / BA)",
    )
    parser.add_argument("--n_nodes", type=int, required=True, help="Number of variables")
    parser.add_argument("--n_samples", type=int, required=True, help="Number of observations")
    parser.add_argument(
        "--er_prob", type=float, default=0.5,
        help="Edge probability for ER graphs (default: 0.5)",
    )
    parser.add_argument(
        "--sf_m", type=int, default=2,
        help="Edges per new node for BA/SF graphs (default: 2)",
    )
    parser.add_argument(
        "--noise_scale", type=float, default=1.0,
        help="Noise std for linear Gaussian SEM (default: 1.0). Higher = harder.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    X = _load_input(args)

    est_graph = run_causal_discovery(X)
    mat = _to_endpoint_matrix(est_graph, args.n_nodes)
    payload = base64.b64encode(
        np.ascontiguousarray(mat, dtype=np.int64).tobytes()
    ).decode("ascii")

    print(
        f"CAUSAL_PRED "
        f"graph_type={args.graph_type} "
        f"n_nodes={args.n_nodes} "
        f"n_samples={args.n_samples} "
        f"er_prob={args.er_prob} "
        f"sf_m={args.sf_m} "
        f"noise_scale={args.noise_scale} "
        f"seed={args.seed} "
        f"n={args.n_nodes} "
        f"adj={payload}",
        flush=True,
    )


if __name__ == "__main__":
    main()
