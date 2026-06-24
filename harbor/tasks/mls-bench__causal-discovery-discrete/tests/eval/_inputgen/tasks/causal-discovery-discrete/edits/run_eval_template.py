"""Evaluation harness for the causal-discovery-discrete task (FIXED driver).

This program does NOT contain the data generator, the evaluation dataset
identity, the ground-truth graph, or the metric. It only:
  1. loads a PRE-GENERATED block of integer-encoded discrete observational data X
     (written into the workspace at setup time; no ground truth is present),
  2. calls the agent's ``run_causal_discovery(X)``,
  3. serializes the returned estimated graph to its integer endpoint adjacency
     matrix, and
  4. prints a single ``CAUSAL_PRED`` line (base64-encoded) for the host-side
     scorer to pick up.

Scoring (regenerating the ground truth and computing the structure-recovery
metrics) happens OFF this machine, in the host harness, against a held-out module
the agent's process cannot reach.
"""
import argparse
import base64
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from custom_algorithm import run_causal_discovery

from causallearn.graph.Graph import Graph


def _inputs_dir():
    d = os.environ.get("CAUSAL_INPUTS_DIR")
    if d:
        return d
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_causal_inputs")


def _load_input(label, seed):
    """Load the pre-generated integer-encoded samples X for (label, seed)."""
    path = os.path.join(_inputs_dir(), f"{label}_seed{seed}.npz.b64")
    with open(path, "r") as f:
        raw = base64.b64decode(f.read())
    d = np.load(io.BytesIO(raw))
    return d["X"]


def _normalize_graph_output(graph_output):
    """Normalize algorithm output to a causallearn Graph object.

    Accepts a Graph/GeneralGraph, a wrapper with ``.G``, or a dict with ``"G"``.
    """
    if isinstance(graph_output, dict) and "G" in graph_output:
        graph_output = graph_output["G"]
    elif hasattr(graph_output, "G"):
        graph_output = graph_output.G

    if not isinstance(graph_output, Graph):
        raise TypeError(
            "run_causal_discovery must return a causallearn Graph/GeneralGraph "
            f"(or wrapper with .G / dict['G']); got {type(graph_output)!r}"
        )
    return graph_output


def main():
    parser = argparse.ArgumentParser(
        description="Run a causal discovery algorithm on pre-generated discrete data."
    )
    parser.add_argument(
        "--label", default=os.environ.get("ENV", ""),
        help="Evaluation case label (also taken from the ENV variable).",
    )
    parser.add_argument("--seed", type=int,
                        default=int(os.environ.get("SEED", "42")), help="Random seed")
    args = parser.parse_args()

    label = args.label
    if not label:
        raise SystemExit("No evaluation label provided (set --label or ENV).")

    X = _load_input(label, args.seed)
    print(
        f"Dataset: {label} | nodes={X.shape[1]} | samples={X.shape[0]}",
        flush=True,
    )

    est_graph = run_causal_discovery(X)
    est_graph = _normalize_graph_output(est_graph)

    # The integer endpoint matrix fully encodes the estimated graph; serialize
    # it losslessly for the host scorer.
    adj = np.ascontiguousarray(np.asarray(est_graph.graph, dtype=np.int64))
    n_vars = int(adj.shape[0])
    payload = base64.b64encode(adj.tobytes()).decode("ascii")

    print(
        f"CAUSAL_PRED label={label} seed={args.seed} n={n_vars} adj={payload}",
        flush=True,
    )


if __name__ == "__main__":
    main()
