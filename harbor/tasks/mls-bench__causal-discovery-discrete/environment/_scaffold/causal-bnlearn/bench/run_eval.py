"""Evaluation harness for the causal-discovery-discrete task."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from custom_algorithm import run_causal_discovery
from data_gen import load_and_sample
from metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CPDAG recovery on bnlearn discrete data."
    )
    parser.add_argument(
        "--network", required=True,
        choices=[
            "cancer", "earthquake", "survey", "asia", "sachs",
            "child", "insurance", "water", "mildew", "alarm",
            "barley", "hailfinder", "hepar2", "win95pts",
        ],
        help="Name of the bnlearn network.",
    )
    parser.add_argument(
        "--n_samples", type=int, required=True,
        help="Number of observations to sample.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    X, true_dag, node_names = load_and_sample(
        network_name=args.network,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    print(
        f"Dataset: {args.network} | "
        f"nodes={len(node_names)} | samples={X.shape[0]}",
        flush=True,
    )

    est_graph = run_causal_discovery(X)
    m = compute_metrics(est_graph, true_dag)

    print(
        f"CAUSAL_METRICS "
        f"shd={m['shd']} "
        f"adj_precision={m['adj_precision']:.4f} "
        f"adj_recall={m['adj_recall']:.4f} "
        f"arrow_precision={m['arrow_precision']:.4f} "
        f"arrow_recall={m['arrow_recall']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
