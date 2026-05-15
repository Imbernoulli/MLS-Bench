"""Evaluation harness for the causal-observational-linear-gaussian task."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from custom_algorithm import run_causal_discovery
from data_gen import simulate_linear_gaussian
from metrics import compute_metrics


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

    X, true_dag = simulate_linear_gaussian(
        n_nodes=args.n_nodes,
        n_samples=args.n_samples,
        graph_type=args.graph_type,
        seed=args.seed,
        er_prob=args.er_prob,
        sf_m=args.sf_m,
        noise_scale=args.noise_scale,
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
