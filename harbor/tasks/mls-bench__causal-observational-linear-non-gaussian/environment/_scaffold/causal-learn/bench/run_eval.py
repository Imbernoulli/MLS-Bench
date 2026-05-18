"""Evaluation harness for the causal-observational-linear-non-gaussian task."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_gen import simulate_lingam
from metrics import compute_metrics
from custom_algorithm import run_causal_discovery


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a causal discovery algorithm on synthetic LiNGAM data."
    )
    parser.add_argument(
        "--graph_type", required=True, choices=["er", "sf"],
        help="DAG topology: 'er' (Erdos-Renyi) or 'sf' (Scale-Free / BA)",
    )
    parser.add_argument("--n_nodes",   type=int,   required=True, help="Number of variables")
    parser.add_argument("--n_samples", type=int,   required=True, help="Number of observations")
    parser.add_argument(
        "--noise_type", required=True, choices=["exp", "laplace", "uniform"],
        help="Exogenous noise distribution",
    )
    parser.add_argument("--er_prob", type=float, default=0.5,
                        help="Edge probability for ER graphs (default: 0.5)")
    parser.add_argument("--sf_m",    type=int,   default=2,
                        help="Edges per new node for BA/SF graphs (default: 2)")
    parser.add_argument("--seed",    type=int,   default=42, help="Random seed")
    args = parser.parse_args()

    X, B_true = simulate_lingam(
        n_nodes=args.n_nodes,
        n_samples=args.n_samples,
        graph_type=args.graph_type,
        noise_type=args.noise_type,
        seed=args.seed,
        er_prob=args.er_prob,
        sf_m=args.sf_m,
    )

    B_est = run_causal_discovery(X)
    m = compute_metrics(B_est, B_true)

    print(
        f"CAUSAL_METRICS "
        f"shd={m['shd']} "
        f"f1={m['f1']:.4f} "
        f"precision={m['precision']:.4f} "
        f"recall={m['recall']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
