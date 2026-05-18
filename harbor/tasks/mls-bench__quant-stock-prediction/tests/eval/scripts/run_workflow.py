"""Run qlib workflow for stock prediction.

Loads workflow_config.yaml, calls task_train(), extracts metrics from the
recorder, and prints structured SIGNAL_METRIC / PORTFOLIO_METRIC lines
for the parser.

Supports CLI overrides for instruments and time segments to test
generalizability across different market universes and time periods.
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import yaml

import qlib
from qlib.config import REG_CN
from qlib.model.trainer import task_train


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruments", default=None,
                        help="Override instruments (e.g. csi100, csi300)")
    parser.add_argument("--fit-start", default=None,
                        help="Override fit_start_time for normalization")
    parser.add_argument("--fit-end", default=None,
                        help="Override fit_end_time for normalization")
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--val-start", default=None)
    parser.add_argument("--val-end", default=None)
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--experiment-name", default="stock_prediction")
    return parser.parse_args()


def apply_overrides(config, args):
    """Apply CLI overrides to the workflow config."""
    handler_kwargs = config["task"]["dataset"]["kwargs"]["handler"]["kwargs"]
    segments = config["task"]["dataset"]["kwargs"]["segments"]

    if args.instruments:
        handler_kwargs["instruments"] = args.instruments
    if args.fit_start:
        handler_kwargs["fit_start_time"] = args.fit_start
    if args.fit_end:
        handler_kwargs["fit_end_time"] = args.fit_end
    if args.train_start and args.train_end:
        segments["train"] = [args.train_start, args.train_end]
    if args.val_start and args.val_end:
        segments["valid"] = [args.val_start, args.val_end]
    if args.test_start and args.test_end:
        segments["test"] = [args.test_start, args.test_end]

    # Update PortAnaRecord backtest range to match test segment
    for record in config["task"].get("record", []):
        if record.get("class") == "PortAnaRecord":
            backtest_cfg = record["kwargs"]["config"]["backtest"]
            if args.test_start:
                backtest_cfg["start_time"] = args.test_start
            if args.test_end:
                backtest_cfg["end_time"] = args.test_end


def main():
    args = parse_args()

    seed = int(os.environ.get("SEED", 42))
    set_seed(seed)

    # Load workflow config from the workspace (cwd) where mid_edit places it
    config_path = "workflow_config.yaml"
    if not os.path.exists(config_path):
        # Fallback: look relative to the script (task dir)
        config_path = os.path.join(os.path.dirname(__file__), "..", "workflow_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    apply_overrides(config, args)

    # Apply sys.rel_path (same as qlib.cli.run.sys_config)
    for rel_path in config.get("sys", {}).get("rel_path", []):
        sys.path.insert(0, os.path.abspath(rel_path))

    # Initialize qlib
    qlib_init_cfg = config.get("qlib_init", {})
    provider_uri = os.path.expanduser(qlib_init_cfg.get("provider_uri", "~/.qlib/qlib_data/cn_data"))
    region_str = qlib_init_cfg.get("region", "cn")
    region = REG_CN if region_str == "cn" else REG_CN
    qlib.init(provider_uri=provider_uri, region=region, kernels=1)

    print(f"SEED={seed}")

    # Run the workflow
    task_config = config["task"]
    recorder = task_train(task_config, experiment_name=args.experiment_name)

    # Extract metrics from recorder
    metrics = recorder.list_metrics()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Signal metrics (from SigAnaRecord)
    signal_keys = {
        "IC": "IC",
        "ICIR": "ICIR",
        "Rank IC": "Rank_IC",
        "Rank ICIR": "Rank_ICIR",
    }
    for qlib_key, output_key in signal_keys.items():
        if qlib_key in metrics:
            val = metrics[qlib_key]
            print(f"SIGNAL_METRIC {output_key}={val:.6f}")

    # Portfolio metrics (from PortAnaRecord, prefixed with frequency)
    portfolio_keys = {
        "annualized_return": "annualized_return",
        "max_drawdown": "max_drawdown",
        "information_ratio": "information_ratio",
    }
    for freq_prefix in ("1day", "day"):
        found = False
        for qlib_suffix, output_key in portfolio_keys.items():
            full_key = f"{freq_prefix}.excess_return_with_cost.{qlib_suffix}"
            if full_key in metrics:
                found = True
                val = metrics[full_key]
                print(f"PORTFOLIO_METRIC {output_key}={val:.6f}")
        if found:
            break
    else:
        for key, val in sorted(metrics.items()):
            for qlib_suffix, output_key in portfolio_keys.items():
                if key.endswith(f"excess_return_with_cost.{qlib_suffix}"):
                    print(f"PORTFOLIO_METRIC {output_key}={val:.6f}")

    print("=" * 60)

    print("\nAll recorded metrics:")
    for key, val in sorted(metrics.items()):
        print(f"  {key} = {val}")


if __name__ == "__main__":
    main()
