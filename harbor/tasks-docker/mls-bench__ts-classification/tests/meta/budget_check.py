"""Parameter budget check for Time-Series-Library tasks.

Run by tools.py before training: python /workspace/_task/budget_check.py
Counts parameters for the agent's Custom model and for the configured
read-only baseline models under the same active ENV setting.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import torch

TASK_DIR = Path(os.environ.get("MLSBENCH_TASK_DIR") or "/workspace/_task")
PKG_DIR = Path(os.environ.get("MLSBENCH_PKG_DIR") or "/workspace/Time-Series-Library")
WORKSPACE_FILE = PKG_DIR / "models" / "Custom.py"

sys.path.insert(0, str(PKG_DIR))

M4_HORIZONS = {"Yearly": 6, "Quarterly": 8, "Monthly": 18, "Weekly": 13, "Daily": 14, "Hourly": 48}
M4_FREQUENCIES = {"Yearly": 1, "Quarterly": 4, "Monthly": 12, "Weekly": 1, "Daily": 1, "Hourly": 24}

DEFAULTS = {
    "task_name": "long_term_forecast",
    "is_training": 1,
    "model_id": "test",
    "model": "Custom",
    "data": "ETTh1",
    "root_path": "./data/ETT/",
    "data_path": "ETTh1.csv",
    "features": "M",
    "target": "OT",
    "freq": "h",
    "checkpoints": "./checkpoints/",
    "seq_len": 96,
    "label_len": 48,
    "pred_len": 96,
    "seasonal_patterns": "Monthly",
    "inverse": False,
    "mask_rate": 0.25,
    "anomaly_ratio": 0.25,
    "expand": 2,
    "d_conv": 4,
    "top_k": 5,
    "num_kernels": 6,
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 7,
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 2048,
    "moving_avg": 25,
    "factor": 1,
    "distil": True,
    "dropout": 0.1,
    "embed": "timeF",
    "activation": "gelu",
    "channel_independence": 1,
    "decomp_method": "moving_avg",
    "use_norm": 1,
    "down_sampling_layers": 0,
    "down_sampling_window": 1,
    "down_sampling_method": None,
    "seg_len": 96,
    "num_workers": 0,
    "itr": 1,
    "train_epochs": 10,
    "batch_size": 32,
    "patience": 3,
    "learning_rate": 0.0001,
    "des": "budget_check",
    "loss": "MSE",
    "lradj": "type1",
    "use_amp": False,
    "use_gpu": False,
    "gpu": 0,
    "gpu_type": "cuda",
    "use_multi_gpu": False,
    "devices": "0",
    "p_hidden_dims": [128, 128],
    "p_hidden_layers": 2,
    "use_dtw": False,
    "augmentation_ratio": 0,
    "seed": 42,
    "patch_len": 16,
    "node_dim": 10,
    "gcn_depth": 2,
    "gcn_dropout": 0.3,
    "propalpha": 0.3,
    "conv_channel": 32,
    "skip_channel": 32,
    "individual": False,
    "alpha": 0.1,
    "top_p": 0.5,
    "pos": 1,
    "num_class": 2,
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expand_script_argv(script_path: Path) -> list[str]:
    shell_cmd = f'python() {{ printf "%s\\n" "$@"; }}; source {shlex.quote(str(script_path))}'
    output = subprocess.check_output(["bash", "-lc", shell_cmd], text=True, env=os.environ.copy())
    argv = [line.strip() for line in output.splitlines() if line.strip()]
    if argv[:2] == ["-u", "run.py"]:
        return argv[2:]
    if argv[:1] == ["run.py"]:
        return argv[1:]
    return argv


def coerce_value(raw: str):
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def parse_run_args(argv: list[str]) -> Namespace:
    values = dict(DEFAULTS)
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token[2:].replace("-", "_")
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            values[key] = coerce_value(argv[i + 1])
            i += 2
        else:
            values[key] = True
            i += 1
    values["use_gpu"] = False
    values["use_multi_gpu"] = False
    values["device"] = torch.device("cpu")
    return Namespace(**values)


def prepare_runtime_args(args: Namespace) -> Namespace:
    data_root = os.environ.get("DATA_ROOT")
    if data_root and isinstance(getattr(args, "root_path", None), str):
        if args.root_path == "/data" or args.root_path.startswith("/data/"):
            args.root_path = args.root_path.replace("/data", data_root, 1)
    if args.task_name == "classification":
        from data_provider.data_factory import data_provider

        train_data, _ = data_provider(args, flag="TRAIN")
        test_data, _ = data_provider(args, flag="TEST")
        args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        args.pred_len = 0
        args.enc_in = train_data.feature_df.shape[1]
        args.num_class = len(train_data.class_names)
    elif args.task_name == "short_term_forecast" and args.data == "m4":
        args.pred_len = M4_HORIZONS[args.seasonal_patterns]
        args.seq_len = 2 * args.pred_len
        args.label_len = args.pred_len
        args.frequency_map = M4_FREQUENCIES[args.seasonal_patterns]
    return args


def count_params(module_path: Path, args: Namespace) -> int:
    module = load_module(module_path, f"_budget_{module_path.stem}_{abs(hash(str(module_path)))}")
    model = module.Model(args)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def active_test_cmd(config: dict) -> dict:
    env_label = os.environ.get("ENV", "")
    for entry in config.get("test_cmds", []):
        if entry.get("label") == env_label:
            return entry
    return config["test_cmds"][0]


config = json.loads((TASK_DIR / "config.json").read_text())
test_entry = active_test_cmd(config)
agent_args = prepare_runtime_args(parse_run_args(expand_script_argv(TASK_DIR / test_entry["cmd"])))

baseline_params = {}
forced_baseline_params = {}
for baseline_name, baseline_cfg in config.get("baselines", {}).items():
    script_path = TASK_DIR / baseline_cfg["cmd"]
    script_args = prepare_runtime_args(parse_run_args(expand_script_argv(script_path)))
    module_path = PKG_DIR / "models" / f"{script_args.model}.py"
    if not module_path.exists():
        print(f"  baseline {baseline_name}: missing module {module_path.name}")
        continue
    try:
        params = count_params(module_path, script_args)
    except Exception as exc:
        print(f"  baseline {baseline_name}: ERROR ({exc})")
        continue
    baseline_params[baseline_name] = params
    print(f"  baseline {baseline_name}: {params} params")

    # Same-spec (matched-capacity) count: rebuild this baseline at the agent's
    # FORCED config -- i.e. the read-only eval script's (d_model, d_ff,
    # e_layers), carried on agent_args -- so the budget compares agent and
    # baselines like-for-like. The per-dataset baseline configs are often
    # *smaller* than the size the agent is forced to use (e.g. TimeXer uses
    # d_model=128 on Weather), which is exactly what makes a plain "1.05 x
    # largest *native* baseline" budget collapse below any honest d_model=512
    # model and zero out every agent. TimesNet is skipped here on purpose: it is
    # conv/FFT-based and balloons to ~3e8 params at d_model=512 (TSLib itself
    # runs it at d_model<=64), so it is not a fair same-spec reference -- its
    # properly tuned size still counts via the native term above.
    if script_args.model != "TimesNet":
        try:
            forced = count_params(module_path, agent_args)
            forced_baseline_params[baseline_name] = forced
            print(f"  baseline {baseline_name} @ agent config: {forced} params")
        except Exception as exc:
            print(f"  baseline {baseline_name} @ agent config: ERROR ({exc})")

if not baseline_params:
    print("WARNING: no baselines could be evaluated, skipping budget check")
    sys.exit(0)

max_baseline_name = max(baseline_params, key=baseline_params.get)
max_baseline_params = baseline_params[max_baseline_name]

# Budget basis: 1.05 x the strongest baseline, compared at a *matched* capacity.
#
# The agent's eval scripts hardcode (d_model, d_ff, e_layers) and the agent
# cannot change them (the scripts are read-only). On several datasets the
# per-dataset-tuned baselines are configured *smaller* than that forced size,
# so a plain "1.05 x largest *native* baseline" budget can fall below any honest
# model built at the forced d_model -- making the budget unsatisfiable and
# collapsing every agent's score to 0 (one failed setting zeroes the gmean)
# through no fault of the agent.
#
# Fix: also consider each baseline rebuilt at the agent's forced config
# (forced_baseline_params, gathered in the loop above) and take the larger
# basis. This is a like-for-like, same-spec comparison -- the agent is held to
# 1.05x the strongest baseline *at the same capacity it is forced to use*. It
# only ever raises the budget (max with the native value), so no previously
# passing run can newly fail.
budget_basis_name = max_baseline_name
budget_basis_params = max_baseline_params
if forced_baseline_params:
    forced_max_name = max(forced_baseline_params, key=forced_baseline_params.get)
    forced_max_params = forced_baseline_params[forced_max_name]
    if forced_max_params > budget_basis_params:
        budget_basis_name = f"{forced_max_name}@agent_cfg"
        budget_basis_params = forced_max_params
budget = int(budget_basis_params * 1.05)

agent_params = count_params(WORKSPACE_FILE, agent_args)
print(f"\n  agent model: {agent_params} params")
print(f"  budget: {budget} (1.05 x {budget_basis_name}={budget_basis_params})")
print(f"  env={os.environ.get('ENV', '')}, task={agent_args.task_name}, model_id={agent_args.model_id}")

if agent_params > budget:
    print(f"\nFAILED: {agent_params} > {budget}", file=sys.stderr)
    sys.exit(1)

print("\nPASSED")
