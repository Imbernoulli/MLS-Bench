"""Parameter budget check for pde-design-solver (standalone).

Run by tools.py before training: python /workspace/_task/budget_check.py

Imports each baseline's modified Custom.py, instantiates Model with the
per-method args (after applying CONFIG_OVERRIDES from the file), counts
params, and asserts the agent's model doesn't exceed 1.05x the LARGEST
of the paper-faithful baseline settings — Transolver at n_hidden=256
(matching vendor/external_packages/Neural-Solver-Library/scripts/DesignBench/car/Transolver.sh).
"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

TASK_DIR = Path(os.environ.get("MLS_TASK_DIR", "/workspace/_task"))
PACKAGE_ROOT = Path(
    os.environ.get("MLS_PACKAGE_ROOT", "/workspace/Neural-Solver-Library")
)
WORKSPACE_FILE = Path(
    os.environ.get(
        "MLS_WORKSPACE_FILE",
        str(PACKAGE_ROOT / "models" / "Custom.py"),
    )
)

# Ensure the package root is on sys.path so that layer imports work.
sys.path.insert(0, str(PACKAGE_ROOT))

# Per-dataset args. n_hidden / slice_num here are the SHELL-SCRIPT defaults
# (matching scripts/car.sh, scripts/airfrans.sh, scripts/aircraft.sh).
# Each baseline can override them via CONFIG_OVERRIDES inside Custom.py.
DATASET_ARGS = {
    "Car": dict(
        n_hidden=128,
        n_heads=8,
        n_layers=8,
        mlp_ratio=2,
        slice_num=32,
        unified_pos=0,
        ref=8,
        space_dim=3,
        fun_dim=7,
        out_dim=4,
        geotype="unstructured",
        act="gelu",
        dropout=0.0,
        time_input=False,
        shapelist=None,
    ),
    "AirfRANS": dict(
        n_hidden=128,
        n_heads=8,
        n_layers=8,
        mlp_ratio=2,
        slice_num=32,
        unified_pos=0,
        ref=8,
        space_dim=2,
        fun_dim=7,
        out_dim=4,
        geotype="unstructured",
        act="gelu",
        dropout=0.0,
        time_input=False,
        shapelist=None,
    ),
    "AirCraft": dict(
        n_hidden=128,
        n_heads=8,
        n_layers=8,
        mlp_ratio=2,
        slice_num=32,
        unified_pos=0,
        ref=8,
        space_dim=3,
        fun_dim=7,
        out_dim=6,
        geotype="unstructured",
        act="gelu",
        dropout=0.0,
        time_input=False,
        shapelist=None,
    ),
}

# Allowed CONFIG_OVERRIDES keys (must be a subset of args attributes).
ALLOWED_OVERRIDES = {"n_hidden", "slice_num"}

env_label = os.environ.get("ENV", "Car")
base_args = DATASET_ARGS.get(env_label, DATASET_ARGS["Car"])


class MockArgs:
    """Lightweight args object that mimics argparse.Namespace."""

    def __init__(self, d):
        self.__dict__.update(d)


def load_module(path, name=None):
    path = str(path)
    name = name or f"_mod_{hash(path)}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def apply_ops(lines, ops, filename):
    result = list(lines)
    sorted_ops = sorted(
        [o for o in ops if o.get("file") == filename],
        key=lambda o: -o.get("start_line", o.get("after_line", 0)),
    )
    for op in sorted_ops:
        if op["op"] == "replace":
            s, e = op["start_line"] - 1, op["end_line"]
            result[s:e] = op["content"].splitlines()
        elif op["op"] == "insert":
            after = op["after_line"]
            result[after:after] = op["content"].splitlines()
        elif op["op"] == "delete":
            s, e = op["start_line"] - 1, op["end_line"]
            del result[s:e]
    return result


def count_params(module_path, args_dict):
    """Import module, instantiate Model, return total param count and overrides used."""
    mod = load_module(module_path, f"_check_{id(module_path)}")
    overrides = getattr(mod, "CONFIG_OVERRIDES", {}) or {}
    # Filter to allowed keys and merge with base args
    effective = dict(args_dict)
    used = {}
    for k, v in overrides.items():
        if k in ALLOWED_OVERRIDES:
            effective[k] = v
            used[k] = v
    model = mod.Model(MockArgs(effective))
    return sum(p.numel() for p in model.parameters()), used


mid_edit = load_module(TASK_DIR / "edits" / "mid_edit.py", "_mid_edit")
config = json.loads((TASK_DIR / "config.json").read_text())
editable_file = None
for f in config.get("files", []):
    if f.get("edit"):
        editable_file = f["filename"]
        break

template_content = None
for op in mid_edit.OPS:
    if op.get("op") == "create" and op.get("file") == editable_file:
        template_content = op["content"]
        break

assert template_content, f"No template found for {editable_file}"
template_lines = template_content.splitlines()

baseline_params = {}
for bl_name, bl_cfg in config.get("baselines", {}).items():
    edit_path = TASK_DIR / bl_cfg["edit_ops"]
    if not edit_path.exists():
        continue
    bl_mod = load_module(edit_path, f"_bl_{bl_name}")
    ops = getattr(bl_mod, "OPS", [])
    modified_lines = apply_ops(template_lines, ops, editable_file)
    modified_code = "\n".join(modified_lines)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(modified_code)
        tmp_path = f.name
    try:
        params, used = count_params(tmp_path, base_args)
        baseline_params[bl_name] = params
        print(f"  baseline {bl_name}: {params} params (overrides={used})")
    except Exception as e:
        print(f"  baseline {bl_name}: ERROR ({e})")
    finally:
        os.unlink(tmp_path)

if not baseline_params:
    print("WARNING: no baselines could be evaluated, skipping budget check")
    sys.exit(0)

max_baseline = max(baseline_params.values())
max_name = max(baseline_params, key=baseline_params.get)
budget = int(max_baseline * 1.05)

# Agent model: also honor its CONFIG_OVERRIDES.
agent_params, agent_used = count_params(WORKSPACE_FILE, base_args)
print(f"\n  agent model: {agent_params} params (overrides={agent_used})")
print(
    f"  budget: {budget} (1.05 x {max_name}={max_baseline}, "
    f"max-of-paper-settings)"
)
print(
    f"  env={env_label}, base n_hidden={base_args['n_hidden']}, "
    f"slice_num={base_args['slice_num']}, geotype={base_args['geotype']}"
)

if agent_params > budget:
    print(f"\nFAILED: {agent_params} > {budget}", file=sys.stderr)
    sys.exit(1)

print("\nPASSED")
