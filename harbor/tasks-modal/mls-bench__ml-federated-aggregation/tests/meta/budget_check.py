"""Parameter budget check for ml-federated-aggregation (standalone).

Run by tools.py before training: python /workspace/_task/budget_check.py
Instantiates each baseline's ServerAggregator, counts any torch parameters
it creates (server-side models, momentum buffers, etc.), and asserts the
agent's version doesn't exceed 1.05x the largest baseline.
"""
import importlib.util
import json
import os
import sys
import tempfile
from argparse import Namespace

import torch
import torch.nn as nn

TASK_DIR = "/workspace/_task"
WORKSPACE_FILE = "/workspace/flower/custom_fl_aggregation.py"

sys.path.insert(0, "/workspace")


def load_module(path, name=None):
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


def make_dummy_model():
    """Simple 2-layer net matching the FL simulation models."""
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
    return model


def make_dummy_args():
    return Namespace(
        num_clients=10, clients_per_round=5, num_rounds=10,
        local_epochs=1, lr=0.01, seed=42,
    )


def count_aggregator_params(module_path):
    """Import module, instantiate ServerAggregator, count extra params."""
    mod = load_module(module_path, f"_check_{id(module_path)}")
    dummy_model = make_dummy_model()
    dummy_args = make_dummy_args()
    agg = mod.ServerAggregator(dummy_model, dummy_args)

    total = 0
    for attr_name in dir(agg):
        if attr_name.startswith("_"):
            continue
        attr = getattr(agg, attr_name, None)
        if isinstance(attr, nn.Module):
            total += sum(p.numel() for p in attr.parameters())
        elif isinstance(attr, nn.Parameter):
            total += attr.numel()
        elif isinstance(attr, torch.Tensor) and attr.requires_grad:
            total += attr.numel()
    return total


# -- Get template content --
mid_edit = load_module(os.path.join(TASK_DIR, "edits", "mid_edit.py"), "_mid_edit")
config = json.loads(open(os.path.join(TASK_DIR, "config.json")).read())
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

# -- Count params for each baseline --
baseline_params = {}
for bl_name, bl_cfg in config.get("baselines", {}).items():
    edit_path = os.path.join(TASK_DIR, bl_cfg["edit_ops"])
    if not os.path.exists(edit_path):
        continue
    bl_mod = load_module(edit_path, f"_bl_{bl_name}")
    ops = getattr(bl_mod, "OPS", [])
    modified_lines = apply_ops(template_lines, ops, editable_file)
    modified_code = "\n".join(modified_lines)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(modified_code)
        tmp_path = f.name
    try:
        params = count_aggregator_params(tmp_path)
        baseline_params[bl_name] = params
        print(f"  baseline {bl_name}: {params} extra params")
    except Exception as e:
        print(f"  baseline {bl_name}: ERROR ({e})")
    finally:
        os.unlink(tmp_path)

if not baseline_params:
    print("WARNING: no baselines could be evaluated, skipping budget check")
    sys.exit(0)

max_baseline = max(baseline_params.values())
max_name = max(baseline_params, key=baseline_params.get)
# Allow up to 1.05x the max baseline, with a floor of 10000 params
# (baselines may have 0 extra params, but agents should still be allowed
# a small server-side model for techniques like server momentum)
budget = max(int(max_baseline * 1.05), 10000)

# -- Count params for agent's version --
agent_params = count_aggregator_params(WORKSPACE_FILE)
print(f"\n  agent aggregator: {agent_params} extra params")
print(f"  budget: {budget} (1.05 x {max_name}={max_baseline}, floor=10000)")

if agent_params > budget:
    print(f"\nFAILED: {agent_params} > {budget}", file=sys.stderr)
    sys.exit(1)

print("\nPASSED")
