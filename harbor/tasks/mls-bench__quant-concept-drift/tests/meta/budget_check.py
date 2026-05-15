"""Parameter budget check for quant-concept-drift (standalone).

Run by tools.py before training: python /workspace/_task/budget_check.py
Instantiates each baseline's CustomModel, counts torch.nn parameters, and
asserts the agent's model doesn't exceed 1.05x the largest neural baseline.
Non-neural baselines (e.g., LightGBM) are excluded from the parameter budget.
"""
import importlib.util
import json
import os
import sys
import tempfile

import torch

TASK_DIR = os.environ.get("MLSBENCH_TASK_DIR", "/workspace/_task")
WORKSPACE_PKG = os.environ.get("MLSBENCH_PKG_DIR", "/workspace/qlib")
WORKSPACE_FILE = os.path.join(WORKSPACE_PKG, "custom_model.py")


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


def count_torch_params(obj):
    """Count total torch.nn parameters across all nn.Module attributes of obj."""
    total = 0
    seen = set()
    for attr_name in dir(obj):
        if attr_name.startswith("__"):
            continue
        try:
            attr = getattr(obj, attr_name)
        except Exception:
            continue
        if isinstance(attr, torch.nn.Module) and id(attr) not in seen:
            seen.add(id(attr))
            total += sum(p.numel() for p in attr.parameters())
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
        mod = load_module(tmp_path, f"_check_{bl_name}")
        model = mod.CustomModel()
        params = count_torch_params(model)
        baseline_params[bl_name] = params
        print(f"  baseline {bl_name}: {params} params")
    except Exception as e:
        print(f"  baseline {bl_name}: ERROR ({e})")
    finally:
        os.unlink(tmp_path)

neural_baselines = {k: v for k, v in baseline_params.items() if v > 0}

if not neural_baselines:
    print("WARNING: no neural baselines found, skipping budget check")
    sys.exit(0)

max_baseline = max(neural_baselines.values())
max_name = max(neural_baselines, key=neural_baselines.get)
budget = int(max_baseline * 1.05)

# -- Count params for agent's version --
agent_mod = load_module(WORKSPACE_FILE, "_agent_model")
agent_model = agent_mod.CustomModel()
agent_params = count_torch_params(agent_model)
print(f"\n  agent model: {agent_params} params")
print(f"  budget: {budget} (1.05 x {max_name}={max_baseline})")

if agent_params == 0:
    print("\nPASSED (non-neural approach)")
    sys.exit(0)

if agent_params > budget:
    print(f"\nFAILED: {agent_params} > {budget}", file=sys.stderr)
    sys.exit(1)

print("\nPASSED")
