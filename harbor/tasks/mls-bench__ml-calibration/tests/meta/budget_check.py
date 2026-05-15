"""Parameter budget check for ml-calibration (standalone).

Run by tools.py before training:
    python $MLSBENCH_TASK_DIR/budget_check.py

Fits each baseline's CalibrationMethod on a small dummy (probs, labels)
set and counts trainable parameters in any torch.nn.Module attached.
Asserts the agent's method doesn't exceed 1.05x the largest baseline,
with a floor of 10_000 params.

Baselines (Platt, temperature scaling, isotonic regression) have 0 torch
params, so the floor is the effective budget — enough for small learned
heads but not for deep calibration networks.
"""
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

TASK_DIR = os.environ.get("MLSBENCH_TASK_DIR", "/workspace/_task")
WORKSPACE_FILE = os.path.join(
    os.environ.get("MLSBENCH_PKG_DIR", "/workspace/scikit-learn"),
    "custom_calibration.py",
)


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
    try:
        import torch.nn as nn
    except ImportError:
        return 0
    total = 0
    visited = set()
    queue = [obj]
    while queue:
        cur = queue.pop()
        cid = id(cur)
        if cid in visited:
            continue
        visited.add(cid)
        if isinstance(cur, nn.Module):
            total += sum(p.numel() for p in cur.parameters())
            continue
        for attr_name in dir(cur):
            if attr_name.startswith("__"):
                continue
            try:
                attr = getattr(cur, attr_name)
            except Exception:
                continue
            if isinstance(attr, nn.Module):
                total += sum(p.numel() for p in attr.parameters())
            elif hasattr(attr, "__dict__") and not callable(attr):
                if id(attr) not in visited:
                    queue.append(attr)
    return total


def count_params_for_module(module_path):
    mod = load_module(module_path, f"_check_{id(module_path)}")
    cls = getattr(mod, "CalibrationMethod", None)
    if cls is None:
        print(f"  WARNING: No CalibrationMethod found in {module_path}")
        return 0
    method = cls()

    # Try both binary (1D probs) and multiclass (2D probs, 3 classes) fits.
    rng = np.random.RandomState(42)
    # Binary fit
    probs_bin = rng.uniform(0.1, 0.9, size=(200,))
    labels_bin = rng.randint(0, 2, size=200)
    try:
        method.fit(probs_bin, labels_bin)
    except Exception:
        # Fall back to multiclass dummy
        try:
            raw = rng.uniform(size=(200, 3))
            probs_mc = raw / raw.sum(axis=1, keepdims=True)
            labels_mc = rng.randint(0, 3, size=200)
            method.fit(probs_mc, labels_mc)
        except Exception as e:
            print(f"  WARNING: fit() failed on both binary+multiclass ({e})")
    return count_torch_params(method)


# -- Load template --
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

# -- Count params per baseline --
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
        params = count_params_for_module(tmp_path)
        baseline_params[bl_name] = params
        print(f"  baseline {bl_name}: {params} torch params")
    except Exception as e:
        print(f"  baseline {bl_name}: ERROR ({e})")
    finally:
        os.unlink(tmp_path)

if not baseline_params:
    print("WARNING: no baselines could be evaluated, skipping budget check")
    sys.exit(0)

max_baseline = max(baseline_params.values())
max_name = max(baseline_params, key=baseline_params.get)
budget = max(int(max_baseline * 1.05), 10_000)

agent_params = count_params_for_module(WORKSPACE_FILE)
print(f"\n  agent method: {agent_params} torch params")
print(f"  budget: {budget} (1.05 x {max_name}={max_baseline}, floor=10000)")

if agent_params > budget:
    print(f"\nFAILED: {agent_params} > {budget}", file=sys.stderr)
    sys.exit(1)

print("\nPASSED")
