"""Parameter budget check for ml-anomaly-detection (standalone).

Run by tools.py before training: python /workspace/_task/budget_check.py

Fits each baseline's CustomAnomalyDetector on a small dummy dataset, then
counts trainable parameters in any torch.nn.Module attached to the detector.
Asserts the agent's version doesn't exceed 1.05x the largest baseline.

Baselines (IForest, LOF, OCSVM, ECOD, COPOD) have 0 torch params, so the
effective budget is the 10,000-param floor — enough for small learned
components but prevents importing large deep models (DIF, DeepSVDD, etc.).
"""
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

_PKG_DIR = os.environ.get("MLSBENCH_PKG_DIR")
TASK_DIR = os.environ.get("MLSBENCH_TASK_DIR", "/workspace/_task")
if _PKG_DIR:
    WORKSPACE_FILE = os.path.join(_PKG_DIR, "custom_anomaly.py")
    sys.path.insert(0, os.path.dirname(_PKG_DIR))
else:
    WORKSPACE_FILE = "/workspace/scikit-learn/custom_anomaly.py"
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


def count_torch_params(detector):
    """Count trainable torch parameters in any nn.Module attached to the detector."""
    total = 0
    try:
        import torch.nn as nn
    except ImportError:
        return 0  # no torch in env → no torch params possible

    visited = set()
    queue = [detector]
    while queue:
        obj = queue.pop()
        obj_id = id(obj)
        if obj_id in visited:
            continue
        visited.add(obj_id)

        if isinstance(obj, nn.Module):
            total += sum(p.numel() for p in obj.parameters())
            continue  # nn.Module.parameters() handles recursion

        # Recurse into object attributes
        for attr_name in dir(obj):
            if attr_name.startswith("__"):
                continue
            try:
                attr = getattr(obj, attr_name)
            except Exception:
                continue
            if isinstance(attr, nn.Module):
                total += sum(p.numel() for p in attr.parameters())
            elif hasattr(attr, "__dict__") and not callable(attr):
                if id(attr) not in visited:
                    queue.append(attr)

    return total


def count_params_for_module(module_path):
    """Load module, instantiate detector, fit on dummy data, count params."""
    mod = load_module(module_path, f"_check_{id(module_path)}")

    # Find CustomAnomalyDetector class in the module
    detector_cls = getattr(mod, "CustomAnomalyDetector", None)
    if detector_cls is None:
        print(f"  WARNING: No CustomAnomalyDetector found in {module_path}")
        return 0

    detector = detector_cls()

    # Fit on small dummy data (21 features like cardio dataset)
    rng = np.random.RandomState(42)
    X_dummy = rng.randn(100, 21)
    try:
        detector.fit(X_dummy)
    except Exception as e:
        print(f"  WARNING: fit() failed ({e}), counting pre-fit params only")

    return count_torch_params(detector)


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
# Allow up to 1.05x the max baseline, with a floor of 10000 params
budget = max(int(max_baseline * 1.05), 10000)

# -- Count params for agent's version --
agent_params = count_params_for_module(WORKSPACE_FILE)
print(f"\n  agent detector: {agent_params} torch params")
print(f"  budget: {budget} (1.05 x {max_name}={max_baseline}, floor=10000)")

if agent_params > budget:
    print(f"\nFAILED: {agent_params} > {budget}", file=sys.stderr)
    sys.exit(1)

print("\nPASSED")
