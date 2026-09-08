"""Parameter budget check for ml-continual-regularization (standalone).

Run by tools.py before training: python /workspace/_task/budget_check.py
Runs each baseline's estimate_importance() on a dummy model to detect
if the regularization strategy attaches extra trainable modules to the model.
Asserts the agent's version doesn't exceed 1.05x the largest baseline.
"""
import importlib.util
import json
import os
import sys
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

TASK_DIR = "/workspace/_task"
WORKSPACE_FILE = "/workspace/continual-learning/custom_regularization.py"

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


class DummyModel(nn.Module):
    """Simple model mimicking the continual learning classifier."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def param_list(self):
        return self.named_parameters()

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.view(x.size(0), -1))))


def count_extra_params(module_path):
    """Import module, run estimate_importance, count any extra params added."""
    mod = load_module(module_path, f"_check_{id(module_path)}")
    model = DummyModel()
    device = torch.device("cpu")

    # Count baseline model params
    base_params = sum(p.numel() for p in model.parameters())

    # Create dummy dataset and prev_params
    dummy_data = TensorDataset(torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,)))
    prev_params = {n: p.clone().detach() for n, p in model.named_parameters()}

    # Run estimate_importance — this may attach extra modules/params to model
    try:
        mod.estimate_importance(model, dummy_data, prev_params, device)
    except Exception:
        pass  # OK if it fails on dummy data — we just want to check what it attaches

    # Count all params now (model + any attached modules)
    total = sum(p.numel() for p in model.parameters())
    # Also count any nn.Module attributes attached to the model
    for attr_name in dir(model):
        if attr_name.startswith("_custom"):
            attr = getattr(model, attr_name, None)
            if isinstance(attr, nn.Module):
                total += sum(p.numel() for p in attr.parameters())

    return total - base_params


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
        params = count_extra_params(tmp_path)
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
budget = max(int(max_baseline * 1.05), 10000)

# -- Count params for agent's version --
agent_params = count_extra_params(WORKSPACE_FILE)
print(f"\n  agent regularizer: {agent_params} extra params")
print(f"  budget: {budget} (1.05 x {max_name}={max_baseline}, floor=10000)")

if agent_params > budget:
    print(f"\nFAILED: {agent_params} > {budget}", file=sys.stderr)
    sys.exit(1)

print("\nPASSED")
