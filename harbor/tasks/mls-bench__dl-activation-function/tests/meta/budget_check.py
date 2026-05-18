"""Parameter budget check for pytorch-vision tasks (standalone).

Run by tools.py before training: python /workspace/_task/budget_check.py
Imports each baseline's edit ops, applies them to the template, instantiates
the model, counts params, and asserts the agent's model doesn't exceed
1.05x the largest baseline.
"""
import importlib.util
import json
import os
import sys
import tempfile

import torch

TASK_DIR = "/workspace/_task"


def load_module(path, name=None):
    name = name or f"_mod_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def apply_ops(lines, ops, filename):
    result = list(lines)
    for op in sorted(
        [o for o in ops if o.get("file") == filename],
        key=lambda o: -o.get("start_line", 0),
    ):
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


def parse_env_label(label):
    """Parse ENV label like 'resnet56-cifar100' into (arch, num_classes)."""
    num_classes_map = {
        "cifar10": 10, "cifar100": 100, "fmnist": 10,
    }
    parts = label.rsplit("-", 1)
    if len(parts) != 2:
        return None, None
    arch, dataset = parts
    return arch, num_classes_map.get(dataset)


def count_params(module_path, arch, num_classes):
    """Import module, build model, return total param count."""
    mod = load_module(module_path, f"_chk_{abs(hash(module_path))}")
    model = mod.build_model(arch, num_classes)
    return sum(p.numel() for p in model.parameters())


# -- Determine arch from ENV --
env_label = os.environ.get("ENV", "")
arch, num_classes = parse_env_label(env_label)
if arch is None or num_classes is None:
    print(f"WARNING: cannot parse ENV={env_label!r}, skipping budget check")
    sys.exit(0)

# -- Load config --
config = json.loads(open(os.path.join(TASK_DIR, "config.json")).read())
editable_file = next(
    f["filename"] for f in config["files"] if f.get("edit")
)
workspace_file = os.path.join("/workspace", editable_file)

# -- Load template --
mid_edit = load_module(os.path.join(TASK_DIR, "edits", "mid_edit.py"), "_mid_edit")
template_content = next(
    op["content"] for op in mid_edit.OPS
    if op.get("op") == "create" and op.get("file") == editable_file
)
template_lines = template_content.splitlines()

# -- Count baseline params --
baseline_params = {}
for bl_name, bl_cfg in config.get("baselines", {}).items():
    edit_path = os.path.join(TASK_DIR, bl_cfg.get("edit_ops", ""))
    if not os.path.exists(edit_path):
        continue
    bl_mod = load_module(edit_path, f"_bl_{bl_name}")
    ops = getattr(bl_mod, "OPS", [])
    modified = apply_ops(template_lines, ops, editable_file)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("\n".join(modified))
        tmp = f.name
    try:
        p = count_params(tmp, arch, num_classes)
        baseline_params[bl_name] = p
        print(f"  baseline {bl_name}: {p:,} params")
    except Exception as e:
        print(f"  baseline {bl_name}: ERROR ({e})")
    finally:
        os.unlink(tmp)

if not baseline_params:
    print("WARNING: no baselines could be evaluated, skipping budget check")
    sys.exit(0)

max_bl = max(baseline_params, key=baseline_params.get)
budget = int(baseline_params[max_bl] * 1.05)

# -- Count agent params --
agent_params = count_params(workspace_file, arch, num_classes)
print(f"\n  agent model: {agent_params:,} params")
print(f"  budget: {budget:,} (1.05 x {max_bl}={baseline_params[max_bl]:,})")

if agent_params > budget:
    print(f"\nFAILED: {agent_params:,} > {budget:,}", file=sys.stderr)
    sys.exit(1)

print("\nPASSED")
