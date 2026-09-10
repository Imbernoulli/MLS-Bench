"""Parameter budget check for graph-link-prediction (standalone).

Run by tools.py before training: python /workspace/_task/budget_check.py
Imports each baseline, instantiates models, counts params, and
asserts the agent's model doesn't exceed 1.05x the largest baseline.

Note on variable node counts: Cora ~2708, CiteSeer ~3327, ogbl-collab
~235868 nodes. The required baselines (gcn_dot, vgae, seal) do not use
per-node embedding tables, so their parameter counts are independent of
the number of nodes. If a baseline does use an embedding table sized to
node count (e.g. node2vec), its param count will scale with ENV via the
embedding's hard-coded max_num_nodes. We count params with only the
input feature dimension varying per env.
"""
import importlib.util
import json
import os
import sys
import tempfile

import torch

TASK_DIR = "/workspace/_task"
WORKSPACE_FILE = "/workspace/pytorch-geometric-lp/custom_linkpred.py"

# Ensure the package root is on sys.path
sys.path.insert(0, "/workspace")

# -- Dataset-specific dimensions (no data loading needed) --
# (num_node_features,) -- ogbl-collab has 128-dim embeddings as features
DATASET_DIMS = {
    "Cora": 1433,
    "CiteSeer": 3703,
    "ogbl-collab": 128,
}

# Training hyperparameters (must match the FIXED defaults of custom_template.py)
HIDDEN_CHANNELS = 256
NUM_LAYERS = 2
DROPOUT = 0.0

env_label = os.environ.get("ENV", "Cora")
in_channels = DATASET_DIMS.get(env_label, 1433)


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


def count_params(module_path):
    """Import module, instantiate LinkPredictor, return total param count."""
    mod = load_module(module_path, f"_check_{id(module_path)}")
    model = mod.LinkPredictor(
        in_channels=in_channels,
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    return sum(p.numel() for p in model.parameters())


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
        params = count_params(tmp_path)
        baseline_params[bl_name] = params
        print(f"  baseline {bl_name}: {params} params")
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

# -- Count params for agent's version --
agent_params = count_params(WORKSPACE_FILE)
print(f"\n  agent model: {agent_params} params")
print(f"  budget: {budget} (1.05 x {max_name}={max_baseline})")
print(f"  env={env_label}, in_channels={in_channels}")

if agent_params > budget:
    print(f"\nFAILED: {agent_params} > {budget}", file=sys.stderr)
    sys.exit(1)

print("\nPASSED")
