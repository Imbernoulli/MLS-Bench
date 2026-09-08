"""Parameter budget check for meta-rl (standalone).

Run by tools.py before training: python /workspace/_task/budget_check.py
Imports each baseline encoder, counts params, and asserts the agent's
encoder doesn't exceed 1.1x the largest baseline.
"""
import importlib.util
import json
import os
import sys
import tempfile

import torch

TASK_DIR = os.environ.get("MLSBENCH_TASK_DIR", "/workspace/_task")
WORKSPACE_FILE = os.path.join(
    os.environ.get("MLSBENCH_PKG_DIR", "/workspace/oyster"),
    "custom_encoder.py",
)

# -- Encoder dimensions per environment (from launch_custom_template.py) --
# context_encoder_input_dim = obs_dim + action_dim + reward_dim
# context_encoder_output_dim = latent_dim * 2 (information bottleneck)
LATENT_DIM = 5
ENV_DIMS = {
    "cheetah-vel": (20, 6),     # obs_dim, action_dim
    "sparse-point-robot": (2, 2),
    "point-robot": (2, 2),
}
env_label = os.environ.get("ENV", "cheetah-vel")
obs_dim, action_dim = ENV_DIMS.get(env_label, (20, 6))
input_size = obs_dim + action_dim + 1  # +1 for reward
output_size = LATENT_DIM * 2           # information bottleneck


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


def count_encoder_params(module_path):
    """Import module, instantiate CustomContextEncoder, return param count."""
    mod = load_module(module_path, f"_check_{id(module_path)}")
    encoder = mod.CustomContextEncoder(
        hidden_sizes=[200, 200, 200],
        input_size=input_size,
        output_size=output_size,
    )
    return sum(p.numel() for p in encoder.parameters())


# -- Get template content --
config = json.loads(open(os.path.join(TASK_DIR, "config.json")).read())
editable_file = None
for f in config.get("files", []):
    if f.get("edit"):
        editable_file = f["filename"]
        break

mid_edit = load_module(os.path.join(TASK_DIR, "edits", "mid_edit.py"), "_mid_edit")
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
        params = count_encoder_params(tmp_path)
        baseline_params[bl_name] = params
        print(f"  baseline {bl_name}: {params:,} params")
    except Exception as e:
        print(f"  baseline {bl_name}: ERROR ({e})")
    finally:
        os.unlink(tmp_path)

if not baseline_params:
    print("WARNING: no baselines could be evaluated, skipping budget check")
    sys.exit(0)

max_baseline = max(baseline_params.values())
max_name = max(baseline_params, key=baseline_params.get)
budget = int(max_baseline * 1.1)

# -- Count params for agent's encoder --
agent_params = count_encoder_params(WORKSPACE_FILE)
print(f"\n  agent encoder: {agent_params:,} params")
print(f"  budget: {budget:,} (1.1 x {max_name}={max_baseline:,})")
print(f"  env={env_label}, input_size={input_size}, output_size={output_size}")

if agent_params > budget:
    print(f"\nFAILED: {agent_params:,} > {budget:,}", file=sys.stderr)
    sys.exit(1)

print("\nPASSED")
