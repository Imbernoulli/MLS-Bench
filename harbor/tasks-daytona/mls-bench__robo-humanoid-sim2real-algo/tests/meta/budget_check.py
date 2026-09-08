"""Parameter budget check for robo-humanoid-sim2real-algo (standalone).

Run by tools.py before training: python /workspace/_task/budget_check.py
Imports each baseline's edit ops, applies them to the template, imports the
resulting ActorCritic, instantiates with the canonical XBot input/output
dims, counts params, and asserts the agent's ActorCritic doesn't exceed
1.05x the largest baseline. Caps the algorithmic-improvement axis so an
agent can't just scale the network to win.

Skipped during 'train' phase only (not during eval) since we need the
edits to be applied to the workspace already.
"""
import importlib.util
import json
import os
import sys
import tempfile

import torch
import torch.nn as nn

TASK_DIR = "/workspace/_task"
WORKSPACE_FILE = "/workspace/humanoid-gym/humanoid/algo/ppo/actor_critic_custom.py"

# XBot canonical dims from XBotLCfg.env (frame_stack=15, num_single_obs=47,
# c_frame_stack=3, single_num_privileged_obs=73, num_actions=12).
NUM_ACTOR_OBS = 15 * 47          # 705
NUM_CRITIC_OBS = 3 * 73          # 219
NUM_ACTIONS = 12
ACTOR_HIDDEN_DIMS = [512, 256, 128]
CRITIC_HIDDEN_DIMS = [768, 256, 128]
INIT_NOISE_STD = 1.0


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


def count_params(module_path):
    mod = load_module(module_path, f"_chk_{abs(hash(module_path))}")
    AC = getattr(mod, "ActorCritic")
    model = AC(
        num_actor_obs=NUM_ACTOR_OBS,
        num_critic_obs=NUM_CRITIC_OBS,
        num_actions=NUM_ACTIONS,
        actor_hidden_dims=ACTOR_HIDDEN_DIMS,
        critic_hidden_dims=CRITIC_HIDDEN_DIMS,
        init_noise_std=INIT_NOISE_STD,
        activation=nn.ELU(),
    )
    return sum(p.numel() for p in model.parameters())


# Skip if we're not in train phase (the workspace file may not exist yet
# or may not be the agent's actor_critic_custom.py).
if not os.path.exists(WORKSPACE_FILE):
    print(f"WARNING: {WORKSPACE_FILE} not found, skipping budget check")
    sys.exit(0)

config = json.loads(open(os.path.join(TASK_DIR, "config.json")).read())
editable_file_relpath = "humanoid-gym/humanoid/algo/ppo/actor_critic_custom.py"

# -- Load template (mid_edit creates the file with the upstream class body) --
mid_edit = load_module(os.path.join(TASK_DIR, "edits", "mid_edit.py"), "_mid_edit")
template_content = next(
    op["content"] for op in mid_edit.OPS
    if op.get("op") == "create" and op.get("file") == editable_file_relpath
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
    modified = apply_ops(template_lines, ops, editable_file_relpath)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("\n".join(modified))
        tmp = f.name
    try:
        p = count_params(tmp)
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

# -- Count agent params (file already has agent edits applied by the harness) --
try:
    agent_params = count_params(WORKSPACE_FILE)
except Exception as e:
    print(f"\nFAILED to instantiate agent ActorCritic: {e}", file=sys.stderr)
    sys.exit(1)

print(f"\n  agent model: {agent_params:,} params")
print(f"  budget: {budget:,} (1.05 x {max_bl}={baseline_params[max_bl]:,})")

if agent_params > budget:
    print(f"\nFAILED: {agent_params:,} > {budget:,}", file=sys.stderr)
    sys.exit(1)

print("\nPASSED")
