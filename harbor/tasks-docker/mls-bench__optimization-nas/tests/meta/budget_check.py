"""Parameter budget check for optimization-nas (standalone).

Run by tools.py before training:
    python $MLSBENCH_TASK_DIR/budget_check.py

Instantiates each baseline's NASOptimizer with a mock BenchmarkAPI,
runs 2 search steps so any lazily-initialized predictor networks are
constructed, and counts trainable parameters in any torch.nn.Module
attached to the optimizer.

Asserts the agent's optimizer doesn't exceed 1.05x the largest baseline,
with a floor of 100_000 params — enough room for a small BANANAS-style
neural predictor (a 2-layer MLP on 30-dim path encoding) plus slack,
but rules out large surrogate networks that would let an agent circumvent
the K=30 validation-query budget via learned extrapolation.
"""
import importlib.util
import json
import os
import random
import sys
import tempfile

import numpy as np

TASK_DIR = os.environ.get("MLSBENCH_TASK_DIR", "/workspace/_task")
WORKSPACE_FILE = os.path.join(
    os.environ.get("MLSBENCH_PKG_DIR", "/workspace/naslib"),
    "custom_nas_search.py",
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


class MockAPI:
    """Minimal stand-in for BenchmarkAPI — enough to run 2 search_step calls."""
    def __init__(self, budget=30, seed=42):
        self.query_count = 0
        self.query_budget = budget
        self._rng = random.Random(seed)

    @property
    def remaining_budget(self):
        return max(0, self.query_budget - self.query_count)

    def query_val_accuracy(self, op_indices):
        self.query_count += 1
        # Return a plausible NAS-Bench-201 val accuracy (~60-90)
        return 60.0 + 30.0 * self._rng.random()


def count_params_for_module(module_path):
    """Load module, instantiate NASOptimizer, run 2 search steps, count params."""
    mod = load_module(module_path, f"_check_{id(module_path)}")
    cls = getattr(mod, "NASOptimizer", None)
    if cls is None:
        print(f"  WARNING: No NASOptimizer found in {module_path}")
        return 0
    api = MockAPI(budget=30, seed=42)
    try:
        optimizer = cls(api=api, num_epochs=30, seed=42)
    except Exception as e:
        print(f"  WARNING: NASOptimizer() failed ({e})")
        return 0

    # Run 2 steps so any predictor networks get lazily initialized.
    for step in range(2):
        try:
            optimizer.search_step(step)
        except Exception as e:
            print(f"  WARNING: search_step({step}) raised {e}")
            break
    return count_torch_params(optimizer)


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
# Floor: 100k params. BANANAS uses a small MLP ensemble (~1-5k per predictor,
# typically <20k total). 100k floor gives room for agent's own predictor plus
# slack, but blocks agent from using a large DNN to extrapolate past K=30.
budget = max(int(max_baseline * 1.05), 100_000)

agent_params = count_params_for_module(WORKSPACE_FILE)
print(f"\n  agent optimizer: {agent_params} torch params")
print(f"  budget: {budget} (1.05 x {max_name}={max_baseline}, floor=100000)")

if agent_params > budget:
    print(f"\nFAILED: {agent_params} > {budget}", file=sys.stderr)
    sys.exit(1)

print("\nPASSED")
