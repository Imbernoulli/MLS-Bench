#!/usr/bin/env python3
"""One-off HOST-SIDE script: bakes the frozen normflows-density train/test 2-D
toy-density samples (moons / checkerboard / pinwheel / 8gaussians, seed=42,
n_train=30000, n_test=30000 -- the only combinations any shipped flow-* task
ever requests) into static .npz files.

Run this ONCE (or whenever the frozen regime below changes) from the repo
root:

    python3 holdout/normflows-density/generate_data.py

Output split (class-3 fix, 2026-07-05): the TRAIN array is legitimate to keep
agent-visible (an agent must still fit it), but the held-out TEST array is the
literal answer key for the scored metric, so it must NOT sit anywhere under
the agent-visible ``vendor/`` tree. This script now writes two separate
locations:

  * ``vendor/normflows-density/_flow_data/<target>_seed<seed>.npz``
    -- ``train_x`` ONLY. The task config marks these archives verifier-only;
    the frozen harness uses them for fitting after verification starts.
  * ``tasks/<task>/data/<target>_seed<seed>_test.npz`` (one copy per task
    that scores against that target)
    -- ``test_x`` ONLY. Lives under each task's ``data/`` dir, which Harbor
    (and score_task.py's ``_task`` symlink) stage into ``tests/meta/data``
    and mount at ``/workspace/_task/data`` ONLY at verification time -- see
    ``harbor_adapter/src/mls_bench/adapter.py::_stage_verifier_assets`` and
    ``harbor_adapter/src/mls_bench/task-template/tests/test.sh``. The agent
    never sees this file during its action session.

The resulting .npz files contain ONLY sampled (x1, x2) points -- no generator
code, no cell sizes / radii / noise scales / rotation rates. That was already
safe from a class-1/4 standpoint; the train/test split above additionally
closes the class-3 "held-out answer sitting in an agent-visible directory"
gap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # holdout/normflows-density
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import dgp  # noqa: E402  (holdout/normflows-density/dgp.py)

VENDOR_DATA_DIR = REPO_ROOT / "vendor" / "normflows-density" / "_flow_data"
TASKS_DIR = REPO_ROOT / "tasks"

# The (target, seed, n_train, n_test) combinations every shipped flow-*
# task's scripts/*.sh actually requests (n-train 30000, n-test 30000, seed 42).
# "circles" is defined in the DGP but never used by a shipped task, so it is
# intentionally NOT baked here.
N_TRAIN = 30000
N_TEST = 30000
SEED = 42
TARGETS = ("moons", "checkerboard", "pinwheel", "8gaussians")

# Which shipped tasks score against which target -- mirrors the --target
# argument in each tasks/<task>/scripts/*.sh. A test-only npz is written into
# every one of these tasks' data/ dirs (small files, cheap to duplicate; each
# task's data/ dir is staged/hidden independently by the adapter).
TASK_TARGETS = {
    "flow-arch-family": ("pinwheel",),
    "flow-autoregressive-coupling": ("8gaussians",),
    "flow-base-distribution": ("8gaussians",),
    "flow-batch-size": ("checkerboard",),
    "flow-conditioner-width": ("checkerboard",),
    "flow-coupling-transform": ("8gaussians", "checkerboard", "moons"),
    "flow-depth-permutation": ("moons",),
    "flow-learning-rate": ("moons",),
    "flow-masking-pattern": ("moons",),
    "flow-spline-bins": ("checkerboard",),
}


def _write_test_copies(target: str, seed: int, test_x: np.ndarray) -> None:
    for task, targets in TASK_TARGETS.items():
        if target not in targets:
            continue
        data_dir = TASKS_DIR / task / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        out_path = data_dir / f"{target}_seed{seed}_test.npz"
        np.savez(out_path, test_x=test_x)
        print(f"[{target}] wrote {out_path} (test={test_x.shape})")


def main() -> None:
    VENDOR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        arrays = dgp.make_dataset_arrays(target, N_TRAIN, N_TEST, SEED)
        out_path = VENDOR_DATA_DIR / f"{target}_seed{SEED}.npz"
        np.savez(out_path, train_x=arrays["train_x"])
        print(f"[{target}] wrote {out_path} (train={arrays['train_x'].shape})")
        _write_test_copies(target, SEED, arrays["test_x"])


if __name__ == "__main__":
    main()
