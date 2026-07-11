#!/usr/bin/env python3
"""One-off HOST-SIDE script: bakes the frozen mdn-density train/test RAW
samples (inverse_sine / two_branch / spiral / rot_bimodal, seed=42,
n_train=20000, n_test=20000 -- the only combination any shipped mdn-* task
requests) into static .npz files.

Run this ONCE (or whenever the frozen regime below changes) from the repo
root:

    python3 holdout/mdn-density/generate_data.py

Output split (class-3 fix, 2026-07-05): the TRAIN array is legitimate to keep
agent-visible (an agent must still fit it), but the held-out TEST array is the
literal answer key for the scored metric, so it must NOT sit anywhere under
the agent-visible ``vendor/`` tree. This script now writes two separate
locations:

  * ``vendor/mdn-density/_mdn_data/<target>_seed<seed>.npz``
    -- ``train_raw`` ONLY. Stays in vendor/ (agent-visible at all times);
    this is what the agent trains on.
  * ``tasks/<task>/data/<target>_seed<seed>_test.npz`` (one copy per task
    that scores against that target)
    -- ``test_raw`` ONLY. Lives under each task's ``data/`` dir, which Harbor
    (and score_task.py's ``_task`` symlink) stage into ``tests/meta/data``
    and mount at ``/workspace/_task/data`` ONLY at verification time -- see
    ``harbor_adapter/src/mls_bench/adapter.py::_stage_verifier_assets`` and
    ``harbor_adapter/src/mls_bench/task-template/tests/test.sh``. The agent
    never sees this file during its action session.

The resulting .npz files contain ONLY the RAW sampled (x, y) arrays -- no
generator code, no forward-map formulas, no noise-scale / rotation /
covariance constants. That was already safe from a class-1/4 standpoint; the
train/test split above additionally closes the class-3 "held-out answer
sitting in an agent-visible directory" gap. The (public) x-standardization
step is intentionally NOT applied here -- it stays in the agent-visible
``common.py`` since it is ordinary, disclosed preprocessing, not part of the
secret.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # holdout/mdn-density
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import dgp  # noqa: E402  (holdout/mdn-density/dgp.py)

VENDOR_DATA_DIR = REPO_ROOT / "vendor" / "mdn-density" / "_mdn_data"
TASKS_DIR = REPO_ROOT / "tasks"

# The one (n_train, n_test, seed) combination every shipped mdn-* task script
# requests.
N_TRAIN = 20000
N_TEST = 20000
SEED = 42
TARGETS_1D = ("inverse_sine", "two_branch", "spiral")
TARGETS_2D = ("rot_bimodal",)

# Which shipped tasks score against which target -- mirrors the --target
# argument in each tasks/<task>/scripts/*.sh. A test-only npz is written into
# every one of these tasks' data/ dirs (small files, cheap to duplicate; each
# task's data/ dir is staged/hidden independently by the adapter).
TASK_TARGETS = {
    "mdn-activation": ("spiral",),
    "mdn-component-balance": ("spiral",),
    "mdn-covariance": ("rot_bimodal",),
    "mdn-density-bench": ("inverse_sine", "two_branch", "spiral"),
    "mdn-initialization": ("spiral",),
    "mdn-learning-rate": ("inverse_sine",),
    "mdn-network-width": ("spiral",),
    "mdn-num-components": ("inverse_sine",),
    "mdn-trunk-depth": ("spiral",),
    "mdn-variance-floor": ("spiral",),
}


def _write_test_copies(target: str, seed: int, test_raw: np.ndarray) -> None:
    for task, targets in TASK_TARGETS.items():
        if target not in targets:
            continue
        data_dir = TASKS_DIR / task / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        out_path = data_dir / f"{target}_seed{seed}_test.npz"
        np.savez(out_path, test_raw=test_raw)
        print(f"[{target}] wrote {out_path} (test={test_raw.shape})")


def main() -> None:
    VENDOR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for target in TARGETS_1D:
        arrays = dgp.make_dataset_arrays(target, N_TRAIN, N_TEST, SEED)
        out_path = VENDOR_DATA_DIR / f"{target}_seed{SEED}.npz"
        np.savez(out_path, train_raw=arrays["train_raw"])
        print(f"[{target}] wrote {out_path} (train={arrays['train_raw'].shape})")
        _write_test_copies(target, SEED, arrays["test_raw"])
    for target in TARGETS_2D:
        arrays = dgp.make_dataset_arrays_2d(target, N_TRAIN, N_TEST, SEED)
        out_path = VENDOR_DATA_DIR / f"{target}_seed{SEED}.npz"
        np.savez(out_path, train_raw=arrays["train_raw"])
        print(f"[{target}] wrote {out_path} (train={arrays['train_raw'].shape})")
        _write_test_copies(target, SEED, arrays["test_raw"])


if __name__ == "__main__":
    main()
