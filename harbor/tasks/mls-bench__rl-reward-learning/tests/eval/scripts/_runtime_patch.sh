#!/bin/bash
# Verifier-only runtime patch for rl-reward-learning's custom_irl.py.
#
# All three seeds of every eval label run concurrently and share
# ${SAVE_PATH}/irl_experts. The original scaffold publishes that expert-demo
# cache non-atomically (np.savez / model.save straight onto the final path)
# and races a bare exists() -> generate -> np.load sequence, so a sibling
# seed can np.load a half-written npz (zipfile.BadZipFile / EOFError), zero
# that seed, and leave a corrupt cache behind. The task template was fixed at
# the source (tasks/rl-reward-learning/edits/custom_template.py): temp-file +
# os.replace atomic publish, flock-serialized generate-or-load, and a
# corrupt-cache self-heal. This script re-applies exactly that fix to the
# workspace copy at eval time, for workspaces built from a pre-fix scaffold.
#
# Anchored on the exact old code blocks, applied all-or-none, and written
# back atomically: if the workspace file already contains the fix, or the
# fixed sections do not match the pre-fix scaffold verbatim (the edit guard
# forbids the agent from touching them, so this only means a future scaffold
# revision), the file is left byte-for-byte untouched. Agent edits live in
# the editable region (RewardNetwork / IRLAlgorithm) and are never modified.
#
# Runs under flock because several eval scripts patch the same file
# concurrently (precedent: mls-bench__cv-dbm-sampler _runtime_patch.sh).

_rl_demo_patch_lock="${RL_DEMO_PATCH_LOCK:-.mlsbench_demo_patch.lock}"
{
if command -v flock >/dev/null 2>&1; then
    flock 9
fi

python3 - <<'PY'
import os
from pathlib import Path

path = Path("custom_irl.py")
if not path.exists():
    print("[runtime-patch] custom_irl.py not found in cwd; nothing to do")
    raise SystemExit(0)

text = path.read_text()

NEW_MARKER = "_locked_demo_load"

OLD_MODEL_SAVE = '    model.save(os.path.join(demo_path, f"{env_id}_expert"))\n'
NEW_MODEL_SAVE = '    _atomic_save_model(model, demo_path, env_id)\n'

OLD_SAVEZ = '    np.savez(os.path.join(demo_path, f"{env_id}_demos.npz"), **demos)\n'
NEW_SAVEZ = '    _atomic_save_npz(demos, os.path.join(demo_path, f"{env_id}_demos.npz"))\n'

OLD_LOAD = '''def load_expert_demos(demo_path, env_id, device):
    """Load expert demonstrations, generating them if needed."""
    path = os.path.join(demo_path, f"{env_id}_demos.npz")
    if not os.path.exists(path):
        generate_expert_demos(demo_path, env_id)
    data = np.load(path)
    demos = {
        "obs": torch.tensor(data["obs"], dtype=torch.float32, device=device),
        "acts": torch.tensor(data["acts"], dtype=torch.float32, device=device),
        "next_obs": torch.tensor(data["next_obs"], dtype=torch.float32, device=device),
        "dones": torch.tensor(data["dones"], dtype=torch.float32, device=device),
    }
    print(f"Loaded {len(demos['obs'])} expert transitions from {path}")
    return demos
'''

NEW_LOAD = '''def load_expert_demos(demo_path, env_id, device):
    """Load expert demonstrations, generating them if needed.

    Concurrent runs (e.g. several seeds) share ``demo_path``, so the
    generate-or-load step is serialized with an inter-process file lock, the
    cache is only ever published atomically, and a corrupt cache file is
    regenerated under the lock (see the concurrency helpers further down).
    """
    path = os.path.join(demo_path, f"{env_id}_demos.npz")
    data = _locked_demo_load(demo_path, env_id, path)
    demos = {k: torch.tensor(data[k], dtype=torch.float32, device=device)
             for k in ("obs", "acts", "next_obs", "dones")}
    print(f"Loaded {len(demos['obs'])} expert transitions from {path}")
    return demos
'''

BANNER = (
    "# =====================================================================\n"
    "# FIXED: Main training loop\n"
    "# =====================================================================\n"
)

HELPERS = '''# =====================================================================
# FIXED: Demo-cache concurrency helpers (atomic publish + file lock)
# =====================================================================
def _atomic_save_model(model, demo_path, env_id):
    """Save the SB3 expert atomically (temp file in the same dir + os.replace)."""
    tmp = os.path.join(demo_path, f".{env_id}_expert.tmp-{os.getpid()}.zip")
    model.save(tmp)
    os.replace(tmp, os.path.join(demo_path, f"{env_id}_expert.zip"))


def _atomic_save_npz(arrays, final_path):
    """np.savez to a temp file in the same dir, then os.replace onto the final
    path, so a concurrent reader can never observe a partially written file."""
    tmp = os.path.join(os.path.dirname(final_path),
                       f".{os.path.basename(final_path)}.tmp-{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, final_path)


def _read_demo_arrays(path):
    """Fully materialize the demo arrays (validates the whole file on read)."""
    with np.load(path) as data:
        return {k: np.asarray(data[k]) for k in ("obs", "acts", "next_obs", "dones")}


def _locked_demo_load(demo_path, env_id, path):
    """Generate-or-load the shared demo cache under an inter-process lock.

    All runs sharing ``demo_path`` serialize here: the first process trains
    the expert and publishes the cache atomically while the others block on
    the lock and then just load it. A cache file that fails to load (e.g. a
    torn write left behind by a crashed/killed earlier run) is deleted and
    regenerated under the same lock.
    """
    import fcntl
    import zipfile

    os.makedirs(demo_path, exist_ok=True)
    with open(path + ".lock", "a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if not os.path.exists(path):
                generate_expert_demos(demo_path, env_id)
            try:
                return _read_demo_arrays(path)
            except (zipfile.BadZipFile, EOFError, KeyError, ValueError, OSError) as exc:
                print(f"Corrupt demo cache {path} ({exc!r}); regenerating...", flush=True)
                os.remove(path)
                generate_expert_demos(demo_path, env_id)
                return _read_demo_arrays(path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


'''

if NEW_MARKER in text:
    print("[runtime-patch] demo-cache concurrency fix already present; no-op")
    raise SystemExit(0)

if not (OLD_MODEL_SAVE in text and OLD_SAVEZ in text
        and OLD_LOAD in text and BANNER in text):
    print("[runtime-patch] pre-fix anchors not all found; "
          "leaving custom_irl.py untouched")
    raise SystemExit(0)

new_text = text.replace(OLD_MODEL_SAVE, NEW_MODEL_SAVE, 1)
new_text = new_text.replace(OLD_SAVEZ, NEW_SAVEZ, 1)
new_text = new_text.replace(OLD_LOAD, NEW_LOAD, 1)
# Insert the helpers right before the LAST occurrence of the main-loop
# banner (the agent's editable region sits above it; using the last
# occurrence keeps a pathological editable-region copy from catching the
# splice).
idx = new_text.rfind(BANNER)
new_text = new_text[:idx] + HELPERS + new_text[idx:]

tmp = path.with_name(".custom_irl.py.patch-tmp")
tmp.write_text(new_text)
os.replace(tmp, path)
print("[runtime-patch] applied demo-cache concurrency fix to custom_irl.py")
PY
} 9>"${_rl_demo_patch_lock}"
