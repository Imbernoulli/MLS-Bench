#!/bin/bash
# Verifier-only runtime patch for the meta-inner-loop-optimizer scaffold
# (learn2learn MetaDataset bookkeeping-cache write race).
#
# The task template tasks/meta-inner-loop-optimizer/edits/custom_template.py
# is fixed at the source, but the rendered scaffold is baked into the
# per-task image as /workspace/learn2learn/custom_maml.py. learn2learn's
# MetaDataset.load_bookkeeping() is a TOCTOU exists-check followed by a
# truncating pickle write; natively each (label, seed) run gets its own
# container FS, but under Harbor all (label, seed) eval subprocesses share
# ONE container FS, so concurrent writers/readers hit UnpicklingError or
# leave a persistently corrupt cache. This anchored patch retrofits the
# template's flock + self-heal wrapper onto a stale image-baked scaffold at
# eval time.
#
# Every replacement below is exact-old-block -> new-block and a no-op when
# the old block is absent (i.e. images rebuilt from a fixed render). Both
# anchors sit OUTSIDE the declared editable range (the InnerLoopOptimizer
# class), so agent edits never disturb them. Runs after the edit-range guard
# (score_task.py guard precedes run-evals), under an flock so concurrent
# (label, seed) eval processes don't race the rewrite itself.

_l2l_patch_lock="${L2L_RUNTIME_PATCH_LOCK:-.l2l_runtime_patch.lock}"
{
if command -v flock >/dev/null 2>&1; then
    flock 9
fi

python3 - <<'PY' || echo "[_runtime_patch] WARNING: custom_maml.py patch failed" >&2
from pathlib import Path

path = Path("learn2learn/custom_maml.py")
if path.exists():
    text = path.read_text()
    old = text

    helper = '''# =====================================================================
# FIXED: Concurrency-safe taskset construction
# =====================================================================
# NOTE: defined below the editable class on purpose — config.json's editable
# range for this file (the InnerLoopOptimizer class) is line-anchored, so
# fixed scaffolding added above it would shift the range.
def _bookkeeping_paths(dataset_name: str, root: str) -> List[str]:
    """learn2learn's bookkeeping cache pickles for this dataset's splits."""
    if dataset_name == "mini_imagenet":
        return [os.path.join(root, "mini-imagenet-bookkeeping-%s.pkl" % m)
                for m in ("train", "validation", "test")]
    if dataset_name == "cifar_fs":
        # CIFARFS maps mode "validation" -> "val" in its bookkeeping filename.
        return [os.path.join(root, "cifarfs-%s-bookkeeping.pkl" % m)
                for m in ("train", "val", "test")]
    return []


def get_tasksets_locked(dataset_name: str, n_way: int, n_shot: int, n_query: int,
                        root: str = os.environ.get("L2L_DATA_ROOT",
                                                   "/workspace/l2l_data")):
    """Serialize taskset construction across concurrent evaluation runs.

    learn2learn's ``MetaDataset.load_bookkeeping`` is an exists-check followed
    by a truncating pickle write of the bookkeeping cache — a TOCTOU race.
    When several (setting, seed) evaluation processes share one filesystem,
    a reader can observe a partially-written pickle (``UnpicklingError`` /
    ``EOFError``) or the racing writers can leave a persistently corrupt
    cache behind.

    This wrapper holds an inter-process ``flock`` on a lockfile next to the
    data root for the whole construct-or-load of all three splits, so exactly
    one process builds each cache and the rest load the finished pickle. A
    corrupt cache (e.g. left by a previously crashed writer) is self-healed
    by deleting this dataset's cache pickles and retrying once, still under
    the lock.
    """
    import fcntl

    try:
        os.makedirs(root, exist_ok=True)
        lock_file = open(os.path.join(root, ".l2l_bookkeeping.lock"), "a+")
    except OSError:
        # Read-only data root: the caches must already exist, constructions
        # only read them, and no serialization is needed.
        lock_file = None
    try:
        if lock_file is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            return get_tasksets(dataset_name, n_way, n_shot, n_query, root=root)
        except Exception:
            # Corrupt bookkeeping caches surface as unpickling errors inside
            # MetaDataset.__init__. Delete them and rebuild once while still
            # holding the lock; unrelated errors re-raise from the retry (and
            # immediately below when there was no cache file to remove).
            removed = False
            for pkl_path in _bookkeeping_paths(dataset_name, root):
                try:
                    os.remove(pkl_path)
                    removed = True
                except OSError:
                    pass
            if not removed:
                raise
            return get_tasksets(dataset_name, n_way, n_shot, n_query, root=root)
    finally:
        if lock_file is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            lock_file.close()


'''
    anchor = (
        "# =====================================================================\n"
        "# FIXED: Meta-Training and Evaluation Loop\n"
        "# =====================================================================\n"
    )
    if "def get_tasksets_locked(" not in text and anchor in text:
        text = text.replace(anchor, helper + anchor, 1)

    text = text.replace(
        "    # Load tasksets\n"
        "    train_tasks, val_tasks, test_tasks = get_tasksets(\n"
        "        DATASET_NAME, N_WAY, N_SHOT, N_QUERY\n"
        "    )\n",
        "    # Load tasksets (serialized against concurrent runs sharing this FS)\n"
        "    train_tasks, val_tasks, test_tasks = get_tasksets_locked(\n"
        "        DATASET_NAME, N_WAY, N_SHOT, N_QUERY\n"
        "    )\n",
        1,
    )

    if text != old:
        path.write_text(text)
PY
} 9>"${_l2l_patch_lock}"
