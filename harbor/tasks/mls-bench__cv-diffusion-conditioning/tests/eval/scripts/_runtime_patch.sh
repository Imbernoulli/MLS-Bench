#!/bin/bash
# Verifier-only runtime patch for cv-diffusion-conditioning.
#
# The task image bakes /workspace/diffusers-main/custom_train.py from
# edits/custom_template.py at render time, so template fixes do NOT reach
# already-built images. This rewrites the compute_fid scratch-dir handling in
# the agent's workspace copy at eval time. The rewritten region is OUTSIDE the
# declared editable range (config.json edit = lines 195-227,
# prepare_conditioning + ClassConditioner only), so on any guard-passing
# submission it is byte-identical to the pristine template.
#
# Bug being fixed: per-rank FID samples went to the shared
# tempfile.gettempdir()/fid_gen_<pid> and rank 0 merged EVERY /tmp/fid_gen_*
# dir it could glob. A crashed earlier label (labels run sequentially in ONE
# container and share /tmp; native Apptainer /tmp is the host's) leaves up to
# 8 dirs x 50k PNGs that later labels' FID silently ingests. Fix: scope
# everything under this run's $OUTPUT_DIR/_fid_tmp, wipe the base at the start
# of every compute_fid call, glob only under the base, clean up at the end.
#
# Anchored exact-block replacement, all-or-nothing; no-ops when the fix is
# already present (re-rendered image) or any anchor is missing.

_fid_patch_lock="${MLSBENCH_FID_PATCH_LOCK:-.mlsbench_fid_patch.lock}"
{
if command -v flock >/dev/null 2>&1; then
    flock 9
fi

python3 - <<'PY'
from pathlib import Path

# Eval scripts run with cwd = the package root (/workspace/diffusers-main).
path = Path("custom_train.py")
if not path.exists():
    raise SystemExit(0)
text = path.read_text()

OLD_GEN = (
    '    gen_dir = os.path.join(tempfile.gettempdir(), f"fid_gen_{os.getpid()}")\n'
    "    os.makedirs(gen_dir, exist_ok=True)\n"
)
NEW_GEN = (
    "    # All FID scratch dirs live under this run's OUTPUT_DIR so leftovers from\n"
    "    # a crashed or concurrent eval sharing /tmp can never leak into this FID.\n"
    "    fid_base = os.path.join(os.environ.get('OUTPUT_DIR', '/tmp/output'), '_fid_tmp')\n"
    "    if rank == 0:\n"
    "        if os.path.exists(fid_base):\n"
    "            shutil.rmtree(fid_base)\n"
    "        os.makedirs(fid_base)\n"
    "    if world_size > 1:\n"
    "        dist.barrier()\n"
    '    gen_dir = os.path.join(fid_base, f"fid_gen_{os.getpid()}")\n'
    "    os.makedirs(gen_dir, exist_ok=True)\n"
)

OLD_MERGE = (
    "    # Rank 0 gathers all images and computes FID\n"
    "    score = 0.0\n"
    "    if rank == 0:\n"
    "        # Merge images from all ranks into one dir\n"
    '        merged_dir = os.path.join(tempfile.gettempdir(), "fid_merged")\n'
    "        if os.path.exists(merged_dir):\n"
    "            shutil.rmtree(merged_dir)\n"
    "        os.makedirs(merged_dir)\n"
    "\n"
    "        # Copy from all per-rank dirs\n"
    "        for f in sorted(os.listdir(gen_dir)):\n"
    "            shutil.copy2(os.path.join(gen_dir, f), os.path.join(merged_dir, f))\n"
    "\n"
    "        if world_size > 1:\n"
    "            # Other ranks wrote to /tmp on the same node\n"
    "            import glob\n"
    '            for other_dir in glob.glob(os.path.join(tempfile.gettempdir(), "fid_gen_*")):\n'
    "                if other_dir == gen_dir:\n"
    "                    continue\n"
    "                for f in os.listdir(other_dir):\n"
    "                    shutil.copy2(os.path.join(other_dir, f), os.path.join(merged_dir, f))\n"
)
NEW_MERGE = (
    "    # Rank 0 gathers all images and computes FID\n"
    "    score = 0.0\n"
    "    if rank == 0:\n"
    "        # Merge images from all ranks into one dir\n"
    '        merged_dir = os.path.join(fid_base, "fid_merged")\n'
    "        if os.path.exists(merged_dir):\n"
    "            shutil.rmtree(merged_dir)\n"
    "        os.makedirs(merged_dir)\n"
    "\n"
    "        # Copy from all per-rank dirs\n"
    "        for f in sorted(os.listdir(gen_dir)):\n"
    "            shutil.copy2(os.path.join(gen_dir, f), os.path.join(merged_dir, f))\n"
    "\n"
    "        if world_size > 1:\n"
    "            # Other ranks wrote their own subdirs under this run's fid_base\n"
    "            import glob\n"
    '            for other_dir in glob.glob(os.path.join(fid_base, "fid_gen_*")):\n'
    "                if other_dir == gen_dir:\n"
    "                    continue\n"
    "                for f in os.listdir(other_dir):\n"
    "                    shutil.copy2(os.path.join(other_dir, f), os.path.join(merged_dir, f))\n"
)

OLD_CLEAN = (
    "    # Clean up per-rank dir\n"
    "    shutil.rmtree(gen_dir, ignore_errors=True)\n"
    "\n"
    "    if world_size > 1:\n"
    "        dist.barrier()\n"
    "\n"
    "    model.train()\n"
    "    return score\n"
)
NEW_CLEAN = (
    "    # Clean up per-rank dir\n"
    "    shutil.rmtree(gen_dir, ignore_errors=True)\n"
    "\n"
    "    if world_size > 1:\n"
    "        dist.barrier()\n"
    "\n"
    "    # Remove this run's FID scratch base (merged_dir + per-rank dirs are gone)\n"
    "    if rank == 0:\n"
    "        shutil.rmtree(fid_base, ignore_errors=True)\n"
    "\n"
    "    model.train()\n"
    "    return score\n"
)

BLOCKS = [(OLD_GEN, NEW_GEN), (OLD_MERGE, NEW_MERGE), (OLD_CLEAN, NEW_CLEAN)]

if "_fid_tmp" in text:
    raise SystemExit(0)  # already patched / fixed template baked
if any(text.count(old) != 1 for old, _ in BLOCKS):
    # Anchor missing or ambiguous (unexpected template variant) — leave the
    # file untouched rather than risk a partially-applied rewrite.
    raise SystemExit(0)
for old, new in BLOCKS:
    text = text.replace(old, new, 1)
path.write_text(text)
print("[runtime-patch] custom_train.py: FID scratch dirs scoped under "
      "$OUTPUT_DIR/_fid_tmp (was shared /tmp fid_gen_* glob)", flush=True)
PY
} 9>"${_fid_patch_lock}"
