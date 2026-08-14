#!/bin/bash
# Verifier-only runtime patch for mls-bench__robo-diffusion-policy.
#
# The template fix makes the checkpoint dir seed-scoped
# (results/<pipeline>/<env>_s<seed>/ — parallel seeds of one label share
# this workspace and used to overwrite each other's ckpts) and deletes
# that dir at the start of the train phase, so an agent edit that ends
# training early WITHOUT error can never silently score a stale
# exact-step checkpoint left by an earlier run (e.g. the agent's own
# in-session test with the same default seed).
#
# That fix lives in tasks/robo-diffusion-policy/edits/custom_template.py and is baked into the
# workspace at image build time — images built before the fix still carry the
# racy block. This patch closes the gap at eval time.
#
# Why a copy instead of patching in place: pipelines/custom_policy.py is guarded byte-for-byte
# against tests/meta/pristine by score_task.py `guard`, and guard runs at the
# start of every verifier pass — mutating the agent's file would zero any
# verifier re-run. File creation is explicitly not a guard violation, so the
# patched code is written to pipelines/_verifier_custom_policy.py and the eval scripts run that copy.
#
# Anchoring: the replace matches the exact pre-fix block. If the baked file
# already carries the fix (image rebuilt from the fixed template) or an agent
# edit touched the block (protected region here, so present whenever guard passed), the replace is a no-op and the
# copy is byte-identical to the agent's file, i.e. identical semantics.

_mls_patch_lock=".mls_runtime_patch.lock"
{
if command -v flock >/dev/null 2>&1; then
    flock 9
fi

python3 - <<'PY'
import os
import tempfile
from pathlib import Path

src = Path('pipelines/custom_policy.py')
dst = Path('pipelines/_verifier_custom_policy.py')

OLD = (
    "    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'\n"
    "    if os.path.exists(save_path) is False:\n"
    "        os.makedirs(save_path)\n"
)

NEW = (
    "    save_path = f'results/{args.pipeline_name}/{args.task.env_name}_s{args.seed}/'  # per-seed dir: parallel seeds of one label must not share ckpts\n"
    "    if args.mode == \"train\": import shutil; shutil.rmtree(save_path, ignore_errors=True)  # fresh train dir: never silently score a stale exact-step ckpt\n"
    "    os.makedirs(save_path, exist_ok=True)\n"
)

text = src.read_text()
if OLD in text:
    text = text.replace(OLD, NEW, 1)

fd, tmp = tempfile.mkstemp(dir=str(dst.parent), prefix="." + dst.name + ".", suffix=".tmp")
try:
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, dst)
finally:
    if os.path.exists(tmp):
        os.unlink(tmp)
PY
} 9>"${_mls_patch_lock}"
