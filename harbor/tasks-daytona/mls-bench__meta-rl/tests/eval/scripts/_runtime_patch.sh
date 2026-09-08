#!/bin/bash
# Verifier-only runtime patch for the meta-rl launcher (concurrent-seed
# artifact race).
#
# The task template tasks/meta-rl/edits/launch_custom_template.py is fixed
# at the source, but the rendered launcher is baked into the per-task image
# as /workspace/oyster/launch_custom.py: it calls setup_logger() with
# exp_id=None, and rlkit's create_log_dir() then names the run directory
# with a second-resolution timestamp and NO seed component. Concurrent seeds
# of a label launch within milliseconds, so they share one output dir,
# truncating each other's progress.csv (rlkit's logger opens it in 'w' mode)
# and racing the params.pkl torch.save. Scores are parsed from stdout and
# are unaffected; the on-disk artifacts are garbage. The patch threads the
# CLI seed into a seed-qualified exp_id so each seed gets its own directory.
#
# Every replacement below is exact-old-block -> new-block and a no-op when
# the old block is absent (i.e. images rebuilt from a fixed render).
# launch_custom.py is declared read-only, so no agent edit can disturb the
# anchors. Runs after the edit-range guard (score_task.py guard precedes
# run-evals), under an flock so concurrent (label, seed) eval processes
# don't race the rewrite itself.

_metarl_patch_lock="${METARL_RUNTIME_PATCH_LOCK:-.metarl_runtime_patch.lock}"
{
if command -v flock >/dev/null 2>&1; then
    flock 9
fi

python3 - <<'PY' || echo "[_runtime_patch] WARNING: launch_custom.py patch failed" >&2
from pathlib import Path

path = Path("launch_custom.py")
if path.exists():
    text = path.read_text()
    old = text

    text = text.replace(
        "from rlkit.launchers.launcher_util import setup_logger\n",
        "from rlkit.launchers.launcher_util import setup_logger, create_simple_exp_name\n",
        1,
    )
    text = text.replace(
        "def experiment(variant):\n",
        "def experiment(variant, seed=42):\n",
        1,
    )
    text = text.replace(
        "    os.environ['DEBUG'] = str(int(variant['util_params']['debug']))\n"
        "    exp_id = 'debug' if variant['util_params']['debug'] else None\n"
        "    experiment_log_dir = setup_logger(\n",
        "    os.environ['DEBUG'] = str(int(variant['util_params']['debug']))\n"
        "    # Seed-qualify the run directory. rlkit's create_log_dir() names the run\n"
        "    # dir with a second-resolution timestamp only (output/<env>/<timestamp>),\n"
        "    # so concurrent seeds launched within the same second would share one\n"
        "    # directory — truncating each other's progress.csv (the logger opens it\n"
        "    # in 'w' mode) and racing the params.pkl torch.save. setup_logger uses\n"
        "    # exp_id verbatim as the directory name, so qualify it with the seed.\n"
        "    run_name = '%s_seed%d' % (create_simple_exp_name(), seed)\n"
        "    exp_id = ('debug_seed%d' % seed) if variant['util_params']['debug'] else run_name\n"
        "    experiment_log_dir = setup_logger(\n",
        1,
    )
    text = text.replace(
        "    experiment(variant)\n",
        "    experiment(variant, seed)\n",
        1,
    )

    if text != old:
        path.write_text(text)
PY
} 9>"${_metarl_patch_lock}"
