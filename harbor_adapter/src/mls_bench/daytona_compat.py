"""Daytona-only compatibility patches for rendered Harbor tasks.

The native ``tasks/<task>/`` tree is the MLS-Bench source of truth and should
not change just because a remote provider needs a different user-space
runtime.  This module is called while rendering Harbor bundles, so provider
workarounds stay in the Harbor/Daytona integration layer: only the rendered
``environment/Dockerfile`` and the verifier's copies of eval scripts are
touched, never the agent-editable workspace or the native task scripts.

Every patch is idempotent so re-rendering a task does not accumulate layers or
duplicate script arguments.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


VERL_TASKS = frozenset(
    {
        "llm-rl-advantage",
        "llm-rl-kl-estimator",
        "llm-rl-reward-normalization",
        "llm-rl-importance-sampling",
    }
)

# Base image rebuilt with the package-level verl pre_edit fixes
# (``vendor/pkg_configs/verl/pre_edit.py``).  The rendered pristine manifest
# must describe exactly this image, so the tag is pinned instead of ``latest``.
VERL_HARBOR_BASE_IMAGE = "bohanlyu2022/mlsbench-harbor-verl:verl-fixes-20260901"

_LIBCUDART_ALIAS = r"""# vLLM/verl's sleep path can ask the dynamic loader for CUDA's fully
# versioned runtime name (for example ``libcudart.so.12.9.79``).  The CUDA
# image ships that file under /usr/local/cuda, but ldconfig only indexes its
# SONAME (libcudart.so.12), so the fully-versioned lookup fails once Harbor's
# verifier sanitizes LD_LIBRARY_PATH.  Install an ordinary system-library
# alias for every full libcudart version present in the image.
RUN set -eux; \
    found=0; \
    for runtime in /usr/local/cuda*/targets/x86_64-linux/lib/libcudart.so.[0-9]*.[0-9]*.[0-9]*; do \
        if [ -f "${runtime}" ]; then \
            found=1; \
            ln -sfn "${runtime}" "/usr/lib/x86_64-linux-gnu/$(basename "${runtime}")"; \
        fi; \
    done; \
    test "${found}" -eq 1; \
    ldconfig
"""

# verl worker-process settings for the Daytona sandbox.  The sandbox exposes a
# small CPU/RAM cgroup, so the verifier copies of ``train.sh`` cap the
# dataloader/agent/reward worker processes.  The native script is unchanged;
# the arguments do not alter the training algorithm or its hyperparameters.
_VERL_TRAIN_ARGS = (
    (
        "    data.shuffle=False \\\n",
        "    data.dataloader_num_workers=0 \\\n",
    ),
    (
        "    actor_rollout_ref.rollout.enforce_eager=True \\\n",
        "    actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS:-1} \\\n",
    ),
    (
        "    actor_rollout_ref.ref.fsdp_config.param_offload=False \\\n",
        "    reward.num_workers=${REWARD_NUM_WORKERS:-1} \\\n",
    ),
)

_VERL_TRAIN_SCRIPTS = (
    Path("eval/scripts/train.sh"),
    Path("meta/scripts/train.sh"),
)

# Package files whose package-level pre_edit fix postdates the pinned base
# image.  They are copied from the pre-edited package source into the task
# scaffold (``COPY _scaffold/ /workspace/``) so the workspace matches the
# rendered pristine manifest without rebuilding the 36 GB image.  Drop an
# entry once the base image has been rebuilt with the fix.
VERL_OVERLAY_FILES = (
    # process_validation_metrics: skip None extra-info values (mixed scorers)
    "verl/verl/trainer/ppo/metric_utils.py",
    # verl #2490 dynamic_bsz DP-group ops (image predates them)
    "verl/verl/workers/actor/dp_actor.py",
)


def _patch_verl_dockerfile(docker_text: str) -> str:
    docker_text = re.sub(
        r"(?m)^FROM bohanlyu2022/mlsbench-harbor-verl:latest$",
        f"FROM {VERL_HARBOR_BASE_IMAGE}",
        docker_text,
    )
    if 'test "${found}" -eq 1;' not in docker_text:
        marker = "COPY _scaffold/"
        insertion = _LIBCUDART_ALIAS + "\n"
        if marker in docker_text:
            docker_text = docker_text.replace(marker, insertion + marker, 1)
        else:
            docker_text = docker_text.rstrip() + "\n\n" + insertion
    return docker_text


def _patch_verl_train_script(text: str) -> str:
    for anchor, argument in _VERL_TRAIN_ARGS:
        if argument in text or anchor not in text:
            continue
        text = text.replace(anchor, anchor + argument, 1)
    # An earlier Daytona workaround disabled vLLM's cache-engine sleep path;
    # the libcudart alias above makes it unnecessary.  Drop it if present.
    text = text.replace(
        "    actor_rollout_ref.rollout.free_cache_engine=False \\\n", ""
    )
    return text


def apply_daytona_compatibility(
    task_id: str,
    *,
    dockerfile: Path,
    tests_dir: Path,
    scaffold_dir: Path | None = None,
    package_source: Path | None = None,
) -> bool:
    """Apply provider-only fixes to one rendered Harbor task.

    ``scaffold_dir``/``package_source`` enable the package-file overlay for
    fixes that are not yet baked into the pinned base image.  Returns ``True``
    when a Daytona compatibility patch applies to the task.
    """

    if task_id not in VERL_TASKS:
        return False

    if scaffold_dir is not None and package_source is not None:
        for relative in VERL_OVERLAY_FILES:
            # Scaffold paths are workspace-relative (``<package dir>/...``);
            # the materialized package source is that package dir itself.
            source = package_source / Path(*Path(relative).parts[1:])
            if not source.is_file():
                raise FileNotFoundError(
                    f"{task_id}: overlay source missing from package source: {source}"
                )
            target = scaffold_dir / relative
            if target.exists():
                # The task's own mid_edit already ships a (task-specific)
                # version of this file; it was derived from the pre-edited
                # source, so keep it.
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    docker_text = dockerfile.read_text()
    patched = _patch_verl_dockerfile(docker_text)
    if patched != docker_text:
        dockerfile.write_text(patched)

    for relative in _VERL_TRAIN_SCRIPTS:
        script = tests_dir / relative
        if not script.is_file():
            continue
        text = script.read_text()
        patched = _patch_verl_train_script(text)
        if patched != text:
            script.write_text(patched)
    return True
