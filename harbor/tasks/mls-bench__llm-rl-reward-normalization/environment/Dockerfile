# Per-task image = per-package harbor base + the task's `create` mid_edit
# scaffolding files. The base image (built by
# harbor_adapter/scripts/build_base_image.py) already contains the package
# source and any data deps. The per-task scaffolding below is what
# MLS-Bench's BaseAgent.setup_workspace() would inject via
# `tasks/<t>/edits/mid_edit.py` create-ops — these files form the initial
# state of the editable surface (e.g. `bench/custom_algorithm.py` with the
# function stub the agent fills in).
#
# Per-task secrets (eval scripts, parser, score_spec, leaderboard,
# baseline_edit_ops.json) AND the pristine diff baseline live under
# ../tests/ and ../solution/, which Harbor mounts at verify and oracle time
# only. The pristine baseline is intentionally NOT baked into the image:
# the agent runs as root and could otherwise tamper with an image-baked
# reference before verification.
FROM bohanlyu2022/mlsbench-harbor-verl:verl-fixes-20260901

# vLLM/verl's sleep path can ask the dynamic loader for CUDA's fully
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

COPY _scaffold/ /workspace/
