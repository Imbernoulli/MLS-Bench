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
FROM bohanlyu2022/mlsbench-harbor-llm-qat-runtime:latest

# A container sees every core the host has (`nproc` reports the host count)
# while its cgroup only grants `task.toml`'s `cpus`, so an unpinned
# OpenMP/MKL/PyTorch process starts one thread per host core and spends the
# quota on context switches — CPU-bound evals then run an order of magnitude
# slower and trip their per-command deadline. Bake the task's own CPU budget
# in as the default so the agent's development runs get it too;
# `tests/score_task.py` recomputes it from the live cgroup at verify time and
# divides it across the commands a wave runs concurrently.
ENV OMP_NUM_THREADS=16 \
    OMP_THREAD_LIMIT=16 \
    MKL_NUM_THREADS=16 \
    OPENBLAS_NUM_THREADS=16 \
    NUMEXPR_NUM_THREADS=16 \
    NUMEXPR_MAX_THREADS=16 \
    VECLIB_MAXIMUM_THREADS=16 \
    BLIS_NUM_THREADS=16

COPY _scaffold/ /workspace/
