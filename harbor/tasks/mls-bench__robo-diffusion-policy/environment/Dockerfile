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
FROM bohanlyu2022/mlsbench-harbor-cleandiffuser:latest

# See robo-diffusion-guidance: Daytona's Blackwell GPU requires a torch wheel
# containing sm_120 kernels; the published base's torch 2.5.1/cu121 does not.
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 'torch==2.7.1'

COPY _scaffold/ /workspace/
