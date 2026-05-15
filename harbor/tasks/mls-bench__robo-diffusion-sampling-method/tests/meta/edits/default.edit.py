"""Default baseline — unmodified template (DDPM, 100 steps).

The template defaults to DDPM sampling with 100 steps via the YAML config
(`solver: ddpm`, `sampling_steps: 100`). This is the canonical high-quality
diffusion-model sampling regime.
"""

_FILE = "CleanDiffuser/pipelines/custom_sampling_method.py"

OPS = []
