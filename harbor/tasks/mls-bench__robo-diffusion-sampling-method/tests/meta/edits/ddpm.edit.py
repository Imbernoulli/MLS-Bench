"""DDPM baseline — 100-step ancestral sampling.

Reference: Denoising Diffusion Probabilistic Models (Ho et al., 2020),
https://arxiv.org/abs/2006.11239.

cleandiffuser implements DDPM as `solver="ddpm"` inside DiscreteDiffusionSDE.sample()
(see cleandiffuser/diffusion/diffusionsde.py:543). The default template config
already sets this, so this baseline is a no-op edit.
"""

_FILE = "CleanDiffuser/pipelines/custom_sampling_method.py"

OPS = []
