"""DDIM baseline: 20-step deterministic sampling.

Reference: Denoising Diffusion Implicit Models (Song et al., 2020),
https://arxiv.org/abs/2010.02502.

cleandiffuser implements DDIM as `solver="ddim"` inside DiscreteDiffusionSDE.sample().
Switch only the sampler fields in the YAML config: 20 inference steps is 5x
fewer than the DDPM default.
"""

_FILE = "CleanDiffuser/configs/custom/mujoco/mujoco.yaml"

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 17,
        "end_line": 17,
        "content": "sampling_steps: 20\n",
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 15,
        "end_line": 15,
        "content": "solver: ddim\n",
    },
]
