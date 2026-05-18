"""DPM-Solver++ 2M baseline: 10-step high-order ODE sampling.

Reference: DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic
Models (Lu et al., 2022), https://arxiv.org/abs/2211.01095.

cleandiffuser implements DPM-Solver++ 2M as `solver="ode_dpmsolver++_2M"`
inside DiscreteDiffusionSDE.sample(). This baseline uses 10 sampling steps,
10x fewer than DDPM and 2x fewer than DDIM.
"""

_FILE = "CleanDiffuser/configs/custom/mujoco/mujoco.yaml"

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 17,
        "end_line": 17,
        "content": "sampling_steps: 10\n",
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 15,
        "end_line": 15,
        "content": "solver: ode_dpmsolver++_2M\n",
    },
]
