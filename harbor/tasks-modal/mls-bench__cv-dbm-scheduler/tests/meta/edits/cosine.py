"""Cosine schedule baseline — denser steps in mid-range noise levels.

From Nichol & Dhariwal (2021), Improved DDPM. The cosine mapping
concentrates time steps in the middle of the noise range, where
perceptual changes are largest.
"""

_FILE = "dbim-codebase/ddbm/karras_diffusion.py"

_COSINE_SCHEDULE = """\
def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    import math
    ramp = torch.linspace(0, 1, n + 1)
    cosine_ramp = (1 - torch.cos(ramp * math.pi)) / 2
    sigmas = t_max + (t_min - t_max) * cosine_ramp
    return sigmas.to(device)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 310,
        "end_line": 321,
        "content": _COSINE_SCHEDULE,
    },
]
