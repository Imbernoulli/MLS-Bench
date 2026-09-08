"""Karras schedule baseline — power-law noise schedule from Karras et al. (2022).

Concentrates steps at higher noise levels using inverse power-law ramp
with rho=7, following the EDM paper.
"""

_FILE = "dbim-codebase/ddbm/karras_diffusion.py"

_KARRAS_SCHEDULE = """\
def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    rho = 7.0
    ramp = torch.linspace(0, 1, n + 1)
    min_inv_rho = t_min ** (1 / rho)
    max_inv_rho = t_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    sigmas[-1] = t_min  # ensure exact terminal value
    return sigmas.to(device)
"""

# Line numbers are POST-pre_edit AND POST-mid_edit. pre_edit at
# karras_diffusion.py:275-279 (+9 shift) puts `def get_sigmas_karras` at line
# 310; mid_edit replaces lines 310-320 (11 lines) with the 12-line template,
# so the template body sits at lines 310-321 in the workspace copy that this
# baseline edit runs against. Same shift on docker.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 310,
        "end_line": 321,
        "content": _KARRAS_SCHEDULE,
    },
]
