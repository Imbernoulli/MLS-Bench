"""Log-linear (geometric) schedule baseline — uniform spacing in log domain.

Places time steps uniformly in log-space, giving denser coverage at
lower noise levels where fine details are resolved.
"""

_FILE = "dbim-codebase/ddbm/karras_diffusion.py"

_LOGLINEAR_SCHEDULE = """\
def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    import math
    log_max = math.log(t_max)
    log_min = math.log(max(t_min, 1e-10))
    sigmas = torch.exp(torch.linspace(log_max, log_min, n + 1))
    sigmas[-1] = t_min  # ensure exact terminal value
    return sigmas.to(device)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 310,
        "end_line": 321,
        "content": _LOGLINEAR_SCHEDULE,
    },
]
