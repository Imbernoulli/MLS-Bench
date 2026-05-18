"""Uniform (linear) schedule baseline — evenly spaced time steps."""

_FILE = "dbim-codebase/ddbm/karras_diffusion.py"

_UNIFORM_SCHEDULE = """\
def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    return torch.linspace(t_max, t_min, n + 1).to(device)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 310,
        "end_line": 321,
        "content": _UNIFORM_SCHEDULE,
    },
]
