"""Do-nothing floor baseline for cv-harmonization-region-norm (the identity).

The input-copy identity: return the composite unchanged (NO harmonization) -> scores
exactly the composite-input foreground PSNR (fg_psnr == comp_fg_psnr, gain 0), the floor
any real harmonizer must beat.
Reference: vendor/image-harmonization/baselines/network_copy.py
"""

_FILE = "image-harmonization/solution/network.py"

_CONTENT = '''def get_network_config():
    # Input-copy identity: NO harmonization (the do-nothing floor).
    return {"arch": "copy"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 26, "end_line": 28, "content": _CONTENT},
]
