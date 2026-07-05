"""Weak residual baseline for deblur-global-residual (the naive answer = the default).

Predict the FULL image directly (no global residual). Harder optimisation at this
budget -> blurrier output, lower deblur PSNR. Reference:
vendor/image-deblur/baselines/residual_off.py
"""

_FILE = "image-deblur/solution/residual.py"

_CONTENT = '''def get_residual_config():
    # Predict the full image directly (no global residual) -> harder, blurrier.
    return {"global_residual": False}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
