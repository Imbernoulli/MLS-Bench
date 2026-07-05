"""Weak multiscale baseline for deblur-multiscale (the naive answer = the default).

Single-scale (full-res only): one deblur pass at full resolution. Large motion blur is
harder to remove in one shot -> lower deblur PSNR. Reference:
vendor/image-deblur/baselines/scale_single.py
"""

_FILE = "image-deblur/solution/multiscale.py"

_CONTENT = '''def get_scale_config():
    # Single-scale (full-res only) -> harder on large blur, lower PSNR.
    return {"scales": 1}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
