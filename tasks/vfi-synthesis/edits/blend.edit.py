"""Weakest baseline (below the floor) for vfi-synthesis: naive linear blend (no motion).

0.5*(frame0 + frame2). Ignores motion entirely -> ghosts / doubles moving edges; its
interpolation PSNR collapses as motion grows (psnr == blend_psnr, gain 0). Reference:
vendor/video-frame-interp/baselines/synthesis_blend.py
"""

_FILE = "video-frame-interp/solution/synthesis.py"

_CONTENT = '''def get_synthesis_config():
    # Naive linear blend 0.5*(frame0+frame2) -> ghosts as motion grows, lowest PSNR.
    return {"method": "blend"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 34, "end_line": 36, "content": _CONTENT},
]
