"""Strong reference (SOTA) for vfi-synthesis: learned flow + refinement synthesis.

Super-SloMo (Jiang et al., CVPR 2018) family: estimate the flow to t=0.5, backward-warp
both frames, then a refinement U-Net predicts a soft per-pixel VISIBILITY mask + a residual
to resolve occlusion/disocclusion. Highest interpolation PSNR in every motion setting, and
its margin over pure flow-warp WIDENS with motion. Reference:
vendor/video-frame-interp/baselines/synthesis_learned.py
"""

_FILE = "video-frame-interp/solution/synthesis.py"

_CONTENT = '''def get_synthesis_config():
    # Learned flow + refinement (visibility mask + residual) -> highest PSNR (SOTA).
    return {"method": "learned"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 34, "end_line": 36, "content": _CONTENT},
]
