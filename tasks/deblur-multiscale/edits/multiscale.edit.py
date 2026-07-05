"""Strong multiscale baseline for deblur-multiscale (the good answer).

3-scale coarse-to-fine pyramid with SHARED weights (the SRN recurrence, Tao et al.
CVPR 2018; cf. DeepDeblur, Nah et al. CVPR 2017): deblur a coarse image first (where
the blur is smaller / easier), then refine at each finer scale -> sharper on large
blur, higher deblur PSNR. Reference: vendor/image-deblur/baselines/scale_multi.py
"""

_FILE = "image-deblur/solution/multiscale.py"

_CONTENT = '''def get_scale_config():
    # 3-scale coarse-to-fine pyramid (shared weights, SRN) -> sharper, higher PSNR.
    return {"scales": 3}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
