"""Strong residual baseline for deblur-global-residual (the good answer).

Predict a GLOBAL RESIDUAL: sharp = blurry + net(blurry). The net only models the
high-frequency deblur correction, which is easier to optimise and yields sharper
restorations (DeepDeblur Nah et al. CVPR 2017; SRN Tao et al. CVPR 2018; MPRNet
Zamir et al. CVPR 2021). Reference: vendor/image-deblur/baselines/residual_on.py
"""

_FILE = "image-deblur/solution/residual.py"

_CONTENT = '''def get_residual_config():
    # Global residual ON: sharp = blurry + net(blurry) (easy correction, sharper).
    return {"global_residual": True}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
