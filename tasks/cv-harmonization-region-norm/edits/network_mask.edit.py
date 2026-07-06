"""Mask-conditioned baseline for cv-harmonization-region-norm (the strong region-aware design).

The MASK-CONDITIONED U-Net (composite RGB + the foreground mask): it knows exactly which
region is the pasted foreground and recolours only it while preserving the background --
the mask-conditioning that every real harmonizer relies on (DoveNet, Cong et al. CVPR 2020;
RainNet, Ling et al. CVPR 2021) -> the highest foreground PSNR (strong reference).
Reference: vendor/image-harmonization/baselines/network_mask.py
"""

_FILE = "image-harmonization/solution/network.py"

_CONTENT = '''def get_network_config():
    # Mask-CONDITIONED U-Net -> recolours only the foreground, highest PSNR.
    return {"arch": "mask"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 26, "end_line": 28, "content": _CONTENT},
]
