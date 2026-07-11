"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "image-matting/solution/refine.py"
_CONTENT = '''def refine(coarse_alpha, image, trimap):
    import torch.nn.functional as F
    smooth = F.avg_pool2d(coarse_alpha.unsqueeze(1), 3, 1, 1).squeeze(1)
    refined = (coarse_alpha + 0.25 * (coarse_alpha - smooth)).clamp(0.0, 1.0)
    refined = torch.where(trimap <= 0.0, torch.zeros_like(refined), refined)
    return torch.where(trimap >= 1.0, torch.ones_like(refined), refined)'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 14, "content": _CONTENT},
]
