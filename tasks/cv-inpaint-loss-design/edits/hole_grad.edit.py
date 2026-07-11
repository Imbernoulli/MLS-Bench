"""Unmeasured full-protocol anchor candidate.

Retained for worker-side calibration. No score, ordering, or full-protocol runtime
is asserted before a complete 256x256, 100,000-step run.
"""

_FILE = "image-inpainting/solution/loss.py"

_CONTENT = '''    hole = mask
    valid = 1.0 - mask
    l_h = (torch.abs(out - gt) * hole).sum() / (hole.sum() + 1e-8)
    l_v = (torch.abs(out - gt) * valid).sum() / (valid.sum() + 1e-8)
    ox = out[:, :, :, 1:] - out[:, :, :, :-1]; tx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    oy = out[:, :, 1:, :] - out[:, :, :-1, :]; ty = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    hx = hole[:, :, :, 1:]; hy = hole[:, :, 1:, :]
    g_x = (torch.abs(ox - tx) * hx).sum() / (hx.sum() + 1e-8)
    g_y = (torch.abs(oy - ty) * hy).sum() / (hy.sum() + 1e-8)
    return 3.0 * l_h + 1.0 * l_v + 0.5 * (g_x + g_y)'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 16, "content": _CONTENT},
]
