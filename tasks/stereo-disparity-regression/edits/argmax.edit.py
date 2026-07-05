"""WEAK regress baseline for stereo-disparity-regression: hard argmax readout.

Non-differentiable, integer-only winner-take-all disparity selection: no sub-pixel
accuracy and no gradient to the cost volume -> high EPE.
Reference: vendor/stereo-matching/baselines/regress_argmax.py
"""

_FILE = "stereo-matching/solution/regress.py"

_CONTENT = '''\
def build_regressor():
    # WEAK baseline: hard argmax (winner-take-all) — non-differentiable,
    # integer-only, no sub-pixel accuracy.
    def regress(cost, disp_values):
        idx = torch.argmax(cost, dim=1)          # (B,H,W) integer level
        return disp_values[idx]
    return regress'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 32,
        "end_line": 38,
        "content": _CONTENT,
    },
]
