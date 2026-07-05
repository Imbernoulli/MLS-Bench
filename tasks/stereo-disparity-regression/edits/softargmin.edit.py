"""STRONG regress baseline for stereo-disparity-regression: GC-Net soft-argmin.

Softmax over the D disparity costs -> probability distribution, then output its
expectation sum_d d * p(d). Differentiable + sub-pixel accurate -> low EPE.
Reference: vendor/stereo-matching/baselines/regress_softargmin.py
"""

_FILE = "stereo-matching/solution/regress.py"

_CONTENT = '''\
def build_regressor():
    # STRONG: GC-Net soft-argmin (probability-weighted disparity expectation).
    def regress(cost, disp_values):
        prob = F.softmax(cost, dim=1)
        dv = disp_values.view(1, -1, 1, 1)
        return torch.sum(prob * dv, dim=1)
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
