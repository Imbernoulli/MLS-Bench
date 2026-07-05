"""WEAK loss baseline for stereo-loss-design: squared-L2 disparity loss.

Squares the few large-error pixels (occlusions / discontinuities), so their
gradient dominates and biases toward an over-smoothed mean disparity -> higher EPE.
Reference: vendor/stereo-matching/baselines/loss_l2.py
"""

_FILE = "stereo-matching/solution/loss.py"

_CONTENT = '''\
def build_loss():
    # WEAK baseline: squared-L2 disparity loss — outlier-sensitive, over-smooths.
    def loss_fn(disp_pred, disp_gt, valid):
        v = valid >= 0.5
        return ((disp_pred[v] - disp_gt[v]) ** 2).mean()
    return loss_fn'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 33,
        "end_line": 38,
        "content": _CONTENT,
    },
]
