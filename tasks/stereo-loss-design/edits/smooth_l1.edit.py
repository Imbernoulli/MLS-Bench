"""STRONG loss baseline for stereo-loss-design: smooth-L1 (Huber) disparity loss.

Quadratic for sub-pixel errors, linear (robust) for the few large errors, the
GC-Net/PSMNet choice -> low EPE.
Reference: vendor/stereo-matching/baselines/loss_smooth_l1.py
"""

_FILE = "stereo-matching/solution/loss.py"

_CONTENT = '''\
def build_loss():
    # STRONG: smooth-L1 (Huber) disparity loss (GC-Net/PSMNet).
    def loss_fn(disp_pred, disp_gt, valid):
        v = valid >= 0.5
        return F.smooth_l1_loss(disp_pred[v], disp_gt[v])
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
