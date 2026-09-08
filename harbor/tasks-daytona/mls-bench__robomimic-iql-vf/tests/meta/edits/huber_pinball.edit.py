"""Huber-pinball loss baseline -- rigorous codebase edit ops.

Combines Huber loss robustness with asymmetric (pinball) weighting.
Instead of asymmetric L2 (expectile) or asymmetric L1 (quantile),
this uses asymmetric Huber loss: quadratic for small errors, linear
for large errors, with different weights above/below the target.

This provides outlier robustness (from Huber) while still pushing
V(s) toward a high quantile of Q(s,a) (from pinball weighting).

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "robomimic/custom_iql_vf.py"

# ── Replace custom_vf_loss function (lines 21-39) ───────────────────────

_HUBER_PINBALL_LOSS = """\
def custom_vf_loss(vf_pred, q_target, quantile=0.9):
    \"\"\"Huber-pinball loss: asymmetric Huber for robust value estimation.

    Uses Huber loss (smooth L1) with asymmetric weighting. Quadratic
    for small errors, linear for large errors, with higher weight on
    under-estimation to push V(s) toward high quantiles of Q(s,a).
    \"\"\"
    diff = vf_pred - q_target
    weight = torch.where(diff > 0, 1.0 - quantile, quantile)
    huber = F.smooth_l1_loss(vf_pred, q_target, reduction='none', beta=1.0)
    return (weight * huber).mean()
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 21,
        "end_line": 38,
        "content": _HUBER_PINBALL_LOSS,
    },
]
