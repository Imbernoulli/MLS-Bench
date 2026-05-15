"""Quantile regression (L1-based) baseline -- rigorous codebase edit ops.

Replaces the default expectile regression (asymmetric L2) with quantile
regression (asymmetric L1). Quantile regression is more robust to outlier
Q-values and estimates the conditional quantile directly.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "robomimic/custom_iql_vf.py"

# ── Replace custom_vf_loss function (lines 21-39) ───────────────────────

_QUANTILE_LOSS = """\
def custom_vf_loss(vf_pred, q_target, quantile=0.9):
    \"\"\"Quantile regression loss (asymmetric L1).

    Uses L1 loss instead of L2, providing robustness to outlier
    Q-value estimates while still pushing V(s) toward a high quantile.
    \"\"\"
    diff = vf_pred - q_target
    weight = torch.where(diff > 0, 1.0 - quantile, quantile)
    return (weight * diff.abs()).mean()
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 21,
        "end_line": 38,
        "content": _QUANTILE_LOSS,
    },
]
