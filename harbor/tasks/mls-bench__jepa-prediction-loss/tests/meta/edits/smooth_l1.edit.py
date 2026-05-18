"""Smooth L1 baseline -- rigorous codebase edit ops.

Replaces the CustomPredictionLoss placeholder with Smooth L1 loss,
a robust alternative to MSE that is less sensitive to outliers.

Ops ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "eb_jepa/custom_prediction_loss.py"

# -- Replace CustomPredictionLoss class body (lines 36-54) --

_SMOOTH_L1_CLASS = """\
class CustomPredictionLoss(nn.Module):
    \"\"\"Smooth L1 prediction loss for temporal JEPA.\"\"\"

    def __init__(self):
        super().__init__()

    def forward(self, state, predicted):
        return F.smooth_l1_loss(state, predicted)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 36,
        "end_line": 54,
        "content": _SMOOTH_L1_CLASS,
    },
]
