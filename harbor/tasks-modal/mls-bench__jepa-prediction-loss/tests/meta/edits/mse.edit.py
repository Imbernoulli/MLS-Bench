"""MSE baseline -- rigorous codebase edit ops.

Replaces the CustomPredictionLoss placeholder with MSE loss,
equivalent to the standard SquareLossSeq behavior without projector.

Ops ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "eb_jepa/custom_prediction_loss.py"

# -- Replace CustomPredictionLoss class body (lines 36-54) --

_MSE_CLASS = """\
class CustomPredictionLoss(nn.Module):
    \"\"\"MSE prediction loss for temporal JEPA.\"\"\"

    def __init__(self):
        super().__init__()

    def forward(self, state, predicted):
        return F.mse_loss(state, predicted)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 36,
        "end_line": 54,
        "content": _MSE_CLASS,
    },
]
