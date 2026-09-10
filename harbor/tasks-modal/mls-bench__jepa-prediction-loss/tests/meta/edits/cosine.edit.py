"""Cosine similarity baseline -- rigorous codebase edit ops.

Replaces the CustomPredictionLoss placeholder with cosine similarity loss,
which measures angular distance in representation space rather than magnitude.

Ops ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "eb_jepa/custom_prediction_loss.py"

# -- Replace CustomPredictionLoss class body (lines 36-54) --

_COSINE_CLASS = """\
class CustomPredictionLoss(nn.Module):
    \"\"\"Cosine similarity prediction loss for temporal JEPA.\"\"\"

    def __init__(self):
        super().__init__()

    def forward(self, state, predicted):
        s = F.normalize(state, dim=1)
        p = F.normalize(predicted, dim=1)
        return (1 - (s * p).sum(dim=1)).mean()
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 36,
        "end_line": 54,
        "content": _COSINE_CLASS,
    },
]
