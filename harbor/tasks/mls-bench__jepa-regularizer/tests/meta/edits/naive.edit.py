"""Naive baseline (MSE-only) -- rigorous codebase edit ops.

Replaces the placeholder CustomRegularizer with a naive invariance-only loss.
No anti-collapse mechanism -- the model will likely collapse to trivial
representations, serving as a lower-bound baseline.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "eb_jepa/custom_regularizer.py"

# ── Replace editable region (lines 33-58) ───────────────────────────────────

_NAIVE_CLASS = """\
class CustomRegularizer(nn.Module):
    \"\"\"Naive MSE-only regularizer (no anti-collapse). Lower-bound baseline.\"\"\"

    def __init__(self):
        super().__init__()

    def forward(self, z1, z2):
        loss = F.mse_loss(z1, z2)
        return {"loss": loss, "invariance_loss": loss}


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: proj_output_dim, proj_hidden_dim.
CONFIG_OVERRIDES = {}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 33,
        "end_line": 58,
        "content": _NAIVE_CLASS,
    },
]
