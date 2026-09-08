"""Ridge regression baseline for mutation effect prediction.

ProteinNPT-inspired "Embeddings" linear baseline: a single nn.Linear head
trained end-to-end with AdamW (weight_decay=5e-2, equivalent to L2
regularization on the linear weights).

Reference: ProteinNPT / ProteinGym supervised embedding baselines motivate a
learnable linear head over PLM features. The exact weight_decay here is a
task-local choice.
"""

_FILE = "ProteinGym/custom_mutation_pred.py"

_MODEL = """\

class MutationPredictor(nn.Module):
    \"\"\"Ridge regression as a single nn.Linear, trained with AdamW (wd=5e-2).

    Uses delta_embedding (mutant - wildtype) as the input feature, so the
    model learns a linear mapping from the mutation-induced embedding shift
    to the fitness score.
    \"\"\"

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, embedding, delta_embedding):
        return self.linear(delta_embedding).squeeze(-1)

"""

_OVERRIDES = """\
    CONFIG_OVERRIDES = {'weight_decay': 5e-2}
"""

# NOTE: ops are applied sequentially. Apply the higher line-number replace
# FIRST so the [108, 137] replace target stays correct.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 345,
        "end_line": 347,
        "content": _OVERRIDES,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 108,
        "end_line": 137,
        "content": _MODEL,
    },
]
