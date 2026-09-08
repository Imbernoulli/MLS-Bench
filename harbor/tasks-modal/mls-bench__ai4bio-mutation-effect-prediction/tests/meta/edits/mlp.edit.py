"""MLP baseline for mutation effect prediction.

Simple single-hidden-layer MLP head over mean-pooled ESM-2 features:

    Linear(embed_dim, hidden) -> Dropout -> ReLU -> Linear(hidden, 1)

This matches the simple nonlinear "MLP head over PLM embeddings" baseline
family commonly used in protein supervised benchmarks (TAPE, ProteinNPT,
ESM downstream probing). It is intentionally a single hidden layer — not a
residual deep MLP — to serve as a clean nonlinear comparison point against
the linear ridge baseline.

Reference: TAPE and ProteinGym/ProteinNPT motivate supervised probing of PLM
embeddings; this file provides a compact nonlinear probing baseline.
"""

_FILE = "ProteinGym/custom_mutation_pred.py"

_MODEL = """\

class MutationPredictor(nn.Module):
    \"\"\"Single-hidden-layer MLP over delta_embedding (mutant - WT).

    Architecture: Linear(embed_dim, hidden) -> Dropout -> ReLU -> Linear(hidden, 1)
    Uses delta_embedding so the network sees the mutation-induced shift
    in PLM representation space directly.
    \"\"\"

    def __init__(self, embed_dim: int = EMBED_DIM, hidden_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, embedding, delta_embedding):
        x = self.fc1(delta_embedding)
        x = self.dropout(x)
        x = F.relu(x)
        return self.fc2(x).squeeze(-1)

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 108,
        "end_line": 137,
        "content": _MODEL,
    },
]
