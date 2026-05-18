"""Mean Pooling baseline for ai4sci-weather-forecast-aggregation.

Simple average across variable tokens at each spatial location. This is the
simplest possible aggregation — no learnable parameters beyond what the
variable embeddings already provide. Serves as a lower bound.

Reference: standard mean pooling baseline
"""

_FILE = "ClimaX/custom_forecast.py"

_CONTENT = """\
class VariableAggregator(nn.Module):
    \"\"\"Mean pooling variable aggregation.

    Simply averages all V variable tokens at each spatial location.
    No additional learnable parameters.

    Args:
        embed_dim (int): Embedding dimension D.
        num_heads (int): Number of attention heads (unused).
        num_vars (int): Number of input variables V (unused).
    \"\"\"

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vars = num_vars

    def forward(self, x):
        \"\"\"
        Args:
            x: [B, V, L, D] — per-variable patch embeddings.
        Returns:
            [B, L, D] — aggregated representation.
        \"\"\"
        # Average across variable dimension
        out = x.mean(dim=1)  # B, L, D
        return out
"""

# Keep training-loop hyperparameters shared across all baselines; this file only
# changes the variable aggregation architecture.
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 310, "end_line": 351, "content": _CONTENT},
]
