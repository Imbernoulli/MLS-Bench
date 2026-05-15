"""Learned Weighted Sum baseline for ai4sci-weather-forecast-aggregation.

Learnable per-variable scalar weights, normalized via softmax, then used
to compute a weighted sum across variable tokens. More expressive than
simple mean pooling but much simpler than cross-attention.

Reference: common attention-free aggregation in multi-modal / multi-source
models (e.g., weighted feature fusion in FPN, multi-view aggregation).
"""

_FILE = "ClimaX/custom_forecast.py"

_CONTENT = """\
class VariableAggregator(nn.Module):
    \"\"\"Learned weighted sum variable aggregation.

    Learns a scalar weight per variable, applies softmax normalization,
    then computes a weighted sum across variable tokens.

    Args:
        embed_dim (int): Embedding dimension D.
        num_heads (int): Number of attention heads (unused).
        num_vars (int): Number of input variables V.
    \"\"\"

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vars = num_vars
        # Learnable weight per variable
        self.var_weights = nn.Parameter(torch.zeros(num_vars), requires_grad=True)

    def forward(self, x):
        \"\"\"
        Args:
            x: [B, V, L, D] — per-variable patch embeddings.
        Returns:
            [B, L, D] — aggregated representation.
        \"\"\"
        # Softmax-normalized variable weights
        w = F.softmax(self.var_weights, dim=0)  # V
        w = w.view(1, -1, 1, 1)                # 1, V, 1, 1
        # Weighted sum across variables
        out = (x * w).sum(dim=1)  # B, L, D
        return out
"""

# Keep training-loop hyperparameters shared across all baselines; this file only
# changes the variable aggregation architecture.
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 310, "end_line": 351, "content": _CONTENT},
]
