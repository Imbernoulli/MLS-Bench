"""Reshape-CNN baseline for mutation effect prediction.

NOT a paper-faithful "1D CNN over residue sequence" baseline. The
precomputed embedding files only store mean-pooled ESM-2 features
([N, 1280]) — per-residue token representations are not available — so a
true Conv1D over the protein sequence dimension is not possible from
these inputs.

This baseline instead **reshapes the 2*1280 = 2560-dim mean-pooled feature
vector into a fake (channels=64, length=40) "image"** and applies 1D
convolutions over the embedding-channel axis. There is no physical
sequence/spatial structure along this axis; the convolutions just enforce
weight sharing across blocks of embedding dimensions.

Reasons to keep this baseline:
  - It probes whether structured weight-sharing over PLM features helps
    versus a flat MLP / linear head (an honest comparison of inductive
    biases on the same input).
  - It rounds out the "linear / nonlinear-MLP / convolutional" baseline
    triplet that mirrors the classical CNN/MLP/linear baseline family in
    deep-learning benchmarks.

To replace this with a true paper-CNN baseline, regenerate the embedding
files with per-residue ESM-2 token outputs ([N, L, 1280]) and adapt the
fixed dataset/collate to provide variable-length inputs; then this file
should be replaced with a Conv1D-over-residues model. That is left as a
future data-pipeline change.
"""

_FILE = "ProteinGym/custom_mutation_pred.py"

_MODEL = """\

class ConvBlock(nn.Module):
    \"\"\"1D convolution block with BatchNorm and residual connection.\"\"\"

    def __init__(self, channels, kernel_size, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.gelu(self.bn(self.conv(x)))
        x = self.dropout(x)
        return x + residual


class MutationPredictor(nn.Module):
    \"\"\"Reshape-CNN over mean-pooled ESM-2 features (NOT per-residue).

    Concatenates [embedding, delta_embedding] -> [B, 2*EMBED_DIM=2560],
    reshapes to (B, channels=64, length=40), applies a stack of 1D
    convolutions with residual connections over the embedding-channel
    axis, then global-average-pools and predicts.

    The reshape axis has NO real sequence structure — see the docstring
    in reshape_cnn.edit.py for why this is not a paper-faithful CNN.
    \"\"\"

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.channels = 64
        self.length = (embed_dim * 2) // self.channels  # 40

        self.input_proj = nn.Linear(embed_dim * 2, self.channels * self.length)

        self.conv_blocks = nn.Sequential(
            ConvBlock(self.channels, kernel_size=3, dropout=0.1),
            ConvBlock(self.channels, kernel_size=5, dropout=0.1),
            ConvBlock(self.channels, kernel_size=7, dropout=0.1),
        )

        self.head = nn.Sequential(
            nn.Linear(self.channels, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, embedding, delta_embedding):
        x = torch.cat([embedding, delta_embedding], dim=-1)  # [B, 2*EMBED_DIM]
        x = F.gelu(self.input_proj(x))                       # [B, C*L]
        x = x.view(x.size(0), self.channels, self.length)    # [B, C, L]
        x = self.conv_blocks(x)                              # [B, C, L]
        x = x.mean(dim=-1)                                   # [B, C]
        return self.head(x).squeeze(-1)                      # [B]

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
