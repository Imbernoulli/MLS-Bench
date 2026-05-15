"""CNN baseline for ai4sci-climate-emulation.

1D Convolutional network with residual blocks that operates on vertical
atmospheric profiles. Multi-level variables are treated as spatial
sequences over 60 vertical levels; single-level scalars are broadcast and
concatenated.

Reference: Yu et al., "ClimSim: A large multi-scale dataset for hybrid
physics-ML climate emulation" (NeurIPS 2023 Datasets & Benchmarks)
Architecture inspired by ClimSim CNN baseline with ResNet-style blocks.
"""

_FILE = "ClimSim/custom_emulator.py"

_CONTENT = """\
class Custom(nn.Module):
    \"\"\"1D CNN with residual blocks for climate emulation.

    Reshapes input into (n_vars, n_levels) for convolution over vertical profiles,
    then projects back to output space.
    \"\"\"

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input structure: 9 multi-level vars x 60 levels = 540, then 16-17 scalars
        self.n_ml_in = 9
        self.n_levels = 60
        self.n_sl_in = input_dim - self.n_ml_in * self.n_levels

        # Project scalar inputs to per-level features
        self.scalar_proj = nn.Linear(self.n_sl_in, self.n_levels)

        # Conv channels: n_ml_in + 1 (from scalar projection)
        in_channels = self.n_ml_in + 1
        hidden_channels = 128
        n_blocks = 8

        # Initial projection
        self.input_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)

        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.BatchNorm1d(hidden_channels),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            ))

        # Output: multi-level tendencies
        self.n_ml_out = 6
        self.ml_head = nn.Conv1d(hidden_channels, self.n_ml_out, kernel_size=1)

        # Output: single-level scalars from pooled features
        self.sl_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

    def forward(self, x):
        B = x.shape[0]
        # Split multi-level and single-level inputs
        ml_in = x[:, :self.n_ml_in * self.n_levels].view(B, self.n_ml_in, self.n_levels)
        sl_in = x[:, self.n_ml_in * self.n_levels:]
        sl_expanded = self.scalar_proj(sl_in).unsqueeze(1)  # (B, 1, 60)
        h = torch.cat([ml_in, sl_expanded], dim=1)  # (B, n_ml_in+1, 60)

        h = F.relu(self.input_conv(h))
        for block in self.blocks:
            h = h + block(h)

        ml_out = self.ml_head(h).reshape(B, -1)  # (B, 360)
        sl_out = self.sl_head(h)  # (B, 8)
        return torch.cat([ml_out, sl_out], dim=-1)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 86,
        "end_line": 118,
        "content": _CONTENT,
    },
]
