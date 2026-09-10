"""CNN baseline for ai4sci-climate-emulation.

Faithful to the ClimSim reference CNN (1D ResNet), Yu et al. NeurIPS 2023 D&B,
`baseline_models/CNN/training/hpo_train.py` (`CNNHyperModel.build`):
  - vertical levels (60) are the spatial axis; variables are channels
    (profile vars are channels; single-level scalars are broadcast over levels)
  - 12 residual blocks, 406 channels, kernel 3, padding "same", NO normalization
    (hp_norm=False), dropout 0.175
  - block = Conv1D -> ReLU -> Dropout -> Conv1D -> ReLU -> Dropout, with a 1x1
    Conv on the residual path, then add
  - pre-output ELU; output split into a linear head (vertically-resolved
    tendencies) and a ReLU head (non-negative scalar outputs)

I/O is adapted to this task's 556-dim input (9 profile x 60 + 16 scalar) and
368-dim output (6 profile x 60 + 8 scalar); the optimizer/LR/batch/epochs are
the task's fixed unified budget (AdamW + cosine), not the reference's cyclic LR.

Reference: Yu et al., "ClimSim: A large multi-scale dataset for hybrid
physics-ML climate emulation" (NeurIPS 2023 Datasets & Benchmarks).
"""

_FILE = "ClimSim/custom_emulator.py"

_CONTENT = """\
class _CNNResBlock(nn.Module):
    \"\"\"ClimSim CNN residual block: Conv-ReLU-Drop-Conv-ReLU-Drop + 1x1 skip.

    No normalization (reference hp_norm=False).\"\"\"
    def __init__(self, channels, kernel=3, dropout=0.175):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.skip = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.relu(self.conv1(x)))
        h = self.drop(F.relu(self.conv2(h)))
        return h + self.skip(x)


class Custom(nn.Module):
    \"\"\"1D ResNet CNN over vertical profiles (ClimSim reference: 12 blocks, 406 ch).\"\"\"

    N_LEVELS = 60
    N_PROFILE_IN = 9
    N_PROFILE_OUT = 6
    N_SCALAR_OUT = 8
    CHANNELS = 406
    N_BLOCKS = 12
    KERNEL = 3
    DROPOUT = 0.175

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_scalar_in = input_dim - self.N_PROFILE_IN * self.N_LEVELS  # 16
        in_ch = self.N_PROFILE_IN + self.n_scalar_in                      # 25 channels

        self.input_conv = nn.Conv1d(in_ch, self.CHANNELS, self.KERNEL, padding=self.KERNEL // 2)
        self.blocks = nn.ModuleList(
            [_CNNResBlock(self.CHANNELS, self.KERNEL, self.DROPOUT) for _ in range(self.N_BLOCKS)]
        )
        # Pre-output ELU projection, then split heads.
        self.out_conv = nn.Conv1d(self.CHANNELS, self.CHANNELS, 1)
        self.ml_head = nn.Conv1d(self.CHANNELS, self.N_PROFILE_OUT, 1)       # linear
        self.sl_head = nn.Sequential(                                        # non-negative scalars
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(self.CHANNELS, self.N_SCALAR_OUT),
        )

    def forward(self, x):
        B = x.shape[0]
        ml = x[:, :self.N_PROFILE_IN * self.N_LEVELS].reshape(B, self.N_PROFILE_IN, self.N_LEVELS)
        sl = x[:, self.N_PROFILE_IN * self.N_LEVELS:].unsqueeze(2).expand(-1, -1, self.N_LEVELS)
        h = torch.cat([ml, sl], dim=1)                  # [B, 25, 60]
        h = F.relu(self.input_conv(h))
        for blk in self.blocks:
            h = blk(h)
        h = F.elu(self.out_conv(h))                     # pre-output ELU
        ml_out = self.ml_head(h).reshape(B, -1)         # [B, 360], linear
        # NOTE: outputs are z-normalized (straddle zero) -> scalar head stays linear.
        sl_out = self.sl_head(h)                         # [B, 8]
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
