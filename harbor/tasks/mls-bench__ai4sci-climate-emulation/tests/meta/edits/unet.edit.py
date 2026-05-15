"""1D U-Net baseline for ai4sci-climate-emulation.

Adapted from the ClimsimUnet (Unet_v4) used in the stable ML parameterization
work for ClimSim-style emulation (arXiv:2407.00124).

Key design: reshapes flat input into profile channels over 60 vertical levels,
runs a 1D encoder-decoder U-Net with self-attention at the bottleneck, then
reshapes back to flat output. Skip connections between encoder and decoder.

Reference: Hu, Subramaniam, Kuang et al., "Stable Machine-Learning
Parameterization of Subgrid Processes with Real Geography and Full-physics
Emulation" (arXiv:2407.00124).
"""

_FILE = "ClimSim/custom_emulator.py"

_CONTENT = """\
class ResBlock1d(nn.Module):
    \"\"\"1D residual block: GroupNorm + Conv1d + SiLU + Conv1d + skip.\"\"\"
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, channels // 4), channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, channels // 4), channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = self.drop(F.silu(self.norm2(h)))
        h = self.conv2(h)
        return (x + h) * (0.5 ** 0.5)


class AttnBlock1d(nn.Module):
    \"\"\"Self-attention over the sequence (level) dimension.\"\"\"
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels // 4), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.num_heads = num_heads
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, L = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, L)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # Scaled dot-product attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhcl,bhcm->bhlm', q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhlm,bhcm->bhcl', attn, v)
        out = out.reshape(B, C, L)
        return (x + self.proj(out)) * (0.5 ** 0.5)


class Custom(nn.Module):
    \"\"\"1D U-Net for climate physics emulation (adapted from ClimsimUnet v4).

    Architecture:
    - Reshape flat [B, 556] -> [B, num_profile_vars + num_scalar_vars, 60]
      (profile vars naturally span 60 levels; scalars broadcast to all levels)
    - Pad to 64 (power of 2) for clean downsampling
    - Encoder: 3 resolution levels with residual blocks + downsampling
    - Bottleneck: residual block + self-attention
    - Decoder: 3 levels with skip connections + upsampling
    - Output projection back to flat [B, 368]
    \"\"\"
    N_LEVELS = 60
    N_PROFILE_IN = 9   # 9 multi-level input vars
    N_SCALAR_IN = 16   # 16 single-level input vars
    N_PROFILE_OUT = 6  # 6 multi-level output vars
    N_SCALAR_OUT = 8   # 8 single-level output vars

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        in_ch = self.N_PROFILE_IN + self.N_SCALAR_IN  # 25 channels
        base_ch = 128

        # Encoder
        self.enc_in = nn.Conv1d(in_ch, base_ch, 3, padding=1)
        self.enc1 = nn.ModuleList([ResBlock1d(base_ch) for _ in range(3)])
        self.down1 = nn.Conv1d(base_ch, base_ch * 2, 2, stride=2)  # 64->32
        self.enc2 = nn.ModuleList([ResBlock1d(base_ch * 2) for _ in range(3)])
        self.down2 = nn.Conv1d(base_ch * 2, base_ch * 2, 2, stride=2)  # 32->16

        # Bottleneck with attention
        self.mid1 = ResBlock1d(base_ch * 2)
        self.mid_attn = AttnBlock1d(base_ch * 2, num_heads=4)
        self.mid2 = ResBlock1d(base_ch * 2)

        # Decoder
        self.up2 = nn.ConvTranspose1d(base_ch * 2, base_ch * 2, 2, stride=2)  # 16->32
        self.dec2 = nn.ModuleList([ResBlock1d(base_ch * 4)] +
                                  [ResBlock1d(base_ch * 4) for _ in range(2)])
        self.dec2_proj = nn.Conv1d(base_ch * 4, base_ch * 2, 1)
        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch, 2, stride=2)  # 32->64
        self.dec1 = nn.ModuleList([ResBlock1d(base_ch * 2)] +
                                  [ResBlock1d(base_ch * 2) for _ in range(2)])
        self.dec1_proj = nn.Conv1d(base_ch * 2, base_ch, 1)

        # Output
        self.out_norm = nn.GroupNorm(min(32, base_ch // 4), base_ch)
        self.out_conv = nn.Conv1d(base_ch, self.N_PROFILE_OUT + self.N_SCALAR_OUT, 3, padding=1)

    def forward(self, x):
        B = x.shape[0]

        # Reshape: split profile (9 vars x 60 levels) and scalar (16 vars)
        x_profile = x[:, :self.N_PROFILE_IN * self.N_LEVELS]
        x_scalar = x[:, self.N_PROFILE_IN * self.N_LEVELS:]

        x_profile = x_profile.reshape(B, self.N_PROFILE_IN, self.N_LEVELS)  # [B, 9, 60]
        x_scalar = x_scalar.unsqueeze(2).expand(-1, -1, self.N_LEVELS)      # [B, 16, 60]
        h = torch.cat([x_profile, x_scalar], dim=1)  # [B, 25, 60]

        # Pad 60 -> 64 for clean 2x downsampling
        h = F.pad(h, (0, 4))  # [B, 25, 64]

        # Encoder
        h = self.enc_in(h)
        for block in self.enc1:
            h = block(h)
        skip1 = h  # [B, 128, 64]
        h = self.down1(h)  # [B, 256, 32]
        for block in self.enc2:
            h = block(h)
        skip2 = h  # [B, 256, 32]
        h = self.down2(h)  # [B, 256, 16]

        # Bottleneck
        h = self.mid1(h)
        h = self.mid_attn(h)
        h = self.mid2(h)

        # Decoder
        h = self.up2(h)  # [B, 256, 32]
        h = torch.cat([h, skip2], dim=1)  # [B, 512, 32]
        for block in self.dec2:
            h = block(h)
        h = self.dec2_proj(h)  # [B, 256, 32]
        h = self.up1(h)  # [B, 128, 64]
        h = torch.cat([h, skip1], dim=1)  # [B, 256, 64]
        for block in self.dec1:
            h = block(h)
        h = self.dec1_proj(h)  # [B, 128, 64]

        # Output
        h = self.out_conv(F.silu(self.out_norm(h)))  # [B, 14, 64]

        # Remove padding and reshape
        h = h[:, :, :self.N_LEVELS]  # [B, 14, 60]

        y_profile = h[:, :self.N_PROFILE_OUT, :].reshape(B, self.N_PROFILE_OUT * self.N_LEVELS)
        y_scalar = h[:, self.N_PROFILE_OUT:, :].mean(dim=2)  # avg over levels
        y_scalar = F.relu(y_scalar)  # non-negative scalar outputs

        return torch.cat([y_profile, y_scalar], dim=1)
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
