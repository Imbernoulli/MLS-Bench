"""1D U-Net baseline for ai4sci-climate-emulation.

Faithful to the ClimsimUnet of Hu, Subramaniam, Kuang et al.
(arXiv:2407.00124, App. B.2/B.4/C):
  - input reshaped to vertical length 60 x channels (profile vars = channels;
    scalar features broadcast as constant-valued channels); top-padded to
    length 64 to allow clean downsampling
  - depth 4, channel schedule N_latent = [128, 256, 256, 256]
  - ResBlock = GroupNorm + SiLU + Conv1d(k3) (x2) with a residual connection
  - U-Net skip connections between encoder and decoder
  - a multi-head self-attention block at the bottleneck (the shipped ClimsimUnet
    in online_testing/baseline_models/Unet_v4 has a bottleneck attention block)
  - Huber loss with delta = 1 (injected; the trainer's MSELoss is replaced)
  - ~13M parameters in the reference

I/O adapted to this task's 556-dim input / 368-dim output. Optimizer/LR/batch/
epochs are the task's fixed unified budget (AdamW + cosine), not the reference's
ReduceLROnPlateau.

Reference: Hu et al., "Stable Machine-Learning Parameterization of Subgrid
Processes with Real Geography and Full-physics Emulation" (arXiv:2407.00124).
"""

_FILE = "ClimSim/custom_emulator.py"

_CONTENT = """\
class _UNetResBlock(nn.Module):
    \"\"\"GroupNorm + SiLU + Conv1d, twice, with a residual connection (no attention).\"\"\"
    def __init__(self, channels):
        super().__init__()
        g = min(32, max(1, channels // 4))
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class _UNetAttn(nn.Module):
    \"\"\"Multi-head self-attention over the vertical-level axis, at the bottleneck
    (the shipped ClimsimUnet has a bottleneck attention block).\"\"\"
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, max(1, channels // 4)), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.num_heads = num_heads
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, L = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, self.num_heads, C // self.num_heads, L)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.einsum('bhcl,bhcm->bhlm', q, k) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhlm,bhcm->bhcl', attn, v).reshape(B, C, L)
        return x + self.proj(out)


class Custom(nn.Module):
    \"\"\"1D U-Net (ClimsimUnet): depth 4, channels [128,256,256,256], bottleneck attention.\"\"\"

    N_LEVELS = 60
    PAD_LEVELS = 64
    N_PROFILE_IN = 9
    N_SCALAR_IN = 16
    N_PROFILE_OUT = 6
    N_SCALAR_OUT = 8
    CH = [128, 256, 256, 256]   # 4 resolution levels: 64, 32, 16, 8
    ENC_BLOCKS = 4              # reference num_blocks per encoder level
    DEC_BLOCKS = 5             # reference num_blocks + 1 per decoder level

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        in_ch = self.N_PROFILE_IN + self.N_SCALAR_IN     # 25 channels

        c0, c1, c2, c3 = self.CH
        mk = lambda c, n: nn.ModuleList([_UNetResBlock(c) for _ in range(n)])

        self.enc_in = nn.Conv1d(in_ch, c0, 3, padding=1)
        self.enc0 = mk(c0, self.ENC_BLOCKS)
        self.down0 = nn.Conv1d(c0, c1, 2, stride=2)      # 64 -> 32
        self.enc1 = mk(c1, self.ENC_BLOCKS)
        self.down1 = nn.Conv1d(c1, c2, 2, stride=2)      # 32 -> 16
        self.enc2 = mk(c2, self.ENC_BLOCKS)
        self.down2 = nn.Conv1d(c2, c3, 2, stride=2)      # 16 -> 8
        self.mid = mk(c3, self.ENC_BLOCKS)               # bottleneck
        self.mid_attn = _UNetAttn(c3)                    # bottleneck self-attention

        self.up2 = nn.ConvTranspose1d(c3, c2, 2, stride=2)   # 8 -> 16
        self.dec2 = mk(c2, self.DEC_BLOCKS); self.dec2_proj = nn.Conv1d(c2 + c2, c2, 1)
        self.up1 = nn.ConvTranspose1d(c2, c1, 2, stride=2)   # 16 -> 32
        self.dec1 = mk(c1, self.DEC_BLOCKS); self.dec1_proj = nn.Conv1d(c1 + c1, c1, 1)
        self.up0 = nn.ConvTranspose1d(c1, c0, 2, stride=2)   # 32 -> 64
        self.dec0 = mk(c0, self.DEC_BLOCKS); self.dec0_proj = nn.Conv1d(c0 + c0, c0, 1)

        self.out_norm = nn.GroupNorm(min(32, max(1, c0 // 4)), c0)
        self.out_conv = nn.Conv1d(c0, self.N_PROFILE_OUT + self.N_SCALAR_OUT, 3, padding=1)

    def _run(self, blocks, h):
        for b in blocks:
            h = b(h)
        return h

    def forward(self, x):
        B = x.shape[0]
        ml = x[:, :self.N_PROFILE_IN * self.N_LEVELS].reshape(B, self.N_PROFILE_IN, self.N_LEVELS)
        sl = x[:, self.N_PROFILE_IN * self.N_LEVELS:].unsqueeze(2).expand(-1, -1, self.N_LEVELS)
        h = torch.cat([ml, sl], dim=1)                          # [B, 25, 60]
        h = F.pad(h, (0, self.PAD_LEVELS - self.N_LEVELS))      # -> 64

        h = self.enc_in(h)
        s0 = self._run(self.enc0, h)                            # [B, c0, 64]
        h = self._run(self.enc1, self.down0(s0)); s1 = h        # [B, c1, 32]
        h = self._run(self.enc2, self.down1(s1)); s2 = h        # [B, c2, 16]
        h = self._run(self.mid, self.down2(s2))                 # [B, c3, 8]
        h = self.mid_attn(h)                                    # bottleneck attention

        h = self.dec2_proj(torch.cat([self.up2(h), s2], dim=1))
        h = self._run(self.dec2, h)
        h = self.dec1_proj(torch.cat([self.up1(h), s1], dim=1))
        h = self._run(self.dec1, h)
        h = self.dec0_proj(torch.cat([self.up0(h), s0], dim=1))
        h = self._run(self.dec0, h)

        h = self.out_conv(F.silu(self.out_norm(h)))             # [B, 14, 64]
        h = h[:, :, :self.N_LEVELS]                             # [B, 14, 60]
        y_ml = h[:, :self.N_PROFILE_OUT, :].reshape(B, -1)      # [B, 360]
        # outputs are z-normalized (straddle zero) -> scalar head stays linear.
        y_sl = h[:, self.N_PROFILE_OUT:, :].mean(dim=2)         # [B, 8]
        return torch.cat([y_ml, y_sl], dim=1)


# Reference trains with Huber loss (delta=1); replace the trainer's MSELoss with
# a Huber loss. Use a proper subclass of the canonical MSELoss (not a lambda) so
# that, if all baselines are imported into one process (e.g. budget_check.py),
# a later baseline that subclasses nn.MSELoss still works.
class _UNetHuberLoss(torch.nn.modules.loss.MSELoss):
    def forward(self, pred, target):
        return F.huber_loss(pred, target, delta=1.0)

nn.MSELoss = _UNetHuberLoss
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
