"""Encoder-Decoder baseline for ai4sci-climate-emulation.

Paper-faithful ClimSim Encoder-Decoder (Yu et al., NeurIPS 2023 D&B): a wide
6-fully-connected encoder compresses the 556-dim atmospheric state to a tiny
5-node latent bottleneck, then a symmetric 6-fully-connected decoder expands
back to the 368-dim tendency output. This is much wider per layer (paper
Table A: 768/512/384/256/128/64) than the previous overly-thin 32-latent
version, and the latent is just 5 nodes, matching the published baseline.

Reference: Yu et al., "ClimSim: A large multi-scale dataset for hybrid
physics-ML climate emulation" (NeurIPS 2023 Datasets & Benchmarks),
ED baseline.
"""

_FILE = "ClimSim/custom_emulator.py"

_CONTENT = """\
class _EDBlock(nn.Module):
    \"\"\"FC + LayerNorm + ELU + Dropout, one rung of the encoder/decoder ladder.\"\"\"
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.net(x)


class Custom(nn.Module):
    \"\"\"Wide Encoder-Decoder with 5-node latent bottleneck.

    Encoder: 6 FC blocks 556 -> 768 -> 512 -> 384 -> 256 -> 128 -> 5
    Latent:  5 nodes (paper-faithful)
    Decoder: 6 FC blocks 5 -> 128 -> 256 -> 384 -> 512 -> 768 -> 368
    \"\"\"

    LATENT_DIM = 5
    ENC_DIMS = [768, 512, 384, 256, 128]   # 6 FC layers (the 6th = projection to LATENT)
    DEC_DIMS = [128, 256, 384, 512, 768]   # mirrors encoder

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # ---- Encoder: 6 FC blocks ending at the 5-node latent ----
        enc_layers = []
        prev = input_dim
        for d in self.ENC_DIMS:
            enc_layers.append(_EDBlock(prev, d, dropout=0.1))
            prev = d
        # 6th FC: projection into the bottleneck (no nonlinearity → linear code)
        enc_layers.append(nn.Linear(prev, self.LATENT_DIM))
        self.encoder = nn.Sequential(*enc_layers)

        # ---- Decoder: 6 FC blocks expanding from the 5-node latent ----
        dec_layers = []
        prev = self.LATENT_DIM
        for d in self.DEC_DIMS:
            dec_layers.append(_EDBlock(prev, d, dropout=0.1))
            prev = d
        # 6th FC: projection to output (linear)
        dec_layers.append(nn.Linear(prev, output_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)              # [B, 5]
        y = self.decoder(z)              # [B, output_dim]
        return y
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
