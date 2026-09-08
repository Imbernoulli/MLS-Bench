"""Encoder-Decoder baseline for ai4sci-climate-emulation.

Faithful to the ClimSim reference ED, Yu et al. NeurIPS 2023 D&B,
`baseline_models/ED/training/ClimSIM_ED_1_3_train.py` (hyperparameters from
Behrens et al. 2022): a plain fully-connected encoder-decoder with
intermediate_dim=463, a 5-node latent bottleneck, ReLU activations, no
normalization or dropout, and an ELU output.

  encoder (7 Dense, ReLU): 463 -> 463 -> 231 -> 115 -> 57 -> 28 -> 5(latent)
  decoder (7 Dense, ReLU):  5 -> 28 -> 57 -> 115 -> 231 -> 463 -> 463 -> out(ELU)

(widths are intermediate_dim / {1,1,2,4,8,16}; rounded down as in the source.)
I/O adapted to this task's 556-dim input / 368-dim output; optimizer/LR/batch/
epochs are the task's fixed unified budget (AdamW + cosine), trained with MSE.

Reference: Yu et al., "ClimSim: A large multi-scale dataset for hybrid
physics-ML climate emulation" (NeurIPS 2023 Datasets & Benchmarks), ED baseline.
"""

_FILE = "ClimSim/custom_emulator.py"

_CONTENT = """\
class Custom(nn.Module):
    \"\"\"Plain fully-connected Encoder-Decoder with a 5-node latent (ClimSim ED).\"\"\"

    INTERMEDIATE = 463
    LATENT_DIM = 5
    # intermediate_dim / {1, 1, 2, 4, 8, 16} (floor), matching the reference taper.
    ENC_DIMS = [463, 463, 231, 115, 57, 28]
    DEC_DIMS = [28, 57, 115, 231, 463, 463]

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        enc = []
        prev = input_dim
        for d in self.ENC_DIMS:
            enc += [nn.Linear(prev, d), nn.ReLU()]
            prev = d
        enc += [nn.Linear(prev, self.LATENT_DIM), nn.ReLU()]  # ReLU latent (reference)
        self.encoder = nn.Sequential(*enc)

        dec = []
        prev = self.LATENT_DIM
        for d in self.DEC_DIMS:
            dec += [nn.Linear(prev, d), nn.ReLU()]
            prev = d
        dec += [nn.Linear(prev, output_dim), nn.ELU()]   # ELU output (reference)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))
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
