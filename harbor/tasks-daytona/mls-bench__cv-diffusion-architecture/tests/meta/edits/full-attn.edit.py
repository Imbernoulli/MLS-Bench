"""Baseline: Full-attention architecture (self-attention at every resolution).

Places AttnDownBlock2D / AttnUpBlock2D at all four resolution levels
(32x32, 16x16, 8x8, 4x4). More expressive but significantly more
compute and memory per step.
"""

_FILE = "diffusers-main/custom_train.py"

_FULL_ATTN = '''
def build_model(device):
    """Full-attention: self-attention at every resolution."""
    channels = (128, 256, 256, 256)
    if os.environ.get('BLOCK_OUT_CHANNELS'):
        channels = tuple(int(x) for x in os.environ['BLOCK_OUT_CHANNELS'].split(','))
    layers = int(os.environ.get('LAYERS_PER_BLOCK', 2))

    return UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        block_out_channels=channels,
        down_block_types=("AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D"),
        layers_per_block=layers,
        norm_num_groups=32,
        norm_eps=1e-6,
        act_fn="silu",
        time_embedding_type="positional",
        flip_sin_to_cos=False,
        freq_shift=1,
        downsample_padding=0,
    ).to(device)
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 31,
        "end_line": 58,
        "content": _FULL_ATTN,
    },
]
