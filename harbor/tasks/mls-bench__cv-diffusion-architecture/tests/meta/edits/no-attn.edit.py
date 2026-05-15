"""Baseline: No-attention architecture (pure convolutional UNet).

Uses only DownBlock2D / UpBlock2D with no self-attention layers at any
resolution. The mid-block still uses the default UNetMidBlock2D (which
includes one self-attention layer). This tests whether per-resolution
attention is necessary for good FID on CIFAR-10.
"""

_FILE = "diffusers-main/custom_train.py"

_NO_ATTN = '''
def build_model(device):
    """No-attention: pure convolutional UNet (no per-resolution attention)."""
    channels = (128, 256, 256, 256)
    if os.environ.get('BLOCK_OUT_CHANNELS'):
        channels = tuple(int(x) for x in os.environ['BLOCK_OUT_CHANNELS'].split(','))
    layers = int(os.environ.get('LAYERS_PER_BLOCK', 2))

    return UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        block_out_channels=channels,
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
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
        "content": _NO_ATTN,
    },
]
