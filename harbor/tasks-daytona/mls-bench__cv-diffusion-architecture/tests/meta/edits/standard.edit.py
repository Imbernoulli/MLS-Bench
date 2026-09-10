"""Baseline: Standard DDPM architecture (attention only at 16x16 resolution).

This is the original architecture from Ho et al., 2020 (google/ddpm-cifar10-32).
Self-attention is placed only at the second resolution level (16x16).
"""

_FILE = "diffusers-main/custom_train.py"

_STANDARD = '''
def build_model(device):
    """Standard DDPM architecture: attention at 16x16 only."""
    channels = (128, 256, 256, 256)
    if os.environ.get('BLOCK_OUT_CHANNELS'):
        channels = tuple(int(x) for x in os.environ['BLOCK_OUT_CHANNELS'].split(','))
    layers = int(os.environ.get('LAYERS_PER_BLOCK', 2))

    return UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        block_out_channels=channels,
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
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
        "content": _STANDARD,
    },
]
