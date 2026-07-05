"""STRONG upsampling baseline: bilinear-resize + conv upsampling (artifact-free)."""


def get_upsampling_config():
    return {"up": "bilinear"}
