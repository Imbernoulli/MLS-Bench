"""WEAK dilation baseline: plain 3x3 bottleneck convs (dilation 1), small receptive field."""


def get_dilation_config():
    return {"dilations": [1, 1]}
