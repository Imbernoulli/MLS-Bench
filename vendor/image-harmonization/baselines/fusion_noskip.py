"""image-harmonization fusion baseline: noskip.

NO skip connections: the decoder loses foreground detail (WEAK).
"""


def get_fusion_config():
    return {'skips': False}
