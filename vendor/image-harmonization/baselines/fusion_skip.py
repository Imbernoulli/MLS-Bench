"""image-harmonization fusion baseline: skip.

U-Net skip connections ON: sharp, accurate recolour (STRONG).
"""


def get_fusion_config():
    return {'skips': True}
