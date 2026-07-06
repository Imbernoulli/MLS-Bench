"""Design surface: encoder->decoder SKIP connections (feature fusion) in the U-Net.

The harmonizer must recolour the foreground WITHOUT destroying its fine structure/texture.
The encoder downsamples away high-frequency detail; U-Net SKIP connections re-inject that
detail from the encoder into the decoder so the recoloured output stays sharp and
spatially accurate (the core U-Net fusion, used by every real harmonizer). Edit ONLY
get_fusion_config() to return a dict {'skips': bool}:

    {'skips': False} -> NO skips: the bottleneck alone must reconstruct the image, losing the
                        high-frequency foreground detail -> a blurred, lower-PSNR recolour.
    {'skips': True}  -> skips ON: encoder detail is fused into the decoder -> sharp, accurate.

Everything else is FIXED. A malformed / crashing return falls back to {'skips': True}.
"""


def get_fusion_config():
    # Default: NO skip connections -> the decoder loses foreground detail (weak).
    return {"skips": False}
