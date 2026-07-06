"""Design surface: the DECODER UPSAMPLING operator in the harmonizer.

The decoder doubles the feature resolution twice on the way back to full size. HOW it
upsamples affects reconstruction sharpness. Edit ONLY get_upsampling() to return one of:

    'nearest'   -> nearest-neighbour upsample + conv: cheap but BLOCKY, producing checker /
                   staircase artefacts in the recolour -> lower foreground PSNR (weak).
    'transpose' -> a learned transposed convolution (DoveNet / standard U-Net decoder).
    'bilinear'  -> bilinear upsample + conv: smooth, artefact-free.

Everything else is FIXED. A malformed / crashing return falls back to 'transpose'.
"""


def get_upsampling():
    # Default: nearest-neighbour upsampling -> blocky reconstruction (weak).
    return "nearest"
