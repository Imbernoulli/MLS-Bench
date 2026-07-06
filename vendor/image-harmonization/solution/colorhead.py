"""Design surface: the OUTPUT PARAMETERIZATION (colour-transform head) of the harmonizer.

How the network produces the harmonized image is a real design axis. DoveNet (Cong et al.
CVPR 2020) predicts a full-resolution RGB RESIDUAL. But image harmonization is fundamentally
a COLOUR TRANSFORM: many strong harmonizers instead predict the PARAMETERS of a per-channel
colour transform (learnable colour curves / 3D-LUTs / affine colour heads). Because the
appearance gap here is a per-channel affine shift, an affine-parametric head matches the
true inverse transform and is more sample-efficient. Edit ONLY get_color_head() to return:

    'residual'       -> predict a full-resolution RGB residual added to the composite (DoveNet).
    'affine_global'  -> predict ONE per-channel (gain, bias) applied to the whole image (a
                        global colour transform).
    'affine_spatial' -> predict a per-pixel per-channel (gain, bias) map -> a spatially-varying
                        colour transform (the colour-transform head; matches the composite's
                        per-channel affine appearance shift most directly).

Everything else is FIXED. A malformed / crashing return falls back to 'affine_spatial'.
"""


def get_color_head():
    # Default: full-resolution RGB residual head (has to regress every pixel from scratch).
    return "residual"
