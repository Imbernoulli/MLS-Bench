"""Design surface: the conv NONLINEARITY (activation) in the harmonizer.

The appearance correction the harmonizer must apply is NONLINEAR (a per-channel affine
followed by a clamp to [0,1], plus a tint), so the network needs a genuine nonlinearity to
represent it. Edit ONLY get_activation() to return one of:

    'identity' -> NO nonlinearity: the conv stack collapses toward a single linear map, which
                  cannot represent the clamped / tinted correction -> under-fits (weak).
    'relu'     -> ReLU (the standard U-Net / DoveNet nonlinearity).
    'gelu'     -> GELU (a smooth alternative).

Everything else is FIXED. A malformed / crashing return falls back to 'relu'.
"""


def get_activation():
    # Default: identity (no nonlinearity) -> the net collapses to a linear map (weak).
    return "identity"
