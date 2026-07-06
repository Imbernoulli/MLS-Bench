"""Design surface: a channel ATTENTION gate on the harmonizer bottleneck.

The appearance correction is largely a per-CHANNEL recalibration (the composite shifts each
colour channel by a different gain/bias). A squeeze-excite channel-attention gate (Hu et al.,
"Squeeze-and-Excitation Networks", CVPR 2018) globally pools the bottleneck features and
learns a per-channel multiplicative gate, letting the net dynamically emphasise the channels
that carry the colour-correction signal. Edit ONLY get_attention_config() to return a dict
{'enabled': bool}:

    {'enabled': False} -> plain conv bottleneck (no channel recalibration).
    {'enabled': True}  -> squeeze-excite channel attention on the bottleneck -> sharper
                          per-channel correction, higher foreground PSNR.

Everything else is FIXED. A malformed / crashing return falls back to {'enabled': True}.
"""


def get_attention_config():
    # Default: no attention gate (weak).
    return {"enabled": False}
