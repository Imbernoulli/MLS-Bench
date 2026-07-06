"""Design surface: whether to apply a fixed BACKGROUND-REFERENCED INPUT NORMALIZATION.

A tempting idea is to WHITEN the whole composite by the BACKGROUND per-channel mean/std
before the network, so the foreground/background appearance gap is expressed relative to
the (already-correct) background, and un-whiten the output before scoring. In practice this
FIXED whitening is a poor input transform for this task: it rescales the image by the
background statistics (which vary wildly per image), destroying the absolute colour levels
the harmonizer needs and injecting instability at the un-whitening step -> a much WORSE
reconstruction than simply feeding the raw composite. Edit ONLY get_input_norm() to return:

    'bg_whiten' -> whiten the whole image by the BACKGROUND per-channel mean/std, then
                   un-whiten the output. The naive background-referencing transform: it
                   corrupts the input scale and the reconstruction collapses (weak).
    'none'      -> feed the RAW composite (the net handles the colour levels directly). The
                   robust choice -> much higher foreground PSNR.

Everything else is FIXED. A malformed / crashing return falls back to 'none'.
"""


def get_input_norm():
    # Default: naive background-whitening pre-normalization (corrupts the input, weak).
    return "bg_whiten"
