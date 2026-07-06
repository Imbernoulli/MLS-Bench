"""Design surface: the BOTTLENECK DILATION rate (receptive-field / context) of the harmonizer.

To recolour the pasted foreground to MATCH the scene, the network must SEE the background
context surrounding the foreground blob (the target colour statistics live outside the
foreground). A dilated convolution in the bottleneck widens the receptive field cheaply so
that context reaches the foreground pixels (dilated context is used throughout dense
prediction, e.g. Yu & Koltun 2016; harmonizers benefit similarly). Edit ONLY get_dilation()
to return an integer rate in [1, 8]:

    1     -> no dilation: the bottleneck sees only a small local window, so foreground pixels
             far from the boundary never see enough background context -> weaker correction.
    2..8  -> a dilated bottleneck: the surrounding background context reaches the foreground,
             improving the inferred target colour. (~4 is a good balance; absurdly large
             dilation gridding-artefacts at 64px give no further gain.)

Everything else is FIXED. A malformed / crashing return falls back to 4.
"""


def get_dilation():
    # Default: no dilation (rate 1) -> limited context (weak).
    return 1
