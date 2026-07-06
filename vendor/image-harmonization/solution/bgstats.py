"""Design surface: FOREGROUND-BACKGROUND STATISTICS MATCHING input pre-conditioning.

Classic colour transfer (Reinhard et al. 2001) and RainNet's stats transfer both say the
same thing: to make the foreground match the scene you need the BACKGROUND colour statistics
as an explicit target. This surface appends the per-channel BACKGROUND MEAN (computed from
the untouched background region, broadcast over the frame) as extra input channels, so the
harmonizer is HANDED the target colour statistics it must align the foreground to -- instead
of having to infer them implicitly. Edit ONLY get_bgstats_config() to return a dict
{'enabled': bool}:

    {'enabled': False} -> the net only sees the composite (and mask); it must infer the target
                          background statistics implicitly.
    {'enabled': True}  -> append the broadcast background per-channel mean as extra input
                          channels -> an explicit statistics-matching prior, higher PSNR.

Everything else is FIXED. A malformed / crashing return falls back to {'enabled': True}.
"""


def get_bgstats_config():
    # Default: no background-statistics conditioning (weak).
    return {"enabled": False}
