"""image-harmonization bgstats baseline: on.

append broadcast BACKGROUND mean as input channels: explicit stats-match prior (STRONG).
"""


def get_bgstats_config():
    return {'enabled': True}
