"""image-harmonization maskcond baseline: concat.

mask CONCATENATED as a 4th channel (DoveNet): knows the foreground (STRONG).
"""


def get_mask_conditioning():
    return 'concat'
