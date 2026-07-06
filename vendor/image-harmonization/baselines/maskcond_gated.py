"""image-harmonization maskcond baseline: gated.

concat + a mask-gated output blend: background provably preserved (SOTA).
"""


def get_mask_conditioning():
    return 'gated'
