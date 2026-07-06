"""image-harmonization inputnorm baseline: bg_whiten.

whiten by BACKGROUND mean/std first: appearance gap relative to the background (STRONG).
"""


def get_input_norm():
    return 'bg_whiten'
