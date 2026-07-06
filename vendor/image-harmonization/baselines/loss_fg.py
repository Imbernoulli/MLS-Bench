"""image-harmonization loss baseline: fg.

whole-image L1 + FOREGROUND emphasis (STRONG).
"""


def get_loss_config():
    return {'mode': 'fg'}
