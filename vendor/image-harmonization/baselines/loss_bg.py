"""image-harmonization loss baseline: bg.

supervise the already-correct BACKGROUND only: no foreground signal (DEGENERATE).
"""


def get_loss_config():
    return {'mode': 'bg'}
