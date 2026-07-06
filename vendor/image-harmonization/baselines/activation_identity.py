"""image-harmonization activation baseline: identity.

NO nonlinearity: the net collapses to a linear map, under-fits (WEAK).
"""


def get_activation():
    return 'identity'
