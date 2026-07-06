"""image-harmonization normalization baseline: instance.

global InstanceNorm: ERASES the FG/BG statistic gap (WEAK).
"""


def get_normalization():
    return 'instance'
