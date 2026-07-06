"""Design surface: the NORMALIZATION layer inside the harmonizer.

The defining move of RainNet (Ling et al., "Region-aware Adaptive Instance Normalization
for Image Harmonization", CVPR 2021) is a REGION-AWARE normalization (RAIN) that transfers
the BACKGROUND feature statistics onto the FOREGROUND features -- explicitly aligning the
pasted region's style to the scene it was dropped into. A naive global normalization does
the OPPOSITE of what harmonization needs. Edit ONLY get_normalization() to return one of:

    'none'     -> no normalization (plain conv blocks).
    'batch'    -> BatchNorm2d.
    'instance' -> InstanceNorm2d: standardizes each feature map GLOBALLY, which ERASES the
                  very foreground-vs-background statistic gap the harmonizer must model ->
                  the worst choice for region-aware harmonization.
    'rain'     -> the RainNet region-aware AdaIN: re-normalizes the FOREGROUND features to
                  the BACKGROUND feature statistics (background left untouched) -> the SOTA
                  region-aware design.

Everything else is FIXED. A malformed / crashing return falls back to 'rain'.
"""


def get_normalization():
    # Default: global InstanceNorm -> erases the FG/BG statistic gap (weak for harmonization).
    return "instance"
