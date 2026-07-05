"""STRONG drange baseline for stereo-disparity-range: range (64 px) covering
every difficulty setting.

The hardest setting's disparities span up to ~48 px; a 64-level range
comfortably covers every setting (easy/medium/hard) so no foreground pixel is
clipped -> low EPE.
Reference: vendor/stereo-matching/baselines/drange_full.py
"""

_FILE = "stereo-matching/solution/drange.py"

_CONTENT = '''\
def build_disp_range():
    # STRONG: disparity range (64 px) that comfortably covers every setting
    # (easy/medium/hard, up to ~48 px).
    return 64'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 24,
        "end_line": 27,
        "content": _CONTENT,
    },
]
