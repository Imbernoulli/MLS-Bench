"""WEAK drange baseline for stereo-disparity-range: far-too-small range (8 px).

Cannot represent the larger (up to ~48 px on the hard setting) disparities;
foreground pixels are clipped at the top of the range -> high EPE.
Reference: vendor/stereo-matching/baselines/drange_small.py
"""

_FILE = "stereo-matching/solution/drange.py"

_CONTENT = '''\
def build_disp_range():
    # WEAK baseline: far-too-small disparity range (8 px) — cannot represent
    # the larger (up to ~48 px, hard setting) disparities.
    return 8'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 24,
        "end_line": 27,
        "content": _CONTENT,
    },
]
