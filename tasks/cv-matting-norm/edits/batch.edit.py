"""SOTA baseline for cv-matting-norm: BatchNorm (cross-image statistics).

BatchNorm after each conv. On this synthetic composite data the fg/bg statistics
recur across the fixed set, so BatchNorm's cross-image statistics are informative and
it converges to a much lower SAD in the short fine-tune -> clear headroom over no-norm.
Reference: vendor/image-matting/baselines/norm_batch.py
"""

_FILE = "image-matting/solution/norm.py"

_CONTENT = '''def make_norm(num_ch):
    return nn.BatchNorm2d(num_ch)'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 30, "end_line": 34, "content": _CONTENT},
]
