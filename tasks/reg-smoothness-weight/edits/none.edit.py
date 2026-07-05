"""WEAK baseline for reg-smoothness-weight: NO smoothness regulariser (lambda=0).

With no penalty on the displacement-field gradient the U-Net is free to produce a
jagged, locally-folding field: it can overfit the similarity but the deformation
becomes non-diffeomorphic (large fraction of pixels with non-positive Jacobian
determinant) and the PSNR at large deformations actually drops. A degenerate
field is a classic registration failure mode.
Reference: vendor/deformable-registration/baselines/smoothness_none.py
"""

_FILE = "deformable-registration/solution/smoothness.py"

_CONTENT = '''\
def build_smoothness_weight():
    # WEAK: no smoothness regularisation (jagged, folding field).
    return 0.0'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 22,
        "end_line": 24,
        "content": _CONTENT,
    },
]
