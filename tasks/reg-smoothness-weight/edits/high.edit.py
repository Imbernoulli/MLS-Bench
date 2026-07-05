"""WEAK baseline for reg-smoothness-weight: OVER-STRONG smoothness (lambda=5.0).

An over-large smoothness penalty dominates the objective and collapses the
displacement field toward identity (near-zero, perfectly smooth but useless): the
folding vanishes but the field can no longer follow the deformation, so the
warped-moving vs fixed PSNR collapses at the medium/large settings. This is the
opposite failure mode to lambda=0.
Reference: vendor/deformable-registration/baselines/smoothness_high.py
"""

_FILE = "deformable-registration/solution/smoothness.py"

_CONTENT = '''\
def build_smoothness_weight():
    # WEAK: over-strong smoothness (field collapses toward identity).
    return 5.0'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 22,
        "end_line": 24,
        "content": _CONTENT,
    },
]
