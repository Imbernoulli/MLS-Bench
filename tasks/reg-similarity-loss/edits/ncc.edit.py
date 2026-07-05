"""STRONG baseline for reg-similarity-loss: local NCC image similarity.

Drive the VoxelMorph displacement field with a local normalized cross-correlation
(the VoxelMorph paper's recommended similarity). Local NCC is invariant to local
intensity/contrast shifts and gives a robust gradient for larger deformations, so
it matches or beats MSE and its advantage grows with the warp magnitude.
Reference: vendor/deformable-registration/baselines/similarity_ncc.py
"""

_FILE = "deformable-registration/solution/similarity.py"

_CONTENT = '''\
def build_similarity():
    # STRONG: local normalized cross-correlation similarity (VoxelMorph default).
    return "ncc"'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 20,
        "end_line": 22,
        "content": _CONTENT,
    },
]
