"""WEAK baseline for reg-similarity-loss: MSE image similarity.

Drive the VoxelMorph displacement field with a plain mean-squared-error between
warped-moving and fixed. MSE assumes the two images differ only by geometry; it
is sensitive to any local intensity/contrast variation and gives a weaker
gradient for large deformations, so it is beaten by local NCC as the warp grows.
Reference: vendor/deformable-registration/baselines/similarity_mse.py
"""

_FILE = "deformable-registration/solution/similarity.py"

_CONTENT = '''\
def build_similarity():
    # WEAK: mean-squared-error similarity.
    return "mse"'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 20,
        "end_line": 22,
        "content": _CONTENT,
    },
]
