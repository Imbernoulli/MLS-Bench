"""TriMap baseline for ml-dimensionality-reduction.

Large-scale Dimensionality Reduction Using Triplets.
Reference: Amid & Warmuth (2019). TriMap: Large-scale Dimensionality
Reduction Using Triplets. arXiv:1910.00204.

Strong triplet-based baseline for preserving both local and global structure;
this task ranks exact performance using the local leaderboard metrics.
"""

_FILE = "scikit-learn/bench/custom_dimred.py"

_TRIMAP = """\
class CustomDimReduction:
    \"\"\"TriMap dimensionality reduction.\"\"\"

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        import trimap
        reducer = trimap.TRIMAP(
            n_dims=self.n_components,
            n_inliers=10,
            n_outliers=5,
            n_random=5,
            n_iters=400,
        )
        return reducer.fit_transform(X)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 15,
        "end_line": 59,
        "content": _TRIMAP,
    },
]
