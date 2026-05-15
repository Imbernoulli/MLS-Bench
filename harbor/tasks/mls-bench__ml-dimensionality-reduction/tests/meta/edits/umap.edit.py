"""UMAP baseline for ml-dimensionality-reduction.

Uniform Manifold Approximation and Projection.
Reference: McInnes, Healy & Melville (2018). UMAP: Uniform Manifold
Approximation and Projection for Dimension Reduction. arXiv:1802.03426.
"""

_FILE = "scikit-learn/bench/custom_dimred.py"

_UMAP = """\
class CustomDimReduction:
    \"\"\"UMAP dimensionality reduction.\"\"\"

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        import umap
        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=self.random_state,
        )
        return reducer.fit_transform(X)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 15,
        "end_line": 59,
        "content": _UMAP,
    },
]
