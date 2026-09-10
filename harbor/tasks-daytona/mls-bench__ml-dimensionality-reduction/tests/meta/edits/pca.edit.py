"""PCA baseline for ml-dimensionality-reduction.

Linear dimensionality reduction via Singular Value Decomposition.
Reference: sklearn.decomposition.PCA (standard baseline).
"""

_FILE = "scikit-learn/bench/custom_dimred.py"

_PCA = """\
class CustomDimReduction:
    \"\"\"PCA dimensionality reduction (linear baseline).\"\"\"

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        return pca.fit_transform(X)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 15,
        "end_line": 59,
        "content": _PCA,
    },
]
