"""t-SNE baseline for ml-dimensionality-reduction.

t-distributed Stochastic Neighbor Embedding.
Reference: sklearn.manifold.TSNE (van der Maaten & Hinton, 2008).
"""

_FILE = "scikit-learn/bench/custom_dimred.py"

_TSNE = """\
class CustomDimReduction:
    \"\"\"t-SNE dimensionality reduction.\"\"\"

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        from sklearn.manifold import TSNE
        tsne = TSNE(
            n_components=self.n_components,
            perplexity=30.0,
            learning_rate="auto",
            init="pca",
            random_state=self.random_state,
            n_iter=1000,
        )
        return tsne.fit_transform(X)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 15,
        "end_line": 59,
        "content": _TSNE,
    },
]
