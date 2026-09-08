"""PaCMAP baseline for ml-dimensionality-reduction.

Pairwise Controlled Manifold Approximation Projection.
Reference: Wang, Huang & Rudin (2021). Understanding How Dimension
Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP,
TriMap, and PaCMAP for Data Visualization. JMLR 22(201):1-73.

SOTA baseline: uses near-pair, mid-near-pair, and far-pair forces for
balanced local/global structure preservation.
"""

_FILE = "scikit-learn/bench/custom_dimred.py"

_PACMAP = """\
class CustomDimReduction:
    \"\"\"PaCMAP dimensionality reduction (SOTA).\"\"\"

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        import pacmap
        reducer = pacmap.PaCMAP(
            n_components=self.n_components,
            n_neighbors=10,
            MN_ratio=0.5,
            FP_ratio=2.0,
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
        "content": _PACMAP,
    },
]
