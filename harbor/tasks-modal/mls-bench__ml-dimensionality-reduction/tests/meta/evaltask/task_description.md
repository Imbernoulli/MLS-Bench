# Dimensionality Reduction: Nonlinear Embedding Method Design

## Research Question
Design a novel nonlinear dimensionality-reduction method that embeds high-dimensional data into 2D while preserving both local neighborhoods and global structure better than existing methods. The contribution is the *embedding algorithm*: graph construction, neighbor forces, optimization schedule, hybrid linear/nonlinear stages — without using class labels at fit time.

## Background
PCA gives a fast linear baseline but misses nonlinear manifold structure. Modern neighbor-embedding methods trade off local/global preservation differently.

Reference baselines:
- **PCA** — Pearson, 1901 / Hotelling, 1933. Project onto top-2 principal components of the (centered) data.
- **t-SNE** — van der Maaten & Hinton, JMLR 2008 ([paper](https://www.jmlr.org/papers/v9/vandermaaten08a.html)). Match Gaussian P-distribution in input space to Student-t Q-distribution in embedding space, minimizing KL. Strong local structure, weaker global structure.
- **UMAP** — McInnes, Healy, Melville, 2018 ([arXiv:1802.03426](https://arxiv.org/abs/1802.03426)). Fuzzy simplicial set built from k-NN graph; minimizes cross-entropy between high- and low-dimensional fuzzy graphs. Default `n_neighbors=15`, `min_dist=0.1`.
- **TriMap** — Amid & Warmuth, 2019 ([arXiv:1910.00204](https://arxiv.org/abs/1910.00204)). Triplet-based loss `(i,j,k)` enforcing "i closer to j than to k", weighted by triplet importance; better global structure than t-SNE/UMAP.
- **PaCMAP** — Wang, Huang, Rudin, Shaposhnik, JMLR 2021 ([arXiv:2012.04456](https://arxiv.org/abs/2012.04456)). Three pair types — Neighbor, Mid-Near, Further — with a three-phase optimization schedule that re-weights them over iterations.

## Implementation Contract
Modify `CustomDimReduction` in `scikit-learn/bench/custom_dimred.py`:

```python
class CustomDimReduction:
    def __init__(self, n_components: int = 2, random_state: int | None = None):
        ...

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        # X: (n_samples, n_features), return shape (n_samples, n_components)
        ...
```

Constraints:
- Inputs: `n_samples ≤ 5000`; `n_features` ranges from 50 to 784.
- Must respect `random_state` for reproducibility.
- Must finish within ~5 minutes per dataset on CPU.
- Available libraries: `numpy`, `scipy`, `scikit-learn`.

## Fixed Pipeline & Evaluation
Datasets:
- **MNIST** — handwritten digit images (784 features).
- **Fashion-MNIST** — grayscale clothing images (784 features).
- **20 Newsgroups** — text, pre-processed to 50D via TF-IDF + truncated SVD.

Metrics with `k=7` neighbors (all higher is better, max 1.0 for trustworthiness/continuity):
- **kNN accuracy** — accuracy of a 7-NN classifier in the 2D embedding (class-structure preservation).
- **Trustworthiness** — fraction of embedding-space neighbors that are also neighbors in the original space.
- **Continuity** — fraction of original-space neighbors that remain neighbors in the embedding.
