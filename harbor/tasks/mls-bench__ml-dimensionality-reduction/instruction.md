# MLS-Bench: ml-dimensionality-reduction

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

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `scikit-learn/bench/custom_dimred.py`
- editable lines **15–59**




## Readable Context


### `scikit-learn/bench/custom_dimred.py`  [EDITABLE — lines 15–59 only]

```python
     1: """Custom dimensionality reduction benchmark -- agent-editable template.
     2: 
     3: The agent modifies `CustomDimReduction` to implement a novel nonlinear
     4: dimensionality reduction method.  The evaluation harness embeds three
     5: datasets into 2D, then measures kNN accuracy, trustworthiness, and
     6: continuity in the reduced space.
     7: """
     8: 
     9: import numpy as np
    10: from numpy.typing import NDArray
    11: 
    12: # =====================================================================
    13: # EDITABLE: implement CustomDimReduction below  (lines 15-59)
    14: # =====================================================================
    15: class CustomDimReduction:
    16:     """Custom dimensionality reduction method.
    17: 
    18:     Must implement fit_transform(X) -> X_reduced.
    19: 
    20:     Parameters
    21:     ----------
    22:     n_components : int
    23:         Target dimensionality (default 2).
    24:     random_state : int or None
    25:         Random seed for reproducibility.
    26: 
    27:     Notes
    28:     -----
    29:     You may use numpy and scipy (already installed).
    30:     The method should work on datasets with 1000-70000 samples
    31:     and 50-784 features, reducing to n_components=2 dimensions.
    32:     It should preserve both local neighborhood structure and
    33:     global data relationships.
    34:     """
    35: 
    36:     def __init__(self, n_components: int = 2, random_state: int | None = None):
    37:         self.n_components = n_components
    38:         self.random_state = random_state
    39: 
    40:     def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
    41:         """Reduce dimensionality of X.
    42: 
    43:         Parameters
    44:         ----------
    45:         X : ndarray of shape (n_samples, n_features)
    46:             High-dimensional input data (standardized).
    47: 
    48:         Returns
    49:         -------
    50:         X_reduced : ndarray of shape (n_samples, n_components)
    51:             Low-dimensional embedding.
    52:         """
    53:         # Default: random projection (poor baseline) -- replace with your design
    54:         rng = np.random.RandomState(self.random_state)
    55:         n_samples, n_features = X.shape
    56:         projection = rng.randn(n_features, self.n_components)
    57:         projection /= np.linalg.norm(projection, axis=0, keepdims=True)
    58:         X_reduced = X @ projection
    59:         return X_reduced
    60: # =====================================================================
    61: # END EDITABLE
    62: # =====================================================================
    63: 
    64: 
    65: # =====================================================================
    66: # FIXED -- evaluation harness below (do not modify)
    67: # =====================================================================
    68: import argparse
    69: import sys
    70: import os
    71: import time
    72: 
    73: from sklearn.datasets import fetch_openml, fetch_20newsgroups
    74: from sklearn.feature_extraction.text import TfidfVectorizer
    75: from sklearn.decomposition import TruncatedSVD
    76: from sklearn.preprocessing import StandardScaler
    77: from sklearn.neighbors import KNeighborsClassifier
    78: from sklearn.model_selection import StratifiedShuffleSplit
    79: from sklearn.metrics import accuracy_score
    80: 
    81: 
    82: def _trustworthiness(X_high, X_low, n_neighbors=7):
    83:     """Compute trustworthiness: are nearest neighbors in the embedding
    84:     also near in the original space?
    85: 
    86:     Measures preservation of local structure.
    87:     Range: [0, 1], higher is better.
    88:     """
    89:     from scipy.spatial.distance import cdist
    90: 
    91:     n = X_high.shape[0]
    92:     D_high = cdist(X_high, X_high, metric="sqeuclidean")
    93:     D_low = cdist(X_low, X_low, metric="sqeuclidean")
    94: 
    95:     k = min(n_neighbors, n - 1)
    96: 
    97:     # Ranks in original space
    98:     ranks_high = np.zeros_like(D_high, dtype=int)
    99:     for i in range(n):
   100:         order = np.argsort(D_high[i])
   101:         ranks_high[i, order] = np.arange(n)
   102: 
   103:     # k-nearest neighbors in low-dimensional space
   104:     trust_sum = 0.0
   105:     for i in range(n):
   106:         nn_low = np.argsort(D_low[i])[1 : k + 1]
   107:         for j in nn_low:
   108:             r = ranks_high[i, j]
   109:             if r > k:
   110:                 trust_sum += r - k
   111: 
   112:     denom = n * k * (2 * n - 3 * k - 1)
   113:     if denom == 0:
   114:         return 1.0
   115:     return 1.0 - (2.0 / denom) * trust_sum
   116: 
   117: 
   118: def _continuity(X_high, X_low, n_neighbors=7):
   119:     """Compute continuity: are nearest neighbors in the original space
   120:     also near in the embedding?
   121: 
   122:     Measures whether the embedding tears apart nearby points.
   123:     Range: [0, 1], higher is better.
   124:     """
   125:     from scipy.spatial.distance import cdist
   126: 
   127:     n = X_high.shape[0]
   128:     D_high = cdist(X_high, X_high, metric="sqeuclidean")
   129:     D_low = cdist(X_low, X_low, metric="sqeuclidean")
   130: 
   131:     k = min(n_neighbors, n - 1)
   132: 
   133:     # Ranks in low-dimensional space
   134:     ranks_low = np.zeros_like(D_low, dtype=int)
   135:     for i in range(n):
   136:         order = np.argsort(D_low[i])
   137:         ranks_low[i, order] = np.arange(n)
   138: 
   139:     # k-nearest neighbors in high-dimensional space
   140:     cont_sum = 0.0
   141:     for i in range(n):
   142:         nn_high = np.argsort(D_high[i])[1 : k + 1]
   143:         for j in nn_high:
   144:             r = ranks_low[i, j]
   145:             if r > k:
   146:                 cont_sum += r - k
   147: 
   148:     denom = n * k * (2 * n - 3 * k - 1)
   149:     if denom == 0:
   150:         return 1.0
   151:     return 1.0 - (2.0 / denom) * cont_sum
   152: 
   153: 
   154: def _knn_accuracy(X_reduced, y, n_neighbors=7, n_splits=3, seed=42):
   155:     """Evaluate kNN classification accuracy in the reduced space.
   156: 
   157:     Uses stratified shuffle split to get mean accuracy.
   158:     """
   159:     sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=seed)
   160:     accs = []
   161:     for train_idx, test_idx in sss.split(X_reduced, y):
   162:         clf = KNeighborsClassifier(n_neighbors=n_neighbors)
   163:         clf.fit(X_reduced[train_idx], y[train_idx])
   164:         y_pred = clf.predict(X_reduced[test_idx])
   165:         accs.append(accuracy_score(y[test_idx], y_pred))
   166:     return float(np.mean(accs))
   167: 
   168: 
   169: _MAX_SAMPLES = 5000  # subsample for tractability
   170: 
   171: _DATASETS = {
   172:     "mnist": {
   173:         "loader": lambda: fetch_openml(
   174:             "mnist_784", version=1,
   175:             data_home=os.environ.get("SKLEARN_DATA_HOME", None),
   176:             parser="auto", as_frame=False,
   177:         ),
   178:         "preprocess": None,  # already numeric
   179:     },
   180:     "fashion_mnist": {
   181:         "loader": lambda: fetch_openml(
   182:             "Fashion-MNIST", version=1,
   183:             data_home=os.environ.get("SKLEARN_DATA_HOME", None),
   184:             parser="auto", as_frame=False,
   185:         ),
   186:         "preprocess": None,
   187:     },
   188:     "newsgroups": {
   189:         "loader": lambda: None,  # special handling below
   190:         "preprocess": "tfidf",
   191:     },
   192: }
   193: 
   194: 
   195: def _load_dataset(name, seed=42):
   196:     """Load and preprocess a dataset, returning (X, y) with at most _MAX_SAMPLES."""
   197:     rng = np.random.RandomState(seed)
   198: 
   199:     if name == "newsgroups":
   200:         cats = [
   201:             "alt.atheism", "comp.graphics", "rec.sport.baseball",
   202:             "sci.med", "talk.politics.guns",
   203:         ]
   204:         data = fetch_20newsgroups(
   205:             subset="all", categories=cats,
   206:             data_home=os.environ.get("SKLEARN_DATA_HOME", None),
   207:             remove=("headers", "footers", "quotes"),
   208:         )
   209:         vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
   210:         X_tfidf = vectorizer.fit_transform(data.data)
   211:         # Reduce to 50 dims with TruncatedSVD for speed
   212:         svd = TruncatedSVD(n_components=50, random_state=seed)
   213:         X = svd.fit_transform(X_tfidf)
   214:         y = np.array(data.target)
   215:     else:
   216:         info = _DATASETS[name]
   217:         bunch = info["loader"]()
   218:         X = np.array(bunch.data, dtype=np.float64)
   219:         y = np.array(bunch.target)
   220:         # Ensure y is integer-encoded
   221:         if y.dtype.kind in ("U", "S", "O"):
   222:             from sklearn.preprocessing import LabelEncoder
   223:             y = LabelEncoder().fit_transform(y)
   224: 
   225:     # Subsample
   226:     n = X.shape[0]
   227:     if n > _MAX_SAMPLES:
   228:         idx = rng.choice(n, _MAX_SAMPLES, replace=False)
   229:         X, y = X[idx], y[idx]
   230: 
   231:     # Standardize
   232:     scaler = StandardScaler()
   233:     X = scaler.fit_transform(X)
   234: 
   235:     return X, y
   236: 
   237: 
   238: def evaluate(dataset_name: str, seed: int = 42, n_neighbors: int = 7):
   239:     """Run dimensionality reduction + evaluation on a single dataset."""
   240:     print(f"Loading dataset: {dataset_name} ...", flush=True)
   241:     X, y = _load_dataset(dataset_name, seed=seed)
   242:     print(f"  Shape: {X.shape}, classes: {len(np.unique(y))}", flush=True)
   243: 
   244:     print("Running custom dimensionality reduction ...", flush=True)
   245:     t0 = time.time()
   246:     reducer = CustomDimReduction(n_components=2, random_state=seed)
   247:     X_reduced = reducer.fit_transform(X)
   248:     elapsed = time.time() - t0
   249:     print(f"TRAIN_METRICS dataset={dataset_name} elapsed={elapsed:.2f}s", flush=True)
   250: 
   251:     # Validate output shape
   252:     assert X_reduced.shape == (X.shape[0], 2), (
   253:         f"Expected shape ({X.shape[0]}, 2), got {X_reduced.shape}"
   254:     )
   255:     assert np.all(np.isfinite(X_reduced)), "Output contains NaN or Inf"
   256: 
   257:     # Compute metrics
   258:     print("Computing kNN accuracy ...", flush=True)
   259:     knn_acc = _knn_accuracy(X_reduced, y, n_neighbors=n_neighbors, seed=seed)
   260: 
   261:     print("Computing trustworthiness ...", flush=True)
   262:     trust = _trustworthiness(X, X_reduced, n_neighbors=n_neighbors)
   263: 
   264:     print("Computing continuity ...", flush=True)
   265:     cont = _continuity(X, X_reduced, n_neighbors=n_neighbors)
   266: 
   267:     print(
   268:         f"DIMRED_METRICS knn_acc={knn_acc:.6f} "
   269:         f"trustworthiness={trust:.6f} continuity={cont:.6f} "
   270:         f"time={elapsed:.2f}",
   271:         flush=True,
   272:     )
   273: 
   274: 
   275: def main():
   276:     parser = argparse.ArgumentParser(
   277:         description="Dimensionality reduction benchmark"
   278:     )
   279:     parser.add_argument(
   280:         "--dataset",
   281:         required=True,
   282:         choices=list(_DATASETS.keys()),
   283:         help="Dataset to evaluate on",
   284:     )
   285:     parser.add_argument("--seed", type=int, default=42, help="Random seed")
   286:     parser.add_argument("--n_neighbors", type=int, default=7, help="k for kNN/trust/cont")
   287:     args = parser.parse_args()
   288: 
   289:     evaluate(args.dataset, seed=args.seed, n_neighbors=args.n_neighbors)
   290: 
   291: 
   292: if __name__ == "__main__":
   293:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `pca` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/bench/custom_dimred.py`:

```python
Lines 15–25:
    12: # =====================================================================
    13: # EDITABLE: implement CustomDimReduction below  (lines 15-59)
    14: # =====================================================================
    15: class CustomDimReduction:
    16:     """PCA dimensionality reduction (linear baseline)."""
    17: 
    18:     def __init__(self, n_components: int = 2, random_state: int | None = None):
    19:         self.n_components = n_components
    20:         self.random_state = random_state
    21: 
    22:     def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
    23:         from sklearn.decomposition import PCA
    24:         pca = PCA(n_components=self.n_components, random_state=self.random_state)
    25:         return pca.fit_transform(X)
    26: # =====================================================================
    27: # END EDITABLE
    28: # =====================================================================
```

### `tsne` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/bench/custom_dimred.py`:

```python
Lines 15–32:
    12: # =====================================================================
    13: # EDITABLE: implement CustomDimReduction below  (lines 15-59)
    14: # =====================================================================
    15: class CustomDimReduction:
    16:     """t-SNE dimensionality reduction."""
    17: 
    18:     def __init__(self, n_components: int = 2, random_state: int | None = None):
    19:         self.n_components = n_components
    20:         self.random_state = random_state
    21: 
    22:     def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
    23:         from sklearn.manifold import TSNE
    24:         tsne = TSNE(
    25:             n_components=self.n_components,
    26:             perplexity=30.0,
    27:             learning_rate="auto",
    28:             init="pca",
    29:             random_state=self.random_state,
    30:             n_iter=1000,
    31:         )
    32:         return tsne.fit_transform(X)
    33: # =====================================================================
    34: # END EDITABLE
    35: # =====================================================================
```

### `umap` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/bench/custom_dimred.py`:

```python
Lines 15–31:
    12: # =====================================================================
    13: # EDITABLE: implement CustomDimReduction below  (lines 15-59)
    14: # =====================================================================
    15: class CustomDimReduction:
    16:     """UMAP dimensionality reduction."""
    17: 
    18:     def __init__(self, n_components: int = 2, random_state: int | None = None):
    19:         self.n_components = n_components
    20:         self.random_state = random_state
    21: 
    22:     def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
    23:         import umap
    24:         reducer = umap.UMAP(
    25:             n_components=self.n_components,
    26:             n_neighbors=15,
    27:             min_dist=0.1,
    28:             metric="euclidean",
    29:             random_state=self.random_state,
    30:         )
    31:         return reducer.fit_transform(X)
    32: # =====================================================================
    33: # END EDITABLE
    34: # =====================================================================
```

### `trimap` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/bench/custom_dimred.py`:

```python
Lines 15–31:
    12: # =====================================================================
    13: # EDITABLE: implement CustomDimReduction below  (lines 15-59)
    14: # =====================================================================
    15: class CustomDimReduction:
    16:     """TriMap dimensionality reduction."""
    17: 
    18:     def __init__(self, n_components: int = 2, random_state: int | None = None):
    19:         self.n_components = n_components
    20:         self.random_state = random_state
    21: 
    22:     def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
    23:         import trimap
    24:         reducer = trimap.TRIMAP(
    25:             n_dims=self.n_components,
    26:             n_inliers=10,
    27:             n_outliers=5,
    28:             n_random=5,
    29:             n_iters=400,
    30:         )
    31:         return reducer.fit_transform(X)
    32: # =====================================================================
    33: # END EDITABLE
    34: # =====================================================================
```

### `pacmap` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/bench/custom_dimred.py`:

```python
Lines 15–31:
    12: # =====================================================================
    13: # EDITABLE: implement CustomDimReduction below  (lines 15-59)
    14: # =====================================================================
    15: class CustomDimReduction:
    16:     """PaCMAP dimensionality reduction (SOTA)."""
    17: 
    18:     def __init__(self, n_components: int = 2, random_state: int | None = None):
    19:         self.n_components = n_components
    20:         self.random_state = random_state
    21: 
    22:     def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
    23:         import pacmap
    24:         reducer = pacmap.PaCMAP(
    25:             n_components=self.n_components,
    26:             n_neighbors=10,
    27:             MN_ratio=0.5,
    28:             FP_ratio=2.0,
    29:             random_state=self.random_state,
    30:         )
    31:         return reducer.fit_transform(X)
    32: # =====================================================================
    33: # END EDITABLE
    34: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
