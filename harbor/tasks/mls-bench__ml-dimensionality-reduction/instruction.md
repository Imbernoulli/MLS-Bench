# MLS-Bench: ml-dimensionality-reduction

# Dimensionality Reduction: Nonlinear Embedding Method Design

## Research Question
Design a novel nonlinear dimensionality-reduction method that embeds high-dimensional data into 2D while preserving both local neighborhoods and global structure better than existing methods. The contribution is the *embedding algorithm*: graph construction, neighbor forces, optimization schedule, hybrid linear/nonlinear stages — without using class labels at fit time.

## Background
Linear projections are fast but miss nonlinear manifold structure. Modern neighbor-embedding methods trade off local/global preservation differently.

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
- Must respect `random_state` for reproducibility.
- Must finish within a few minutes per dataset on CPU.
- Available libraries: `numpy`, `scipy`, `scikit-learn`.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will be rejected.

- `scikit-learn/bench/custom_dimred.py`
- editable lines **15–59**

## Readable Context

### `scikit-learn/bench/custom_dimred.py`  [EDITABLE — lines 15–59 only]

```python
     1: """Custom dimensionality reduction scaffold."""
     2:
     3: import numpy as np
     4: from numpy.typing import NDArray
     5:
     6: # =====================================================================
     7: # EDITABLE: implement CustomDimReduction below  (lines 15-59)
     8: # =====================================================================
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
    30:     The method should preserve both local neighborhood structure and
    31:     global data relationships.
    32:     """
    33:
    34:     def __init__(self, n_components: int = 2, random_state: int | None = None):
    35:         self.n_components = n_components
    36:         self.random_state = random_state
    37:
    38:     def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
    39:         """Reduce dimensionality of X.
    40:
    41:         Parameters
    42:         ----------
    43:         X : ndarray of shape (n_samples, n_features)
    44:             High-dimensional input data (standardized).
    45:
    46:         Returns
    47:         -------
    48:         X_reduced : ndarray of shape (n_samples, n_components)
    49:             Low-dimensional embedding.
    50:         """
    51:         rng = np.random.RandomState(self.random_state)
    52:         n_samples, n_features = X.shape
    53:         projection = rng.randn(n_features, self.n_components)
    54:         projection /= np.linalg.norm(projection, axis=0, keepdims=True)
    55:         X_reduced = X @ projection
    56:         return X_reduced
    59: # =====================================================================
    60: # END EDITABLE
    61: # =====================================================================
```

## Tips

- Keep the function/class signatures of the editable regions identical.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.

Good luck.
