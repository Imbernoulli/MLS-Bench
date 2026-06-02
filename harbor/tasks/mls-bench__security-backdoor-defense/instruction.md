# MLS-Bench: security-backdoor-defense

# Backdoor Defense via Poisoned-Sample Scoring

## Research Question
How can we design a better poisoned-sample scoring rule that identifies backdoored training examples while preserving clean utility after filtering and retraining?

## Background
Backdoor attacks (BadNets: Gu, Dolan-Gavitt, Garg, 2017, arXiv:1708.06733; Blended attack: Chen et al., 2017, arXiv:1712.05526) implant a trigger pattern into a subset of training examples and relabel them to an attacker-chosen target class. Models trained on this data retain high clean accuracy while also predicting the target label whenever the trigger appears. Many defenses try to identify suspicious points using feature statistics (Spectral Signatures: Tran, Li, Madry, NeurIPS 2018, arXiv:1811.00636), confidence patterns, or clustering structure (Activation Clustering: Chen et al., 2018, arXiv:1811.03728) before retraining on the filtered set, while pruning-style defenses (Fine-Pruning: Liu, Dolan-Gavitt, Garg, 2018, arXiv:1805.12185) operate on activations of suspicious neurons.

## Task
Implement a stronger backdoor defense in `bench/backdoor/custom_backdoor_defense.py`. The fixed harness will:

1. Construct a poisoned training set for a fixed trigger pattern (full dataset, no subsampling).
2. Train a victim model on the poisoned data for 100 epochs (SGD + CosineAnnealingLR).
3. Extract features from the penultimate layer and logits for the entire training set.
4. Call your defense to assign suspicion scores to training examples.
5. Remove the top `1.5 * epsilon` fraction of highest-scoring samples (over-estimate of the poison count, as recommended by Tran et al., 2018, Sec. 4.1) and retrain on the filtered set for 100 epochs.
6. Evaluate the retrained model.

The objective is to reduce the backdoor attack success rate without sacrificing clean accuracy.

## Editable Interface
You must implement:

```python
class BackdoorDefense:
    def fit(self, features, labels, poison_fraction, **kwargs):
        ...

    def score_samples(self, features, logits):
        ...
```

- `features`: feature matrix of shape `(N, D)` from a fixed penultimate layer.
- `labels`: training labels after poisoning.
- `poison_fraction`: approximate fraction of poisoned points in the training data.
- `logits`: model logits of shape `(N, C)`.
- Return value from `score_samples`: 1-D suspicion scores; higher means more suspicious.

The model architecture, poison injection process, filtering budget, and retraining schedule are fixed.

## Baselines
The baselines below run inside the same harness via edit ops:

- `confidence_filter`: ranks samples by target-label confidence.
- `spectral_signature`: scores by leading singular-vector outlier magnitude (Tran et al., NeurIPS 2018, arXiv:1811.00636).
- `activation_clustering`: class-conditional cluster-distance heuristic (Chen et al., 2018, arXiv:1811.03728).
- `zscore_outlier`: per-class feature z-score outlier heuristic used as a simple statistical reference.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-vision/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `pytorch-vision/bench/backdoor/custom_backdoor_defense.py`
- editable: **entire file**


Other files you may **read** for context (do not modify):
- `pytorch-vision/bench/backdoor/run_backdoor_defense.py`


## Readable Context


### `pytorch-vision/bench/backdoor/custom_backdoor_defense.py`  [EDITABLE — entire file only]

```python
     1: """Editable backdoor defense for MLS-Bench."""
     2: 
     3: import numpy as np
     4: import torch
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: # ============================================================
     9: class BackdoorDefense:
    10:     """Sample-scoring defense for poisoned-example filtering.
    11: 
    12:     Given penultimate-layer features and model logits from a
    13:     backdoor-poisoned training set, score each sample so that
    14:     higher scores indicate more suspicious (likely poisoned)
    15:     examples. The fixed harness will remove the top-scoring
    16:     fraction and retrain on the filtered data.
    17:     """
    18: 
    19:     def __init__(self):
    20:         self.class_means = None
    21: 
    22:     def fit(self, features, labels, poison_fraction, **kwargs):
    23:         """Fit the defense on the training features.
    24: 
    25:         Args:
    26:             features: (N, D) feature matrix from penultimate layer.
    27:             labels: (N,) integer labels (after poisoning).
    28:             poison_fraction: approximate fraction of poisoned samples.
    29:         """
    30:         features = np.asarray(features)
    31:         labels = np.asarray(labels)
    32:         self.class_means = {}
    33:         for cls in np.unique(labels):
    34:             cls_features = features[labels == cls]
    35:             self.class_means[int(cls)] = cls_features.mean(axis=0)
    36: 
    37:     def score_samples(self, features, logits):
    38:         """Score each sample's suspicion level.
    39: 
    40:         Args:
    41:             features: (N, D) feature matrix from penultimate layer.
    42:             logits: (N, C) model output logits.
    43: 
    44:         Returns:
    45:             1D array/list of length N: higher = more suspicious.
    46:         """
    47:         probs = torch.softmax(torch.as_tensor(logits), dim=1).cpu().numpy()
    48:         return probs.max(axis=1)
    49: # ============================================================
    50: # END EDITABLE
    51: # ============================================================
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `confidence_filter` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/bench/backdoor/custom_backdoor_defense.py`:

```python
     1: """Editable backdoor defense for MLS-Bench."""
     2: 
     3: import numpy as np
     4: import torch
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: # ============================================================
     9: class BackdoorDefense:
    10:     """Per-class confidence outlier scoring.
    11: 
    12:     For each class, compute the mean and std of max softmax confidence.
    13:     Score each sample by how many standard deviations above the class
    14:     mean its confidence is.  Backdoor samples tend to be confidently
    15:     classified into the target class.
    16:     """
    17: 
    18:     def __init__(self):
    19:         self.class_conf_stats = {}  # cls -> (mean, std)
    20: 
    21:     def fit(self, features, labels, poison_fraction, **kwargs):
    22:         features = np.asarray(features)
    23:         labels = np.asarray(labels)
    24:         # We don't use features here, just labels for class grouping
    25:         self.labels = labels
    26: 
    27:     def score_samples(self, features, logits):
    28:         probs = torch.softmax(torch.as_tensor(logits), dim=1).cpu().numpy()
    29:         max_conf = probs.max(axis=1)
    30:         preds = probs.argmax(axis=1)
    31:         # Compute per-class confidence stats
    32:         class_stats = {}
    33:         for cls in np.unique(preds):
    34:             mask = preds == cls
    35:             cls_conf = max_conf[mask]
    36:             class_stats[int(cls)] = (cls_conf.mean(), cls_conf.std() + 1e-8)
    37:         # Score = z-score of confidence within predicted class
    38:         scores = np.zeros(len(logits))
    39:         for i in range(len(logits)):
    40:             cls = int(preds[i])
    41:             mean, std = class_stats[cls]
    42:             scores[i] = (max_conf[i] - mean) / std
    43:         return scores
    44: # ============================================================
    45: # END EDITABLE
    46: # ============================================================
```

### `spectral_signature` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/bench/backdoor/custom_backdoor_defense.py`:

```python
     1: """Editable backdoor defense for MLS-Bench."""
     2: 
     3: import numpy as np
     4: import torch
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: # ============================================================
     9: class BackdoorDefense:
    10:     """Per-class spectral signatures (Tran et al., NeurIPS 2018).
    11: 
    12:     Faithful to Algorithm 1 of the paper and the MadryLab reference
    13:     implementation at
    14:       https://github.com/MadryLab/backdoor_data_poisoning/blob/master/compute_corr.py
    15: 
    16:     For each class c: compute the mean mu_c of its penultimate-layer
    17:     features, center them, take the top right singular vector v_c of
    18:     the centered matrix, and score each sample by
    19:         tau_i = ((r_i - mu_c) . v_c)^2.
    20: 
    21:     We remember the training labels from fit() so that score_samples()
    22:     applies each sample's class-specific direction, rather than a
    23:     (potentially mismatched) predicted-class direction.
    24:     """
    25: 
    26:     def __init__(self):
    27:         self.class_centers = {}       # cls -> mean vector (D,)
    28:         self.class_directions = {}    # cls -> top right singular vector (D,)
    29:         self.cached_labels = None     # training labels from fit()
    30: 
    31:     def fit(self, features, labels, poison_fraction, **kwargs):
    32:         features = np.asarray(features, dtype=np.float64)
    33:         labels = np.asarray(labels)
    34:         self.cached_labels = labels.copy()
    35:         for cls in np.unique(labels):
    36:             mask = labels == cls
    37:             cls_feat = features[mask]
    38:             if cls_feat.shape[0] < 2:
    39:                 # Degenerate class; zero direction
    40:                 self.class_centers[int(cls)] = cls_feat.mean(axis=0) if len(cls_feat) else np.zeros(features.shape[1])
    41:                 self.class_directions[int(cls)] = np.zeros(features.shape[1])
    42:                 continue
    43:             mu = cls_feat.mean(axis=0)
    44:             centered = cls_feat - mu
    45:             # Top right singular vector of the centered class matrix.
    46:             # np.linalg.svd returns vh with rows = right singular vectors.
    47:             _, _, vh = np.linalg.svd(centered, full_matrices=False)
    48:             self.class_centers[int(cls)] = mu
    49:             self.class_directions[int(cls)] = vh[0]
    50: 
    51:     def score_samples(self, features, logits):
    52:         features = np.asarray(features, dtype=np.float64)
    53:         # Use TRAINING LABELS (cached in fit) to pick the per-class
    54:         # direction.  Matches Algorithm 1 of Tran et al.  Using
    55:         # argmax(logits) would mismatch any sample whose prediction
    56:         # disagrees with its training label, corrupting the per-class
    57:         # projection.
    58:         if self.cached_labels is not None and len(self.cached_labels) == len(features):
    59:             cls_for_sample = self.cached_labels
    60:         else:
    61:             cls_for_sample = np.asarray(logits).argmax(axis=1)
    62: 
    63:         scores = np.zeros(len(features), dtype=np.float64)
    64:         for cls, direction in self.class_directions.items():
    65:             mask = (cls_for_sample == cls)
    66:             if not mask.any():
    67:                 continue
    68:             centered = features[mask] - self.class_centers[cls]
    69:             proj = centered @ direction
    70:             scores[mask] = proj * proj  # tau_i = (centered . v)^2
    71:         return scores
    72: # ============================================================
    73: # END EDITABLE
    74: # ============================================================
```

### `activation_clustering` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/bench/backdoor/custom_backdoor_defense.py`:

```python
     1: """Editable backdoor defense for MLS-Bench."""
     2: 
     3: import numpy as np
     4: import torch
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: # ============================================================
     9: class BackdoorDefense:
    10:     """Per-class 2-means activation clustering (Chen et al., 2018).
    11: 
    12:     For each class, split features into two clusters via k-means.
    13:     The smaller cluster in a class is likely the poisoned sub-population.
    14:     Samples belonging to the smaller cluster get high suspicion scores.
    15:     """
    16: 
    17:     def __init__(self):
    18:         self.class_labels_cluster = {}   # cls -> array of cluster labels per sample
    19:         self.class_small_center = {}     # cls -> centroid of smaller cluster
    20:         self.class_indices = {}          # cls -> global indices
    21: 
    22:     def _kmeans2(self, X, max_iter=50):
    23:         """Simple 2-means on rows of X. Returns labels (0 or 1)."""
    24:         n = len(X)
    25:         # Initialise with two random points (deterministic via spread)
    26:         idx0, idx1 = 0, n // 2
    27:         c0, c1 = X[idx0].copy(), X[idx1].copy()
    28:         labels = np.zeros(n, dtype=int)
    29:         for _ in range(max_iter):
    30:             d0 = np.linalg.norm(X - c0, axis=1)
    31:             d1 = np.linalg.norm(X - c1, axis=1)
    32:             new_labels = (d1 < d0).astype(int)  # 0 = closer to c0
    33:             new_labels = 1 - new_labels  # flip so 0 = closer to c0
    34:             # Actually: label 0 if closer to c0, label 1 if closer to c1
    35:             new_labels = (d1 < d0).astype(int)
    36:             if np.array_equal(new_labels, labels):
    37:                 break
    38:             labels = new_labels
    39:             mask0 = labels == 0
    40:             mask1 = labels == 1
    41:             if mask0.any():
    42:                 c0 = X[mask0].mean(axis=0)
    43:             if mask1.any():
    44:                 c1 = X[mask1].mean(axis=0)
    45:         # Determine which cluster is smaller
    46:         n0 = (labels == 0).sum()
    47:         n1 = (labels == 1).sum()
    48:         small_label = 0 if n0 <= n1 else 1
    49:         small_center = c0 if small_label == 0 else c1
    50:         return labels, small_label, small_center
    51: 
    52:     def fit(self, features, labels, poison_fraction, **kwargs):
    53:         features = np.asarray(features, dtype=np.float64)
    54:         labels_arr = np.asarray(labels)
    55:         for cls in np.unique(labels_arr):
    56:             mask = labels_arr == cls
    57:             cls_feat = features[mask]
    58:             indices = np.where(mask)[0]
    59:             self.class_indices[int(cls)] = indices
    60:             if len(cls_feat) < 4:
    61:                 # Too few samples to cluster
    62:                 self.class_labels_cluster[int(cls)] = np.zeros(len(cls_feat), dtype=int)
    63:                 self.class_small_center[int(cls)] = cls_feat.mean(axis=0)
    64:                 continue
    65:             clust_labels, small_label, small_center = self._kmeans2(cls_feat)
    66:             self.class_labels_cluster[int(cls)] = (clust_labels == small_label).astype(int)
    67:             self.class_small_center[int(cls)] = small_center
    68: 
    69:     def score_samples(self, features, logits):
    70:         features = np.asarray(features, dtype=np.float64)
    71:         preds = np.asarray(logits).argmax(axis=1)
    72:         scores = np.zeros(len(features))
    73:         for cls in self.class_indices:
    74:             indices = self.class_indices[cls]
    75:             is_small = self.class_labels_cluster[cls]
    76:             small_center = self.class_small_center[cls]
    77:             for j, global_idx in enumerate(indices):
    78:                 # Base score: membership in smaller cluster
    79:                 membership_score = float(is_small[j]) * 10.0
    80:                 # Plus: closeness to the small-cluster center
    81:                 dist = np.linalg.norm(features[global_idx] - small_center)
    82:                 closeness = 1.0 / (dist + 1e-8)
    83:                 scores[global_idx] = membership_score + closeness
    84:         return scores
    85: # ============================================================
    86: # END EDITABLE
    87: # ============================================================
```

### `zscore_outlier` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/bench/backdoor/custom_backdoor_defense.py`:

```python
     1: """Editable backdoor defense for MLS-Bench."""
     2: 
     3: import numpy as np
     4: import torch
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: # ============================================================
     9: class BackdoorDefense:
    10:     """Per-class z-score outlier detector on penultimate features.
    11: 
    12:     NOT the Fine-Pruning defense (Liu et al., RAID 2018); that
    13:     defense prunes neurons + fine-tunes on clean data, which needs
    14:     model access incompatible with the score_samples interface here.
    15:     This baseline is a simple activation outlier z-score: for each
    16:     class, compute per-dim mean/std, and score each sample by
    17:     mean(((x - mu) / sigma)^2).  Poisoned inputs activating
    18:     normally-dormant dims earn high scores.
    19:     """
    20: 
    21:     def __init__(self):
    22:         self.class_stats = {}       # cls -> (mean, std) per feature dim
    23:         self.cached_labels = None
    24: 
    25:     def fit(self, features, labels, poison_fraction, **kwargs):
    26:         features = np.asarray(features, dtype=np.float64)
    27:         labels = np.asarray(labels)
    28:         self.cached_labels = labels.copy()
    29:         for cls in np.unique(labels):
    30:             mask = labels == cls
    31:             cls_feat = features[mask]
    32:             mean = cls_feat.mean(axis=0)
    33:             std = cls_feat.std(axis=0) + 1e-8
    34:             self.class_stats[int(cls)] = (mean, std)
    35: 
    36:     def score_samples(self, features, logits):
    37:         features = np.asarray(features, dtype=np.float64)
    38:         # Use cached training labels so the per-class stats apply to
    39:         # samples of that class, rather than the predicted class.
    40:         if self.cached_labels is not None and len(self.cached_labels) == len(features):
    41:             cls_for_sample = self.cached_labels
    42:         else:
    43:             cls_for_sample = np.asarray(logits).argmax(axis=1)
    44: 
    45:         scores = np.zeros(len(features), dtype=np.float64)
    46:         for cls, (mean, std) in self.class_stats.items():
    47:             mask = (cls_for_sample == cls)
    48:             if not mask.any():
    49:                 continue
    50:             z = (features[mask] - mean) / std
    51:             scores[mask] = np.mean(z * z, axis=1)
    52:         return scores
    53: # ============================================================
    54: # END EDITABLE
    55: # ============================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
