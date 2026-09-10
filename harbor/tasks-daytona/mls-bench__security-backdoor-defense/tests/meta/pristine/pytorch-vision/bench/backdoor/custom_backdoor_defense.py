"""Editable backdoor defense for MLS-Bench."""

import numpy as np
import torch

# ============================================================
# EDITABLE
# ============================================================
class BackdoorDefense:
    """Sample-scoring defense for poisoned-example filtering.

    Given penultimate-layer features and model logits from a
    backdoor-poisoned training set, score each sample so that
    higher scores indicate more suspicious (likely poisoned)
    examples. The fixed harness will remove the top-scoring
    fraction and retrain on the filtered data.
    """

    def __init__(self):
        self.class_means = None

    def fit(self, features, labels, poison_fraction, **kwargs):
        """Fit the defense on the training features.

        Args:
            features: (N, D) feature matrix from penultimate layer.
            labels: (N,) integer labels (after poisoning).
            poison_fraction: approximate fraction of poisoned samples.
        """
        features = np.asarray(features)
        labels = np.asarray(labels)
        self.class_means = {}
        for cls in np.unique(labels):
            cls_features = features[labels == cls]
            self.class_means[int(cls)] = cls_features.mean(axis=0)

    def score_samples(self, features, logits):
        """Score each sample's suspicion level.

        Args:
            features: (N, D) feature matrix from penultimate layer.
            logits: (N, C) model output logits.

        Returns:
            1D array/list of length N: higher = more suspicious.
        """
        probs = torch.softmax(torch.as_tensor(logits), dim=1).cpu().numpy()
        return probs.max(axis=1)
# ============================================================
# END EDITABLE
# ============================================================
