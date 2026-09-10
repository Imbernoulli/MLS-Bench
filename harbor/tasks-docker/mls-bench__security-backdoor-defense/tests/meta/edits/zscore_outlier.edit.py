"""Z-score outlier baseline for security-backdoor-defense.

NOTE: This is a generic per-class Mahalanobis-style z-score outlier
detector on penultimate-layer features.  It is NOT the Fine-Pruning
defense of Liu, Dolan-Gavitt, Garg (RAID 2018,
https://arxiv.org/abs/1805.12185), which operates at the *model* level
by pruning dormant neurons and fine-tuning on clean data.  That
model-level defense does not fit this task's sample-scoring
(score_samples) interface, so we provide this lightweight activation
outlier baseline under its true name.

Idea: within each class (by training label), compute mean/std per
feature dimension and score each sample by its mean squared z-score
(diagonal Mahalanobis distance).  Poisoned samples that activate
normally-dormant neurons in the target class show high average
squared z-scores.
"""

_FILE = "pytorch-vision/bench/backdoor/custom_backdoor_defense.py"

_CONTENT = """\
class BackdoorDefense:
    \"\"\"Per-class z-score outlier detector on penultimate features.

    NOT the Fine-Pruning defense (Liu et al., RAID 2018); that
    defense prunes neurons + fine-tunes on clean data, which needs
    model access incompatible with the score_samples interface here.
    This baseline is a simple activation outlier z-score: for each
    class, compute per-dim mean/std, and score each sample by
    mean(((x - mu) / sigma)^2).  Poisoned inputs activating
    normally-dormant dims earn high scores.
    \"\"\"

    def __init__(self):
        self.class_stats = {}       # cls -> (mean, std) per feature dim
        self.cached_labels = None

    def fit(self, features, labels, poison_fraction, **kwargs):
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        self.cached_labels = labels.copy()
        for cls in np.unique(labels):
            mask = labels == cls
            cls_feat = features[mask]
            mean = cls_feat.mean(axis=0)
            std = cls_feat.std(axis=0) + 1e-8
            self.class_stats[int(cls)] = (mean, std)

    def score_samples(self, features, logits):
        features = np.asarray(features, dtype=np.float64)
        # Use cached training labels so the per-class stats apply to
        # samples of that class, rather than the predicted class.
        if self.cached_labels is not None and len(self.cached_labels) == len(features):
            cls_for_sample = self.cached_labels
        else:
            cls_for_sample = np.asarray(logits).argmax(axis=1)

        scores = np.zeros(len(features), dtype=np.float64)
        for cls, (mean, std) in self.class_stats.items():
            mask = (cls_for_sample == cls)
            if not mask.any():
                continue
            z = (features[mask] - mean) / std
            scores[mask] = np.mean(z * z, axis=1)
        return scores
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 9, "end_line": 48, "content": _CONTENT}
]
