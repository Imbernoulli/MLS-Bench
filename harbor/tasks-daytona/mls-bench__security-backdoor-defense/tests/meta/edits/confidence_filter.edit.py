"""Confidence-filter baseline for security-backdoor-defense.

Simple confidence-based scoring: high softmax confidence on the
*predicted* class combined with per-class analysis.  Poisoned samples
often show higher confidence for the target class than genuine samples.
"""

_FILE = "pytorch-vision/bench/backdoor/custom_backdoor_defense.py"

_CONTENT = """\
class BackdoorDefense:
    \"\"\"Per-class confidence outlier scoring.

    For each class, compute the mean and std of max softmax confidence.
    Score each sample by how many standard deviations above the class
    mean its confidence is.  Backdoor samples tend to be confidently
    classified into the target class.
    \"\"\"

    def __init__(self):
        self.class_conf_stats = {}  # cls -> (mean, std)

    def fit(self, features, labels, poison_fraction, **kwargs):
        features = np.asarray(features)
        labels = np.asarray(labels)
        # We don't use features here, just labels for class grouping
        self.labels = labels

    def score_samples(self, features, logits):
        probs = torch.softmax(torch.as_tensor(logits), dim=1).cpu().numpy()
        max_conf = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        # Compute per-class confidence stats
        class_stats = {}
        for cls in np.unique(preds):
            mask = preds == cls
            cls_conf = max_conf[mask]
            class_stats[int(cls)] = (cls_conf.mean(), cls_conf.std() + 1e-8)
        # Score = z-score of confidence within predicted class
        scores = np.zeros(len(logits))
        for i in range(len(logits)):
            cls = int(preds[i])
            mean, std = class_stats[cls]
            scores[i] = (max_conf[i] - mean) / std
        return scores
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 9, "end_line": 48, "content": _CONTENT}
]
