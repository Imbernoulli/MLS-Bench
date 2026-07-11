"""Design the Training-Example Sampling — weak baseline (lenbalance).

Reference implementation for the caption-train-sampling surface (sample_weights). See tasks/caption-train-sampling/edits/lenbalance.edit.py.
"""
import numpy as np


def sample_weights(emb, caps):
    # Length-balanced sampling: bucket captions by word length and give each
    # bucket equal total mass, so short and long captions are seen equally often.
    n = len(caps)
    lens = np.array([max(1, len(c.split())) for c in caps], dtype=np.float64)
    bins = np.clip(np.round(lens / 3.0), 0, None).astype(int)
    w = np.ones(n, dtype=np.float64)
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        w[idx] = 1.0 / max(len(idx), 1)
    return w
