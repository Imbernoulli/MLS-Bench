"""Design the Training-Example Sampling — strong baseline (uniform).

Reference implementation for the caption-train-sampling surface (sample_weights). See tasks/caption-train-sampling/edits/uniform.edit.py.
"""
import numpy as np


def sample_weights(emb, caps):
    # Uniform: every training image is sampled with equal probability.
    return np.ones(len(caps), dtype=np.float64)
