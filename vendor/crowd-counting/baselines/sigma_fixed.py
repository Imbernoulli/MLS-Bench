"""Weak baseline for cv-count-sigma: OVERSIZED fixed Gaussian kernel.

Every GT point is blurred with a large FIXED sigma (px). In dense/occluded scenes the
big kernels heavily OVERLAP and smear neighbouring objects together, so the density
target is over-smoothed and the model cannot resolve individual objects -> higher
counting MAE in crowded regions. (Fixed-sigma is fine only for sparse scenes; it hurts
as density grows -- exactly why MCNN/CSRNet moved to geometry-adaptive kernels.)
"""
import numpy as np


def gt_sigma(points, H, W):
    # Oversized fixed kernel (px): smears dense scenes.
    return np.full((len(points),), 14.0, dtype=np.float32)
