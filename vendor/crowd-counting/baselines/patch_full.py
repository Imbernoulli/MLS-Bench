"""Weak baseline for cv-count-patch: NO patch augmentation (full image).

Trains on the full 128x128 image as-is (identity). With only 120 training images and no
cropping augmentation, the counter sees few effective samples and generalises worse to
the shifted val counts -> higher counting MAE. This is the no-augmentation control.
"""
import numpy as np


def crop_train(img, pts):
    # Identity: return the full image and its points unchanged.
    return np.asarray(img, dtype=np.float32), np.asarray(pts, dtype=np.float32).reshape(-1, 2)
