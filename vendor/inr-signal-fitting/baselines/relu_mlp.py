"""Reference: PLAIN ReLU coordinate MLP (no encoding). WEAK baseline (spectral bias).

Fits raw (x, y) with a ReLU MLP. Rahaman (2019) spectral bias => only low frequencies
are learned => blurry reconstruction, LOW PSNR (worst on medium/high-frequency signals).
"""
from __future__ import annotations

import torch

import common


def fit_inr(coords, target, dev):
    model = common.build_relu_mlp(in_dim=2)
    model = common.train_inr(model, coords, target, dev, encoder=None, label="relu")

    @torch.no_grad()
    def predict(coords):
        model.eval()
        return model(coords.to(dev)).detach()

    return predict
