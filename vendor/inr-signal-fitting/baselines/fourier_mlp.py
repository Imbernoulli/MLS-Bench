"""Reference: Fourier-feature encoding + ReLU MLP (Tancik et al., NeurIPS 2020). STRONG.

Lifts (x, y) through gamma(v) = [cos(2*pi*B v), sin(2*pi*B v)], B ~ N(0, sigma^2), then a
ReLU MLP. Overcomes spectral bias => recovers high-frequency detail => HIGH PSNR. sigma
is tuned to the signal band (the well-tuned reference used across the inr-* tasks).
"""
from __future__ import annotations

import torch

import common


def fit_inr(coords, target, dev, sigma: float = 10.0):
    encoder = common.FourierFeatures(in_dim=2, num_freqs=common.FOURIER_FREQS, sigma=sigma)
    model = common.build_relu_mlp_head(encoder.out_dim)
    model = common.train_inr(model, coords, target, dev, encoder=encoder,
                             label=f"fourier_sigma{sigma}")
    enc = encoder.to(dev)

    @torch.no_grad()
    def predict(coords):
        model.eval()
        return model(enc(coords.to(dev))).detach()

    return predict
