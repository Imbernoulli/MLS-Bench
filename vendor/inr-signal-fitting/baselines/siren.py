"""Reference: SIREN (sinusoidal MLP + principled init), Sitzmann et al., NeurIPS 2020. SOTA.

Sine activations throughout with the SIREN initialization; the first sine layer is a
trainable Fourier encoding (w0=30). Highest reconstruction fidelity on natural signals
=> the SOTA reference PSNR the inr-* tasks aggregate over.
"""
from __future__ import annotations

import torch

import common


def fit_inr(coords, target, dev, w0: float = 30.0):
    model = common.SirenMLP(in_dim=2, w0=w0, w0_hidden=w0)
    model = common.train_inr(model, coords, target, dev, label=f"siren_w0{w0}")

    @torch.no_grad()
    def predict(coords):
        model.eval()
        return model(coords.to(dev)).detach()

    return predict
