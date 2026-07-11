#!/usr/bin/env python3
"""Fixed entropy-model comparison harness for learned image compression.

The editable surface selects one supported entropy-model implementation. The
transforms, data, channel counts, optimizer, budget, metric, and low/mid/high
evaluation settings are fixed. Invalid surfaces and non-finite runtime state
terminate the command instead of selecting another design.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common
import nets
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel

RD_LAMBDA = 20.0  # dB penalty per bpp (fixed); ~1.7 dB per 0.085 bpp step

# --------------------------------------------------------------------------- #
# Classical pre-entropy-model reference point: a plain autoencoder with NO   #
# entropy model at all (fixed, non-adaptive bit cost -- no entropy coding).  #
# --------------------------------------------------------------------------- #

SIMPLE_AE_BITS = 8.0  # fixed bits / latent element (naive fixed-length code)
_DOWNSAMPLE = 16      # nets.g_a is 4 stride-2 convs -> 2**4 spatial downsample


class SimpleAENet(nn.Module):
    """Plain autoencoder: same g_a/g_s transforms, latent y is STE-rounded then
    passed straight to g_s. It has no learned entropy model or rate signal at
    train time; evaluation assigns the fixed SIMPLE_AE_BITS cost per latent
    element."""

    def __init__(self, N, M):
        super().__init__()
        self.g_a = nets.g_a(N, M)
        self.g_s = nets.g_s(N, M)
        self.M = M

    def forward(self, x):
        y = self.g_a(x)
        y_hat = y + (torch.round(y) - y).detach()
        x_hat = self.g_s(y_hat)
        return x_hat


def _train_simple_ae(model, patches, steps, batch, lr, device, seed):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    N = patches.size(0)
    patches = patches.to(device)
    for step in range(steps):
        idx = torch.randint(0, N, (batch,), generator=g).to(device)
        x = patches[idx]
        opt.zero_grad()
        x_hat = model(x)
        loss = F.mse_loss(x_hat, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"TRAIN step={step} loss={loss.item():.4f} bpp=fixed", flush=True)
    return model


def _eval_simple_ae(model, eval_imgs, device):
    model.eval()
    psnrs = []
    with torch.no_grad():
        for img in eval_imgs:
            x = img.unsqueeze(0).to(device)
            x_hat = model(x)
            psnrs.append(common.psnr(x, x_hat))
    bpp = model.M * SIMPLE_AE_BITS / (_DOWNSAMPLE ** 2)
    return float(np.mean(psnrs)), bpp


class FactorizedNet(CompressionModel):
    def __init__(self, N, M):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(M)
        self.g_a = nets.g_a(N, M)
        self.g_s = nets.g_s(N, M)

    def forward(self, x):
        y = self.g_a(x)
        y_hat, y_lk = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat, "likelihoods": {"y": y_lk}}


class ScaleHyperNet(CompressionModel):
    def __init__(self, N, M):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.g_a = nets.g_a(N, M)
        self.g_s = nets.g_s(N, M)
        self.h_a = nets.h_a(N, M)
        self.h_s = nets.h_s_scale(N, M)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_lk = self.entropy_bottleneck(z)
        scales = self.h_s(z_hat)
        y_hat, y_lk = self.gaussian_conditional(y, scales)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat, "likelihoods": {"y": y_lk, "z": z_lk}}


class MeanScaleNet(CompressionModel):
    def __init__(self, N, M):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.g_a = nets.g_a(N, M)
        self.g_s = nets.g_s(N, M)
        self.h_a = nets.h_a(N, M)
        self.h_s = nets.h_s_meanscale(N, M)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_lk = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales, means = gaussian_params.chunk(2, 1)
        y_hat, y_lk = self.gaussian_conditional(y, scales, means=means)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat, "likelihoods": {"y": y_lk, "z": z_lk}}


_BUILDERS = {
    "factorized": FactorizedNet,
    "hyperprior_scale": ScaleHyperNet,
    "meanscale": MeanScaleNet,
}


def build(design: str, N: int, M: int):
    if design not in _BUILDERS:
        raise ValueError(f"invalid entropy-model design: {design!r}")
    return _BUILDERS[design](N, M), design


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--M", type=int, default=96)
    ap.add_argument("--lmbda", type=float, default=0.02)
    ap.add_argument("--setting", choices=["low", "mid", "high"], default=None,
                     help="If set, evaluate on data-dir/<setting> (3-way content-"
                          "complexity split) instead of the flat original split.")
    args = ap.parse_args()

    common.set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    from pathlib import Path
    data_dir = Path(args.data_dir)
    if args.setting:
        data_dir = data_dir / args.setting

    entropy_model = common.load_surface(args.solution, "entropy_model")
    design = entropy_model()
    if not isinstance(design, str):
        raise TypeError("entropy_model() must return a string")
    if design not in {"simple_ae", *_BUILDERS}:
        raise ValueError(f"invalid entropy-model design: {design!r}")

    eval_imgs = common.load_eval_images(data_dir)

    if design == "simple_ae":
        # Same transforms and training budget, with a fixed latent bit cost.
        model = SimpleAENet(args.N, args.M)
        print(f"DESIGN simple_ae params={sum(p.numel() for p in model.parameters())}",
              flush=True)
        patches = common.build_train_patches(data_dir, n_patches=1024, patch=128,
                                              seed=args.seed)
        _train_simple_ae(model, patches, steps=args.steps, batch=8, lr=1e-4,
                          device=device, seed=args.seed)
        ps, bpp = _eval_simple_ae(model, eval_imgs, device)
        rd = ps - RD_LAMBDA * bpp
        dt = time.time() - t0
        print(f"COMPRESS_METRICS psnr={ps:.4f} bpp={bpp:.4f} rd={rd:.4f} "
              f"design=simple_ae elapsed={dt:.1f}", flush=True)
        return

    model, design = build(design, args.N, args.M)
    print(f"DESIGN {design} params={sum(p.numel() for p in model.parameters())}",
          flush=True)

    patches = common.build_train_patches(data_dir, n_patches=1024, patch=128,
                                          seed=args.seed)
    common.train_model(model, patches, lmbda=args.lmbda, steps=args.steps,
                       batch=8, lr=1e-4, device=device, seed=args.seed)

    ps, bpp = common.evaluate(model, eval_imgs, device)
    rd = ps - RD_LAMBDA * bpp
    dt = time.time() - t0
    print(f"COMPRESS_METRICS psnr={ps:.4f} bpp={bpp:.4f} rd={rd:.4f} "
          f"design={design} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
