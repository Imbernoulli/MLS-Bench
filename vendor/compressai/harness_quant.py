#!/usr/bin/env python3
"""Fixed quantization-surrogate training and evaluation harness.

The editable surface selects one supported train-time surrogate. Transform,
entropy model, data, budget, optimizer, and the quantized evaluation path remain
fixed. The harness emits PSNR, bpp, and their fixed rate-distortion aggregate.
Invalid surfaces or runtime states are allowed to terminate the command so the
outer verifier can assign zero.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import common
import nets
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel

# PSNR-dominant RD scores reconstruction quality while retaining a rate penalty.
RD_LAMBDA = 3.0


def _surrogate(y, mode):
    """Return (y_recon, y_rate) at TRAIN time under the chosen surrogate.

    y_recon feeds the synthesis transform (distortion gradient); y_rate feeds the
    Gaussian likelihood (rate gradient). Both follow the agent's surrogate so all
    modes are genuinely distinct. EVAL always uses the FIXED honest round path."""
    y_noise = y + (torch.rand_like(y) - 0.5)
    y_hard = torch.round(y)
    y_ste = y + (y_hard - y).detach()
    if mode == "none":
        return y, y                    # no quantization at all -> collapses at eval
    if mode == "ste":
        return y_ste, y_ste            # STE for both branches
    if mode == "softround":
        T = 0.5
        r = y - torch.floor(y) - 0.5
        soft = torch.floor(y) + 0.5 + 0.5 * torch.tanh(r / T) / math.tanh(0.5 / T)
        return soft, soft
    if mode == "ste_noise":
        return y_ste, y_noise          # STE distortion, noise rate (universal-quant)
    return y_noise, y_noise            # "noise": additive uniform for both branches


class QuantHyperNet(CompressionModel):
    """Scale hyperprior whose y-quantization surrogate is agent-controlled.

    At TRAIN time both the reconstruction and the rate likelihood use the agent's
    surrogate (via a hand-computed Gaussian likelihood), so every mode is distinct.
    At EVAL the fully FIXED honest path (CompressAI's own round + likelihood) is
    used, so the surrogate is scored on the model it actually produced."""

    def __init__(self, N, M, mode):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.g_a = nets.g_a(N, M)
        self.g_s = nets.g_s(N, M)
        self.h_a = nets.h_a(N, M)
        self.h_s = nets.h_s_scale(N, M)
        self.gaussian_conditional = GaussianConditional(None)
        self.mode = mode

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_lk = self.entropy_bottleneck(z)
        scales = self.h_s(z_hat)
        if self.training:
            y_recon, y_rate = _surrogate(y, self.mode)
            y_lk = self.gaussian_conditional._likelihood(y_rate, scales)
            if self.gaussian_conditional.use_likelihood_bound:
                y_lk = self.gaussian_conditional.likelihood_lower_bound(y_lk)
            x_hat = self.g_s(y_recon)
        else:
            # FIXED honest eval: CompressAI's own STE-round + likelihood
            y_hat, y_lk = self.gaussian_conditional(y, scales)
            x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat, "likelihoods": {"y": y_lk, "z": z_lk}}


_VALID = {"none", "ste", "noise", "softround", "ste_noise"}


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
    data_dir = Path(args.data_dir)
    if args.setting:
        data_dir = data_dir / args.setting

    quantize = common.load_surface(args.solution, "quantize")
    mode = quantize()
    if not isinstance(mode, str):
        raise TypeError("quantize() must return a string")
    if mode not in _VALID:
        raise ValueError(f"invalid quantization surrogate: {mode!r}")
    model = QuantHyperNet(args.N, args.M, mode)
    print(f"DESIGN {mode} params={sum(p.numel() for p in model.parameters())}",
          flush=True)

    patches = common.build_train_patches(data_dir, n_patches=1024, patch=128,
                                          seed=args.seed)
    common.train_model(model, patches, lmbda=args.lmbda, steps=args.steps,
                       batch=8, lr=1e-4, device=device, seed=args.seed)

    eval_imgs = common.load_eval_images(data_dir)
    ps, bpp = common.evaluate(model, eval_imgs, device)
    rd = ps - RD_LAMBDA * bpp
    dt = time.time() - t0
    print(f"COMPRESS_METRICS psnr={ps:.4f} bpp={bpp:.4f} rd={rd:.4f} "
          f"design={mode} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
