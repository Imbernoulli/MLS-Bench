#!/usr/bin/env python3
"""Shared FIXED harness for the compress-* ARCHITECTURE-AXIS tasks.

Each task in this family probes ONE architectural design axis of a small
learned image codec (scale-hyperprior backbone) while freezing everything
else. The agent edits a single `<axis>()` surface in
solution/<axis>.py that returns a short config dict; `nets.build_transform` /
`nets.build_hyper_a` / `nets.build_hyper_s_scale` / `nets.SmallContextModel`
(all wrapping CompressAI's own real layers: GDN, ResidualBlock*, AttentionBlock,
MaskedConv2d, subpel_conv3x3, ...) turn that dict into the actual model.

Unlike the 3 original compress-* tasks (single "kodak" setting, fixed N/M/steps),
every task in this family is scored on THREE fixed content-complexity settings
(low / mid / high texture terciles of the 24 standard Kodak images; see
vendor/data_scripts/compressai/prepare_data.py).

Metric per setting (higher is better): rd = mean_PSNR(dB) - RD_LAMBDA * mean_bpp.
Emits one line per run:
  COMPRESS_METRICS psnr=<P> bpp=<B> rd=<R> design=<repr(cfg)> elapsed=<S>
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import common
import nets
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel

RD_LAMBDA = 8.0  # dB penalty per bpp; tuned so PSNR differences from architecture
                 # choices dominate small bpp deltas (this family varies capacity/
                 # wiring at a roughly fixed rate budget, not the rate itself).


class ArchNet(CompressionModel):
    """Scale-hyperprior codec whose g_a/g_s/h_a/h_s/context wiring is fully
    config-driven. Every knob defaults to the value that reproduces the
    ORIGINAL small transforms in nets.py (g_a/g_s/h_a/h_s_scale)."""

    def __init__(self, N: int, M: int, cfg: dict):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.g_a = nets.build_transform(
            3, M, N, M, up=False,
            depth=cfg.get("depth", 4),
            activation=cfg.get("activation", "gdn"),
            norm=cfg.get("norm", "none"),
            residual=cfg.get("residual", False),
            attention=cfg.get("attention", False),
        )
        self.g_s = nets.build_transform(
            M, 3, N, M, up=True,
            depth=cfg.get("depth", 4),
            activation=cfg.get("activation", "gdn"),
            norm=cfg.get("norm", "none"),
            residual=cfg.get("residual", False),
            attention=cfg.get("attention", False),
            upsample_mode=cfg.get("upsample_mode", "deconv"),
        )
        self.h_a = nets.build_hyper_a(N, M, depth=2)
        self.use_context = bool(cfg.get("use_context", False))
        if self.use_context:
            self.h_s = nets.h_s_meanscale(N, M)
            self.ctx = nets.SmallContextModel(N, M)
        else:
            self.h_s = nets.build_hyper_s_scale(N, M, depth=2)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)
        if self.use_context:
            z = self.h_a(y)
        else:
            z = self.h_a(torch.abs(y))
        z_hat, z_lk = self.entropy_bottleneck(z)
        if self.use_context:
            hyper_params = self.h_s(z_hat)
            y_hat_for_ctx = self.gaussian_conditional.quantize(
                y, "noise" if self.training else "dequantize"
            )
            scales, means = self.ctx(y_hat_for_ctx, hyper_params)
            y_hat, y_lk = self.gaussian_conditional(y, scales, means=means)
        else:
            scales = self.h_s(z_hat)
            y_hat, y_lk = self.gaussian_conditional(y, scales)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat, "likelihoods": {"y": y_lk, "z": z_lk}}


_SURFACE_KEYS = {
    "activation_design": "activation",
    "attention_design": "attention",
    "context_design": "use_context",
    "width_design": "N",
    "norm_design": "norm",
    "residual_design": "residual",
    "upsample_design": "upsample_mode",
}


def _clip_cfg(cfg: dict, expected_key: str) -> dict:
    """Validate an agent-returned config without replacing invalid choices."""
    if not isinstance(cfg, dict):
        raise TypeError("design surface must return a dict")
    if set(cfg) != {expected_key}:
        raise ValueError(f"design surface must return exactly the {expected_key!r} key")

    out = dict(cfg)
    choices = {
        "activation": {"gdn", "relu", "identity"},
        "norm": {"none", "batchnorm"},
        "upsample_mode": {"deconv", "subpel", "nearest"},
    }
    for key, values in choices.items():
        if key in out and out[key] not in values:
            raise ValueError(f"invalid {key}: {out[key]!r}")
    for key in ("residual", "attention", "use_context"):
        if key in out and not isinstance(out[key], bool):
            raise TypeError(f"{key} must be a bool")
    for key, lower, upper in (("N", 8, 128),):
        if key in out:
            value = out[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{key} must be an integer")
            if not lower <= value <= upper:
                raise ValueError(f"{key} must be in [{lower}, {upper}]")
    return out


def run_one(data_dir: Path, cfg: dict, N: int, M: int, steps: int, lmbda: float,
            seed: int, device) -> tuple[float, float]:
    common.set_seeds(seed)
    n_use = cfg.get("N", N)
    m_use = cfg.get("M", M)
    model = ArchNet(n_use, m_use, cfg)
    patches = common.build_train_patches(data_dir, n_patches=1024, patch=128, seed=seed)
    common.train_model(model, patches, lmbda=lmbda, steps=steps, batch=8, lr=1e-4,
                        device=device, seed=seed)
    eval_imgs = common.load_eval_images(data_dir)
    ps, bpp = common.evaluate(model, eval_imgs, device)
    return ps, bpp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="Root containing low/mid/high split subdirs (each with manifest.json)")
    ap.add_argument("--setting", required=True, choices=["low", "mid", "high"])
    ap.add_argument("--solution", required=True)
    ap.add_argument("--surface", required=True,
                    help="Attribute name of the config-returning function in --solution")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--M", type=int, default=96)
    ap.add_argument("--lmbda", type=float, default=0.02)
    args = ap.parse_args()

    if args.surface not in _SURFACE_KEYS:
        raise ValueError(f"unsupported architecture surface: {args.surface!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    data_dir = Path(args.data_dir) / args.setting

    surface_fn = common.load_surface(args.solution, args.surface)
    cfg = _clip_cfg(surface_fn(), _SURFACE_KEYS[args.surface])
    print(f"DESIGN {cfg}", flush=True)

    ps, bpp = run_one(data_dir, cfg, args.N, args.M, args.steps, args.lmbda,
                       args.seed, device)
    rd = ps - RD_LAMBDA * bpp
    dt = time.time() - t0
    print(f"COMPRESS_METRICS psnr={ps:.4f} bpp={bpp:.4f} rd={rd:.4f} "
          f"design={cfg} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
