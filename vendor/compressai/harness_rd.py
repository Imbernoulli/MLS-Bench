#!/usr/bin/env python3
"""compress-rd-target harness (FIXED pipeline).

RQ: rate-distortion loss balancing / TARGET-RATE control. A learned codec is
trained with  loss = lambda * distortion + rate.  The single knob lambda decides
where on the R-D curve the model lands. The task: hit a FIXED TARGET bitrate while
maximizing reconstruction quality. The agent returns the R-D control via
`rd_control()` in solution/rd_control.py; the transform, entropy model, channel
counts, optimizer, step budget, patches, Kodak eval, PSNR and bpp are all FIXED.

`rd_control()` returns a dict:
  {"lmbda": <float>, "target_bpp": <float or None>, "rate_gain": <float>}
  - lmbda:      fixed Lagrangian weight on 255^2*MSE (the classic knob).
  - target_bpp: if set, a proportional controller nudges an effective lambda each
                step to steer bpp toward this target (target-rate control).
  - rate_gain:  extra multiplicative weight on the rate term (>=0).

Metric (fixed and rate-aware):
  score = PSNR_dB - BETA * |bpp - TARGET_BPP|          (higher is better)
so both bitrate deviation and reconstruction quality affect the result.

Emits: COMPRESS_METRICS psnr=<..> bpp=<..> rd=<..> target=<..> elapsed=<..>
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import common
import nets
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel

TARGET_BPP = 0.35   # FIXED target bitrate the codec must hit
BETA = 12.0         # FIXED dB penalty per bpp of deviation from target


def validate_control(raw) -> dict:
    if not isinstance(raw, dict):
        raise TypeError("rd_control() must return a dict")
    unknown = set(raw) - {"lmbda", "target_bpp", "rate_gain"}
    if unknown:
        raise ValueError(f"unknown R-D control keys: {sorted(unknown)}")
    ctrl = dict(raw)
    lmbda = ctrl.get("lmbda", 0.02)
    rate_gain = ctrl.get("rate_gain", 1.0)
    target = ctrl.get("target_bpp", None)
    for name, value in (("lmbda", lmbda), ("rate_gain", rate_gain)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if not 1e-4 <= float(lmbda) <= 1.0:
        raise ValueError("lmbda must be in [1e-4, 1.0]")
    if not 0.0 <= float(rate_gain) <= 10.0:
        raise ValueError("rate_gain must be in [0, 10]")
    if target is not None:
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise TypeError("target_bpp must be numeric or None")
        if not math.isfinite(float(target)) or float(target) <= 0:
            raise ValueError("target_bpp must be finite and positive")
    return {"lmbda": float(lmbda), "target_bpp": target, "rate_gain": float(rate_gain)}


class RDHyperNet(CompressionModel):
    """Fixed scale-hyperprior; only the R-D loss weighting is agent-controlled."""

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


def train_rd(model, patches, ctrl, steps, batch, lr, device, seed):
    """FIXED training loop with agent-controlled R-D weighting / target-rate
    proportional control. The loss form and all hyper-params except (lmbda,
    target_bpp, rate_gain) are fixed."""
    from compressai.optimizers import net_aux_optimizer

    lmbda = float(ctrl.get("lmbda", 0.02))
    target = ctrl.get("target_bpp", None)
    rate_gain = float(ctrl.get("rate_gain", 1.0))

    model.to(device).train()
    conf = {"net": {"type": "Adam", "lr": lr}, "aux": {"type": "Adam", "lr": 1e-3}}
    opt = net_aux_optimizer(model, conf)
    net_opt, aux_opt = opt["net"], opt["aux"]
    g = torch.Generator().manual_seed(seed)
    N = patches.size(0)
    patches = patches.to(device)
    eff_lmbda = lmbda

    for step in range(steps):
        idx = torch.randint(0, N, (batch,), generator=g).to(device)
        x = patches[idx]
        net_opt.zero_grad()
        out = model(x)
        common.validate_codec_output(out, x)
        _, _, H, W = x.shape
        num_pixels = x.size(0) * H * W
        bpp = sum(torch.log(lk).sum() / (-math.log(2) * num_pixels)
                  for lk in out["likelihoods"].values())
        mse = F.mse_loss(out["x_hat"], x)
        # target-rate proportional control on lambda (bounded)
        if target is not None:
            err = bpp.detach().item() - float(target)
            eff_lmbda = eff_lmbda * math.exp(-0.05 * err)
            eff_lmbda = max(1e-4, min(eff_lmbda, 2.0))
        else:
            eff_lmbda = lmbda
        loss = eff_lmbda * (255.0 ** 2) * mse + rate_gain * bpp
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"training loss is non-finite at step {step}")
        loss.backward()
        common._require_finite_gradients(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm).item():
            raise RuntimeError(f"gradient norm is non-finite at step {step}")
        net_opt.step()
        aux_loss = model.aux_loss()
        if not torch.isfinite(aux_loss).item():
            raise RuntimeError(f"auxiliary loss is non-finite at step {step}")
        aux_opt.zero_grad()
        aux_loss.backward()
        common._require_finite_gradients(model)
        aux_opt.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"TRAIN step={step} loss={loss.item():.4f} "
                  f"bpp={bpp.detach().item():.4f} eff_lmbda={eff_lmbda:.4f}",
                  flush=True)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--M", type=int, default=96)
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

    rd_control = common.load_surface(args.solution, "rd_control")
    ctrl = validate_control(rd_control())
    print(f"RD_CONTROL {ctrl}", flush=True)

    model = RDHyperNet(args.N, args.M)
    patches = common.build_train_patches(data_dir, n_patches=1024, patch=128,
                                          seed=args.seed)
    train_rd(model, patches, ctrl, steps=args.steps, batch=8, lr=1e-4,
             device=device, seed=args.seed)

    eval_imgs = common.load_eval_images(data_dir)
    ps, bpp = common.evaluate(model, eval_imgs, device)
    rd = ps - BETA * abs(bpp - TARGET_BPP)
    dt = time.time() - t0
    print(f"COMPRESS_METRICS psnr={ps:.4f} bpp={bpp:.4f} rd={rd:.4f} "
          f"target={TARGET_BPP} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
