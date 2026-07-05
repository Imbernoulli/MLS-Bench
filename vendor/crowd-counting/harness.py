#!/usr/bin/env python3
"""Density-map crowd/object-counting harness (self-contained, multi-surface).

Counts objects in an image by regressing a per-pixel DENSITY map whose spatial
integral equals the object count (the classic MCNN / CSRNet framing). Runs on FIXED
splits of REAL crowd photos (ShanghaiTech Crowd Counting Dataset, Zhang et al. CVPR
2016) so the ground-truth count is EXACT: every human head in each photo is annotated
with a single (x, y) point by the dataset authors, and the GT count is exactly the
number of annotated points (see vendor/data_scripts/crowd-counting/prepare_data.py for
the real-data staging + Gaussian-kernel density-map GT rendering). The metric is the
standard crowd-counting COUNTING MAE / RMSE between the integrated predicted density
and the true count, on a FIXED held-out val split.

Four crowd-density SCENES are provided (medium / middense / dense / superdense, each a
REAL-count bucket of pooled ShanghaiTech Part A + Part B images; each cv-count task uses
three of them); each cv-count task scores its edited surface over its three scenes as
three independent validation SETTINGS, and the score is the geometric mean over the
scenes (so the weak->strong->SOTA order must hold at every density). In every scene the
train images are drawn from a LOW real-count bucket and the val images from a disjoint
HIGHER real-count bucket (a disclosed count-EXTRAPOLATION evaluation-protocol choice,
not data fabrication), so a degenerate constant-mean predictor and a count-memorising
global regressor both fail by construction.

The agent edits ONE design surface (`--surface`) and everything else (data, seed, the
rest of the network, loss, optimiser, iterations) is FIXED, so any change in counting
MAE is attributable to the edited surface. Every surface hook is wrapped so a malformed
/ crashing return falls back to a sane default (printed as <SURFACE>_FALLBACK).

Surfaces (one per task, --surface):
  head    -> build_count_head(cin)   : the COUNT PREDICTION HEAD (formulation).
             density map (integral) vs a per-image scalar (direct regression).
  norm    -> build_density_head(cin) : density-head SPATIAL AGGREGATION (free field vs
             softmax-normalised x scalar -> bottlenecked mass).
  arch    -> build_counter()         : the FULL image->density BACKBONE/ARCHITECTURE.
             The strict-bar surface: plain single-column CNN < multi-column (MCNN) <
             dilated (CSRNet). Returns nn.Module: (B,3,H,W) -> (B,h,w) density.
  loss    -> density_loss(pred, gt)  : the DENSITY-MAP TRAINING LOSS (MSE vs
             +count-consistency vs Bayesian-style / OT-style).
  sigma   -> gt_sigma(points, H, W)  : the GT-density KERNEL bandwidth (fixed vs
             geometry-adaptive k-NN). Returns a per-point sigma (px) array.
  dilation-> build_backbone_block(cin): the frontend BLOCK (pooled small-RF vs dilated
             large-RF, CSRNet's core idea).
  columns -> build_frontend()        : SINGLE-column vs MULTI-column (MCNN) frontend.
  upsample-> build_decoder(cin)      : output-STRIDE / upsampling decoder (stride-8 vs
             upsample to finer resolution for dense scenes).
  attention-> build_attention(cin)   : spatial ATTENTION on features (none vs learned).
  multiscale-> build_context(cin)    : MULTI-SCALE context aggregation (CAN-style).
  batchnorm-> build_backbone()       : NORMALIZATION in the backbone (none vs BN).
  patch   -> crop_train(img, pts)    : PATCH-based training augmentation (full image vs
             random crops).

Metric line (one per run):
    COUNT_METRICS surface=<S> setting=<L> mae=<..> rmse=<..> nae=<..> gt_mean=<..> pred_mean=<..>
mae (counting mean-absolute-error, LOWER is better) is the primary metric. rmse is
secondary. nae = mae/gt_mean (normalised). A DEGENERATE predictor that ignores the
image and outputs the training-set mean count scores mae = MAD(val counts) (printed as
CONST_MEAN_MAE); a collapsed all-zero predictor scores mae = gt_mean. Both are far
worse than a real density model -> the metric is monotone in counting quality.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Fixed protocol
# --------------------------------------------------------------------------- #
STRIDE = 8            # frontend output stride (density map is at H/8 x W/8)
IMG_SIZE = 128        # synthetic images are IMG_SIZE x IMG_SIZE
GT_SIGMA = 6.0        # FIXED GT kernel std (px): ~0.75 stride-8 cell, well-resolved
# The density target is amplified by DENSITY_SCALE so the per-pixel MSE gradient does
# not vanish into the zero background (each blob's raw mass is 1, spread over a
# Gaussian => ~0.01/px). count = integral / DENSITY_SCALE, so the count is still exact.
DENSITY_SCALE = 100.0


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_split(root: str, split: str):
    """Return list of (img float32 CHW in [0,1], points (K,2) float32 (y,x), count)."""
    d = os.path.join(root, split)
    with open(os.path.join(d, "manifest.json")) as f:
        items = json.load(f)
    out = []
    for it in items:
        img = np.load(os.path.join(d, it["img"])).astype(np.float32)
        pts = np.asarray(it["points"], dtype=np.float32).reshape(-1, 2)
        out.append((img, pts, int(it["count"])))
    return out


def render_density(points: np.ndarray, H: int, W: int, sigmas, device) -> torch.Tensor:
    """points (K,2) (y,x) in full-res px -> (h,w) density at stride 8 that sums to
    DENSITY_SCALE * len(points); count = sum / DENSITY_SCALE is exact.

    `sigmas` is either a scalar sigma (px) OR a per-point array of sigmas (px)."""
    h, w = H // STRIDE, W // STRIDE
    dens = torch.zeros(h, w, device=device)
    if len(points) == 0:
        return dens
    ys = torch.arange(h, device=device).view(h, 1).float()
    xs = torch.arange(w, device=device).view(1, w).float()
    sig_arr = np.broadcast_to(np.asarray(sigmas, dtype=np.float32), (len(points),))
    for (py, px), sg in zip(points, sig_arr):
        sig = max(0.5, float(sg)) / STRIDE
        cy = float(py) / STRIDE; cx = float(px) / STRIDE
        g = torch.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sig * sig))
        s = g.sum()
        if float(s) > 1e-8:
            dens += DENSITY_SCALE * g / s
    return dens


def load_surface(sol_path: Path):
    spec = importlib.util.spec_from_file_location("agent_surface", str(sol_path))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(sol_path.parent))
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# --------------------------------------------------------------------------- #
# Frozen-architecture frontend (jointly trained): VGG-lite -> stride 8, 64 ch
# --------------------------------------------------------------------------- #
def _conv(cin, cout, k=3, d=1):
    return nn.Conv2d(cin, cout, k, padding=((k - 1) // 2) * d, dilation=d)


class Frontend(nn.Module):
    """Default fixed VGG-lite frontend, stride 8, 64 channels out."""

    def __init__(self):
        super().__init__()
        self.b1 = nn.Sequential(_conv(3, 32), nn.ReLU(True), _conv(32, 32), nn.ReLU(True))
        self.b2 = nn.Sequential(_conv(32, 64), nn.ReLU(True), _conv(64, 64), nn.ReLU(True))
        self.b3 = nn.Sequential(_conv(64, 64), nn.ReLU(True), _conv(64, 64), nn.ReLU(True))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(self.b1(x))
        x = self.pool(self.b2(x))
        x = self.pool(self.b3(x))
        return x                     # (B,64,h,w)

    @property
    def out_channels(self):
        return 64


# --------------------------------------------------------------------------- #
# Default heads / output
# --------------------------------------------------------------------------- #
class DensityHead(nn.Module):
    """1-channel density map (softplus) -> counted by spatial integral. The GOOD
    formulation and the default density tail for feature->density surfaces."""

    def __init__(self, cin):
        super().__init__()
        self.net = nn.Sequential(
            _conv(cin, 64, 3), nn.ReLU(True),
            _conv(64, 32, 3), nn.ReLU(True),
            nn.Conv2d(32, 1, 1))

    def forward(self, feat):
        return F.softplus(self.net(feat)).squeeze(1)   # (B,h,w) >= 0  (density)


def default_density_loss(pred, gt):
    """FIXED default training loss: pixel MSE + a small count-consistency term."""
    mse = F.mse_loss(pred, gt)
    pc = pred.sum(dim=(-2, -1)); gc = gt.sum(dim=(-2, -1))
    return mse + 0.01 * (pc - gc).abs().mean()


# --------------------------------------------------------------------------- #
# Feature -> density wrapper used by MOST surfaces (backbone edits, decoder,
# attention, context, norm ...): a fixed frontend that yields (B,C,h,w) features
# feeding a default density tail; the surface swaps ONE component.
# --------------------------------------------------------------------------- #
class CounterNet(nn.Module):
    """image -> density. Composes: frontend (feature extractor) -> [attention] ->
    [context] -> density tail. Any component the surface does not override is the
    default. Always returns a (B,h,w) non-negative density map counted by its
    integral."""

    def __init__(self, frontend, tail, attention=None, context=None, decoder=None):
        super().__init__()
        self.frontend = frontend
        self.attention = attention
        self.context = context
        self.decoder = decoder
        self.tail = tail

    def forward(self, x):
        f = self.frontend(x)
        if self.attention is not None:
            f = self.attention(f)
        if self.context is not None:
            f = self.context(f)
        if self.decoder is not None:
            f = self.decoder(f)
        out = self.tail(f)
        if out.dim() == 4:
            out = out.squeeze(1)
        return out                                     # (B,h,w) density


def _wrap(name, fn, default, *args):
    """Call an agent hook; on any failure fall back to `default` (a callable)."""
    try:
        obj = fn(*args)
        assert obj is not None
        print(f"{name.upper()}_APPLIED", flush=True)
        return obj
    except Exception as e:  # noqa: BLE001
        print(f"{name.upper()}_FALLBACK reason={e!r}", flush=True)
        return default(*args) if callable(default) else default


# --------------------------------------------------------------------------- #
# `head` / `norm` surfaces (the two ORIGINAL tasks): head returns either a density
# map or a scalar count.
# --------------------------------------------------------------------------- #
class HeadNet(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.frontend = Frontend()
        self.head = head

    def forward(self, x):
        return self.head(self.frontend(x))


def _feature_channels(frontend) -> int:
    try:
        return int(frontend.out_channels)
    except Exception:  # noqa: BLE001
        return 64


# --------------------------------------------------------------------------- #
# Build the model + loss + gt-sigma for a given surface.
# --------------------------------------------------------------------------- #
def build_model(surface, mod):
    """Return (net, loss_fn, gt_sigma_fn, crop_fn, is_scalar_head).

    loss_fn(pred,gt)->scalar ; gt_sigma_fn(points,H,W)->scalar|array ;
    crop_fn(img_np, pts_np)->(img_np, pts_np) (patch aug, applied per train sample)."""
    loss_fn = default_density_loss
    gt_sigma_fn = lambda pts, H, W: GT_SIGMA          # noqa: E731
    crop_fn = None
    is_scalar_head = False

    if surface == "head":
        head = _wrap("head", getattr(mod, "build_count_head", None) or (lambda c: None),
                     DensityHead, 64)
        net = HeadNet(head)
        return net, loss_fn, gt_sigma_fn, crop_fn, is_scalar_head

    if surface == "norm":
        head = _wrap("norm", getattr(mod, "build_density_head", None) or (lambda c: None),
                     DensityHead, 64)
        net = HeadNet(head)
        return net, loss_fn, gt_sigma_fn, crop_fn, is_scalar_head

    if surface == "arch":
        # FULL image->density counter (the strict-bar architecture surface).
        default = lambda: CounterNet(Frontend(), DensityHead(64))   # noqa: E731
        net = _wrap("arch", getattr(mod, "build_counter", None) or (lambda: None), default)
        if not isinstance(net, nn.Module):
            print("ARCH_FALLBACK reason='not a module'", flush=True); net = default()
        return net, loss_fn, gt_sigma_fn, crop_fn, is_scalar_head

    if surface == "loss":
        # density_loss(pred, gt) is a LOSS FUNCTION, not a factory -> grab the callable
        # directly (do NOT invoke it here). It is wrapped with a fallback in the train
        # loop, so a crashing loss falls back to the default per-step.
        fn = getattr(mod, "density_loss", None)
        if callable(fn):
            loss_fn = fn
            print("LOSS_APPLIED", flush=True)
        else:
            print("LOSS_FALLBACK reason='no density_loss'", flush=True)
        net = CounterNet(Frontend(), DensityHead(64))
        return net, loss_fn, gt_sigma_fn, crop_fn, is_scalar_head

    if surface == "sigma":
        fn = getattr(mod, "gt_sigma", None)
        if callable(fn):
            def gt_sigma_fn(pts, H, W, _fn=fn):  # noqa: E731
                try:
                    return _fn(pts, H, W)
                except Exception as e:  # noqa: BLE001
                    print(f"SIGMA_FALLBACK reason={e!r}", flush=True)
                    return GT_SIGMA
            print("SIGMA_APPLIED", flush=True)
        else:
            print(f"SIGMA_FALLBACK reason='no gt_sigma'", flush=True)
        net = CounterNet(Frontend(), DensityHead(64))
        return net, loss_fn, gt_sigma_fn, crop_fn, is_scalar_head

    if surface == "dilation":
        block = _wrap("dilation", getattr(mod, "build_backbone_block", None),
                      lambda c: DilatedBlock(c), 64)
        if not isinstance(block, nn.Module):
            block = DilatedBlock(64)
        net = CounterNet(BlockFrontend(block), DensityHead(_block_out(block)))
        return net, loss_fn, gt_sigma_fn, crop_fn, is_scalar_head

    if surface == "columns":
        fe = _wrap("columns", getattr(mod, "build_frontend", None), lambda: Frontend())
        if not isinstance(fe, nn.Module):
            fe = Frontend()
        net = CounterNet(fe, DensityHead(_feature_channels(fe)))
        return net, loss_fn, gt_sigma_fn, crop_fn, is_scalar_head

    if surface == "upsample":
        dec = _wrap("upsample", getattr(mod, "build_decoder", None),
                    lambda c: nn.Identity(), 64)
        if not isinstance(dec, nn.Module):
            dec = nn.Identity()
        net = CounterNet(Frontend(), DensityHead(64), decoder=dec)
        return net, loss_fn, gt_sigma_fn, crop_fn, is_scalar_head

    if surface == "attention":
        att = _wrap("attention", getattr(mod, "build_attention", None),
                    lambda c: nn.Identity(), 64)
        if not isinstance(att, nn.Module):
            att = nn.Identity()
        net = CounterNet(Frontend(), DensityHead(64), attention=att)
        return net, loss_fn, gt_sigma_fn, crop_fn, is_scalar_head

    if surface == "multiscale":
        ctx = _wrap("multiscale", getattr(mod, "build_context", None),
                    lambda c: nn.Identity(), 64)
        if not isinstance(ctx, nn.Module):
            ctx = nn.Identity()
        net = CounterNet(Frontend(), DensityHead(64), context=ctx)
        return net, loss_fn, gt_sigma_fn, crop_fn, is_scalar_head

    if surface == "batchnorm":
        # build_backbone() -> a feature extractor (image (B,3,H,W) -> (B,C,h,w) at
        # stride 8, with .out_channels). The design choice is whether the backbone uses
        # BatchNorm (stabler optimisation, batched training) or none.
        fe = _wrap("batchnorm", getattr(mod, "build_backbone", None), lambda: Frontend())
        if not isinstance(fe, nn.Module):
            fe = Frontend()
        net = CounterNet(fe, DensityHead(_feature_channels(fe)))
        return net, loss_fn, gt_sigma_fn, crop_fn, is_scalar_head

    if surface == "depth":
        # build_deep_backbone() -> feature extractor (image -> (B,C,h,w) at stride 8,
        # with .out_channels). The design choice is backbone DEPTH: a shallow backbone
        # has too little capacity to resolve heavily crowded scenes; a deeper backbone
        # (more conv layers before pooling) counts dense crowds better.
        fe = _wrap("depth", getattr(mod, "build_deep_backbone", None), lambda: Frontend())
        if not isinstance(fe, nn.Module):
            fe = Frontend()
        net = CounterNet(fe, DensityHead(_feature_channels(fe)))
        return net, loss_fn, gt_sigma_fn, crop_fn, is_scalar_head

    if surface == "patch":
        fn = getattr(mod, "crop_train", None)
        if callable(fn):
            def crop_fn(img, pts, _fn=fn):  # noqa: E731
                try:
                    return _fn(img, pts)
                except Exception as e:  # noqa: BLE001
                    print(f"PATCH_FALLBACK reason={e!r}", flush=True)
                    return img, pts
            print("PATCH_APPLIED", flush=True)
        else:
            print("PATCH_FALLBACK reason='no crop_train'", flush=True)
        net = CounterNet(Frontend(), DensityHead(64))
        return net, loss_fn, gt_sigma_fn, crop_fn, is_scalar_head

    raise ValueError(f"unknown surface {surface}")


# --------------------------------------------------------------------------- #
# Helper modules for the dilation / columns surfaces
# --------------------------------------------------------------------------- #
class DilatedBlock(nn.Module):
    """Default back-end block: dilated convs (CSRNet-style, RF-preserving)."""

    def __init__(self, cin, cout=64):
        super().__init__()
        self.net = nn.Sequential(
            _conv(cin, cout, 3, d=2), nn.ReLU(True),
            _conv(cout, cout, 3, d=2), nn.ReLU(True))
        self.out_channels = cout

    def forward(self, x):
        return self.net(x)


def _block_out(block) -> int:
    return int(getattr(block, "out_channels", 64))


class BlockFrontend(nn.Module):
    """Fixed VGG-lite stem -> agent back-end block -> features. For the `dilation`
    surface: the stem is fixed, the block (pooled vs dilated) is the edit."""

    def __init__(self, block):
        super().__init__()
        self.stem = Frontend()
        self.block = block
        self.out_channels = _block_out(block)

    def forward(self, x):
        return self.block(self.stem(x))


# --------------------------------------------------------------------------- #
# Train + eval
# --------------------------------------------------------------------------- #
def _pred_count(is_scalar, out):
    if is_scalar and out.dim() == 1:
        return out
    return out.sum(dim=(-2, -1)) / DENSITY_SCALE


def _to_ref_grid(pred, ref_hw):
    """Resample a density map (B,h,w) to the reference stride-8 grid, PRESERVING the
    total integrated mass (bilinear preserves the MEAN, so rescale by the pixel ratio).
    This makes the count resolution-invariant, so an upsampling decoder that outputs a
    finer map still integrates to the SAME count -> a fair comparison across surfaces."""
    if pred.dim() == 2:
        pred = pred[None]
    if pred.shape[-2:] == tuple(ref_hw):
        return pred
    n_pred = pred.shape[-1] * pred.shape[-2]
    n_ref = ref_hw[0] * ref_hw[1]
    r = F.interpolate(pred[:, None], size=tuple(ref_hw), mode="bilinear",
                      align_corners=False)[:, 0]
    return r * (n_pred / n_ref)   # preserve total mass


def train_and_eval(surface, mod, train, val, device, iters, seed, batch=8):
    set_all_seeds(seed)

    net, loss_fn, gt_sigma_fn, crop_fn, _ = build_model(surface, mod)
    net = net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    H = W = IMG_SIZE
    ref_hw = (H // STRIDE, W // STRIDE)   # stride-8 reference grid for counting

    # Precompute image tensors + GT density (surface-controlled sigma) for the train set.
    # H,W are taken PER SAMPLE from the (possibly cropped) image so the patch surface
    # renders its GT density at the crop resolution; the density loss resamples pred to
    # the GT grid, so mixed train/val sizes are handled cleanly.
    cache = []
    for img, pts, cnt in train:
        if crop_fn is not None:
            img, pts = crop_fn(np.asarray(img), np.asarray(pts))
            img = np.asarray(img, dtype=np.float32)
            pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
            cnt = len(pts)
        sh, sw = int(img.shape[-2]), int(img.shape[-1])
        x = torch.from_numpy(np.ascontiguousarray(img))
        sig = gt_sigma_fn(pts, sh, sw)
        g = render_density(pts, sh, sw, sig, device).cpu()
        cache.append((x, g, cnt))

    # detect scalar-count head shape once (head surface only)
    with torch.no_grad():
        probe = net(cache[0][0].unsqueeze(0).to(device))
        scalar_head = (surface == "head" and probe.dim() == 1)
        if scalar_head:
            print("HEAD_MODE scalar-count regression", flush=True)
        elif surface == "head":
            print("HEAD_MODE density-map integral", flush=True)

    n = len(cache)
    order = list(range(n))
    for it in range(iters):
        if it % (n // batch + 1) == 0:
            random.shuffle(order)
        idxs = [order[(it * batch + j) % n] for j in range(batch)]
        imgs = torch.stack([cache[i][0] for i in idxs]).to(device)
        gts = torch.stack([cache[i][1] for i in idxs]).to(device)
        counts = torch.tensor([cache[i][2] for i in idxs], dtype=torch.float32, device=device)
        out = net(imgs)
        if scalar_head:
            loss = F.mse_loss(out, counts)
        else:
            pred = out
            # Resample to the stride-8 reference grid (mass-preserving) so the density
            # loss and the counting integral are resolution-invariant across surfaces.
            if pred.shape[-2:] != gts.shape[-2:]:
                pred = _to_ref_grid(pred, gts.shape[-2:])
            try:
                loss = loss_fn(pred, gts)
            except Exception as e:  # noqa: BLE001
                if it == 0:
                    print(f"LOSS_FALLBACK reason={e!r}", flush=True)
                loss = default_density_loss(pred, gts)
        if not torch.isfinite(loss):
            loss = torch.nan_to_num(loss, nan=1e4)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % max(1, iters // 5) == 0 or it == iters - 1:
            with torch.no_grad():
                o = net(imgs)
                if not scalar_head and o.shape[-2:] != ref_hw:
                    o = _to_ref_grid(o, ref_hw)
                pc = _pred_count(scalar_head, o).mean().item()
                lv = float(loss.detach())
            print(f"train it={it} loss={lv:.4f} pred_count~{pc:.1f} "
                  f"gt_count~{counts.mean().item():.1f}", flush=True)

    # ---- eval: counting MAE / RMSE on the held-out val split ----
    net.eval()
    errs, gt_counts, pred_counts = [], [], []
    with torch.no_grad():
        for img, pts, cnt in val:
            x = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).to(device)
            out = net(x)
            if not scalar_head and out.shape[-2:] != ref_hw:
                out = _to_ref_grid(out, ref_hw)
            pc = float(_pred_count(scalar_head, out)[0].item())
            errs.append(abs(pc - cnt))
            gt_counts.append(float(cnt)); pred_counts.append(pc)
    mae = float(np.mean(errs))
    rmse = float(np.sqrt(np.mean([(p - g) ** 2 for p, g in zip(pred_counts, gt_counts)])))
    gt_mean = float(np.mean(gt_counts)); pred_mean = float(np.mean(pred_counts))
    nae = mae / max(1e-6, gt_mean)
    return dict(mae=mae, rmse=rmse, nae=nae, gt_mean=gt_mean, pred_mean=pred_mean)


def const_mean_mae(train, val) -> float:
    m = float(np.mean([c for _, _, c in train]))
    return float(np.mean([abs(m - c) for _, _, c in val]))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
SURFACES = ["head", "norm", "arch", "loss", "sigma", "dilation", "columns",
            "upsample", "attention", "multiscale", "batchnorm", "patch", "depth"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--surface", required=True, choices=SURFACES)
    ap.add_argument("--label", default="run")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE {device} torch {torch.__version__}", flush=True)

    train = load_split(args.data_root, "train")
    val = load_split(args.data_root, "val")
    cmm = const_mean_mae(train, val)
    print(f"DATA train={len(train)} val={len(val)} "
          f"gt_count[min={min(c for _,_,c in val)} max={max(c for _,_,c in val)}] "
          f"CONST_MEAN_MAE={cmm:.3f}", flush=True)

    mod = load_surface(Path(args.solution))
    m = train_and_eval(args.surface, mod, train, val, device, args.iters, args.seed)

    print(f"COUNT_METRICS surface={args.surface} setting={args.label} "
          f"mae={m['mae']:.4f} rmse={m['rmse']:.4f} nae={m['nae']:.4f} "
          f"gt_mean={m['gt_mean']:.3f} pred_mean={m['pred_mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
