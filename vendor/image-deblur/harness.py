#!/usr/bin/env python3
"""Image-deblurring harness (self-contained).

Single-image MOTION DEBLURRING: restore a sharp image from a blurred one. This is a
genuinely new restoration direction, DISTINCT from super-resolution / denoising
(basicsr-sr), colorization (image-colorization), inpainting (image-inpainting),
matting (image-matting) -- deblurring removes a *motion-blur* degradation. The
reference architectures are the multi-scale coarse-to-fine deblur nets DeepDeblur
(Nah et al., CVPR 2017), SRN-DeblurNet (Tao et al., CVPR 2018) and MPRNet
(Zamir et al., CVPR 2021).

A compact residual encoder-decoder deblur net is trained a few hundred steps on a
TINY fixed set of blurry->sharp patch pairs and evaluated by DEBLUR PSNR on a
held-out split. Data are REAL photographs from the GoPro Large-Scale Blur Dataset
(Nah, Kim & Lee, CVPR 2017): the blurry frame is a genuine long-exposure/motion-
averaged capture and the sharp frame a genuine high-speed capture of the SAME instant
(prepared offline by prepare_data.py, which tiles full-resolution GoPro frames into
64x64 patches -- see that file's docstring for the real-data provenance and the
tercile severity-bucketing methodology). The train and val splits use DISJOINT GoPro
SCENES (GoPro's own train/test split), so held-out PSNR measures genuine cross-scene
generalisation.

The agent edits ONE design surface (chosen by --surface); everything else (data,
backbone width/depth, optimiser, iterations, seed, eval split, blur kernels, the
metric) is FIXED, so any change in the score is attributable to the edited surface.

Surfaces (one per task):
  residual    -> get_residual_config() : whether the net predicts a GLOBAL RESIDUAL
                 (sharp = blurry + net(blurry), the strong answer -- the net only has
                 to model the high-freq deblur correction) vs the FULL image directly
                 (bad: harder optimisation, blurrier output). (DeepDeblur/SRN/MPRNet
                 all use global residual / long skip.)
  loss        -> get_loss_config() : the reconstruction loss -- 'l2' (MSE, the naive
                 baseline; over-smooths, penalises edges softly), 'charbonnier'
                 (robust L1-like sqrt(e^2+eps^2), sharper), or charbonnier + an EDGE
                 (image-gradient) term that explicitly rewards restoring high-freq
                 detail. Also the TARGET the net is matched to (true sharp GT vs an
                 over-smoothed low-pass GT). (bad: over-smoothed target -> smooth.)
  multiscale  -> get_scale_config() : single-scale (deblur only at full res) vs a
                 MULTI-SCALE coarse-to-fine pyramid (2-3 levels, SHARED weights, each
                 level refines the upsampled coarser output -- the SRN recurrence).
                 (bad: 1 scale ; good: 3 scales coarse-to-fine.)
  edge        -> get_loss_config() : weight of the EDGE (image-gradient) loss term.
                 0 = plain reconstruction (edges under-restored); a positive weight
                 explicitly rewards restoring high-frequency detail (sharper, higher
                 PSNR). (LapSRN / MPRNet edge / gradient loss.)
  recurrence  -> get_recurrence_config() : number of WITHIN-SCALE full-res refinement
                 passes with SHARED weights (the SRN recurrence WITHOUT the pyramid).
                 (bad: 1 pass ; good: 3 passes progressively remove larger blur.)
  ARCHITECTURE surfaces -> get_arch_config() : each toggles ONE construction choice of
                 the parameterised deblur backbone, everything else fixed at the strong
                 reference (ArchDeblurNet / default_arch()):
    depth       n_resblocks : ResBlocks per stage (shallow -> weak on heavy blur).
    width       width       : base channel width (narrow -> weak, under-fits).
    attention   attention   : channel attention (SE/CAB, MPRNet) on vs off.
    dilation    dilation    : bottleneck dilation = receptive field (narrow -> weak on
                              large blur ; wide -> strong).
    upsample / norm / activation / skip : additional arch levers supported by the harness
                 (upsample method, BatchNorm vs none, ReLU vs LeakyReLU/GELU, U-Net skip).
                 NOTE: on this tiny synthetic scale these four did NOT produce a monotone
                 strong>weak order across 3 severities (they invert / flatten), so no task
                 ships them; they remain as harness options for reference / larger scales.

The SHIPPED deblur tasks (each = one surface, scored over 3 REAL motion-blur severity
settings where the lever is monotone -- see vendor/data_scripts/image-deblur/
prepare_data.py for the real-data tercile construction; every legacy band code below
(rs/rm/rl, ms/mm/ml, hs/hm/hl, em/el) is now an ALIAS onto the single real small/
medium/large ladder, since a real photograph pair offers one genuine severity axis,
not five independently-tunable synthetic bands): residual (rs/rm/rl), loss/
target-smooth (small/medium/large), multiscale (ms/mm/ml), depth (ms/mm/ml), width
(ms/mm/ml), attention (ms/mm/ml), dilation (hs/hm/hl), recurrence (ms/mm/ml),
loss-kind l2-vs-charbonnier (em/el/medium), edge-loss weight (rs/rm/rl).

Metric line (one per run):
  DEBLUR_METRICS surface=<S> setting=<L> psnr=<..> psnr_gain=<..> blurry_psnr=<..> \
      ssim=<..> mse=<..>
`psnr` is the DEBLUR PSNR of the RESTORED output vs the sharp GT (dB, HIGHER better)
and is the PRIMARY metric. `blurry_psnr` is the PSNR of the *blurry input* vs the
sharp GT -- the identity ("do-nothing") floor. `psnr_gain = psnr - blurry_psnr` is
reported so it is explicit that the restored output must BEAT passing the input
through: a degenerate net that copies its input scores psnr==blurry_psnr (gain 0),
and a constant/gray output scores far BELOW the blurry floor. `ssim` and `mse` are
diagnostics.

Every hook is wrapped so a malformed / crashing return falls back to a sane default
(the strong reference config) rather than aborting the run.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# FIXED protocol
# --------------------------------------------------------------------------- #
IMG = 64                 # patch size (upsampled CIFAR)
BASE = 32                # deblur net base width (FIXED)
BS = 32                  # batch size (FIXED)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------- #
# Data: blurry->sharp pairs. Prepared offline (prepare_data.py) into npz with
#   blur (N,3,H,W) in [0,1]   sharp (N,3,H,W) in [0,1]
# train / val splits use DISJOINT sharp patches AND disjoint kernels (fixed).
# --------------------------------------------------------------------------- #
# Every legacy synthetic "severity band" code (rs/rm/rl, ms/mm/ml, hs/hm/hl, es/em/el)
# is an ALIAS onto the ONE real small/medium/large tercile ladder built by
# prepare_data.py from REAL GoPro blurry/sharp photograph pairs -- a real photograph
# pair offers one genuine measured severity axis, not five independently-tunable
# synthetic bands. Kept in sync with _ALIAS in vendor/data_scripts/image-deblur/
# prepare_data.py so every --blur-type a task script passes resolves to real data on
# disk without needing prepare_data.py to also materialise 15 duplicate directories.
_BLUR_TYPE_ALIAS = {
    "small": "small", "medium": "medium", "large": "large",
    "rs": "small", "rm": "medium", "rl": "large",
    "ms": "small", "mm": "medium", "ml": "large",
    "es": "small", "em": "small", "el": "large",
    "hs": "small", "hm": "medium", "hl": "large",
}


def load_split(root: str, split: str, blur_type: str = "medium"):
    """Load a blurry->sharp split for ONE motion-blur severity.

    Data live under <root>/<setting>/<split>.npz where <setting> is one of the THREE
    real settings (small|medium|large -- see prepare_data.py for the REAL GoPro-derived
    tercile construction). `blur_type` may be any of the legacy band codes (rs/rm/rl,
    ms/mm/ml, hs/hm/hl, es/em/el); it is resolved to its real setting via
    `_BLUR_TYPE_ALIAS` before loading. For backward compatibility, if the per-severity
    subdir is absent we fall back to <root>/<split>.npz.
    """
    setting = _BLUR_TYPE_ALIAS.get(blur_type, blur_type)
    sub = os.path.join(root, setting, f"{split}.npz")
    path = sub if os.path.exists(sub) else os.path.join(root, f"{split}.npz")
    arr = np.load(path)
    blur = torch.from_numpy(arr["blur"].astype(np.float32))
    sharp = torch.from_numpy(arr["sharp"].astype(np.float32))
    return blur, sharp


# --------------------------------------------------------------------------- #
# Compact residual encoder-decoder deblur net (FIXED width/depth). A single
# forward maps a blurry image to a deblurred one at one scale. Optional GLOBAL
# RESIDUAL is applied OUTSIDE (in the training/eval loop) so the residual surface
# controls it cleanly. This ORIGINAL net is used UNCHANGED by the residual / loss /
# multiscale surfaces, so their validated anchors stay exactly valid. The new
# ARCHITECTURE surfaces use the parameterised ArchDeblurNet below instead.
# --------------------------------------------------------------------------- #
class _ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        y = F.relu(self.c1(x), inplace=True)
        y = self.c2(y)
        return x + y


class DeblurNet(nn.Module):
    """Encoder-decoder with ResBlocks. Takes a 3-ch (or 6-ch, for coarse-to-fine
    with an upsampled prior) image, returns a 3-ch image at the same resolution."""

    def __init__(self, in_ch: int = 3, base: int = BASE):
        super().__init__()
        self.inc = nn.Conv2d(in_ch, base, 3, padding=1)
        self.enc1 = _ResBlock(base)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc2 = _ResBlock(base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.mid = nn.Sequential(_ResBlock(base * 4), _ResBlock(base * 4))
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.dec2 = _ResBlock(base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec1 = _ResBlock(base)
        self.outc = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x):
        h0 = F.relu(self.inc(x), inplace=True)
        e1 = self.enc1(h0)
        e2 = self.enc2(F.relu(self.down1(e1), inplace=True))
        m = self.mid(F.relu(self.down2(e2), inplace=True))
        d2 = self.dec2(F.relu(self.up2(m), inplace=True) + e2)
        d1 = self.dec1(F.relu(self.up1(d2), inplace=True) + e1)
        return self.outc(d1)


# --------------------------------------------------------------------------- #
# PARAMETERISED architecture net for the new ARCHITECTURE surfaces (depth /
# attention / upsample / norm / activation / skip / dilation). Every construction
# choice comes from an `arch_cfg` dict so each surface toggles exactly ONE lever
# while everything else is FIXED at the strong reference. The DEFAULT arch_cfg is
# the STRONG reference config (what a good modern deblur net uses): no BatchNorm,
# LeakyReLU, resize-conv upsampling, additive long skips, channel attention on,
# moderate depth, dilation for a wide receptive field. Only the architecture
# surfaces instantiate this net; residual/loss/multiscale keep DeblurNet above.
# --------------------------------------------------------------------------- #
def default_arch() -> dict:
    return dict(
        n_resblocks=2,     # ResBlocks per enc/dec stage (mid uses 2x this)  [depth surface]
        width=32,          # base channel width of the backbone                [width surface]
        attention=True,    # channel attention (SE/CAB) inside each ResBlock  [attention surface]
        upsample="resize", # 'resize' (bilinear+conv, artefact-free, strong) |
                           #  'deconv' (ConvTranspose, checkerboard) | 'nearest'  [upsample surface]
        norm="none",       # 'none' (strong; BN hurts restoration) | 'batch'      [norm surface]
        act="leaky",       # 'leaky' | 'gelu' (strong) | 'relu' (dead units, weak) [activation surface]
        skip=True,         # additive encoder->decoder long skip (U-Net)          [skip/fusion surface]
        dilation=2,        # dilation of the mid (bottleneck) convs -> receptive field [dilation surface]
    )


def _act(name: str):
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    return nn.LeakyReLU(0.2, inplace=True)      # default / strong


def _norm(name: str, c: int):
    if str(name).lower() == "batch":
        return nn.BatchNorm2d(c)
    return nn.Identity()                        # 'none' -> strong for restoration


class _ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention (the CAB used by MPRNet, Zamir et al.
    CVPR 2021): global-avg-pool -> 2-layer MLP -> per-channel gate. Off -> Identity."""

    def __init__(self, c: int, reduction: int = 8):
        super().__init__()
        r = max(1, c // reduction)
        self.fc1 = nn.Conv2d(c, r, 1)
        self.fc2 = nn.Conv2d(r, c, 1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class _ArchResBlock(nn.Module):
    """Residual block, parameterised by the arch surfaces (norm / activation /
    channel-attention). With the DEFAULT arch_cfg (attention on, no norm, leaky act)
    this is an MPRNet-style CAB; the arch surfaces toggle one piece at a time. The
    residual branch is scaled by 0.1 (EDSR residual scaling, Lim et al. CVPR 2017) so that
    DEEP unnormalised stacks stay stable at lr=1e-3 instead of diverging."""

    def __init__(self, c, arch: dict):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.n1 = _norm(arch.get("norm", "none"), c)
        self.n2 = _norm(arch.get("norm", "none"), c)
        self.act = _act(arch.get("act", "leaky"))
        self.ca = _ChannelAttention(c) if arch.get("attention", True) else nn.Identity()
        self.res_scale = 0.1

    def forward(self, x):
        y = self.act(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        y = self.ca(y)
        return x + self.res_scale * y


class _ResizeConv(nn.Module):
    def __init__(self, in_c, out_c, interp):
        super().__init__()
        self.interp = interp
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1)

    def forward(self, x):
        kw = dict(scale_factor=2, mode=self.interp)
        if self.interp != "nearest":
            kw["align_corners"] = False
        x = F.interpolate(x, **kw)
        return self.conv(x)


def _upsample_block(in_c: int, out_c: int, mode: str):
    """Upsampling from a coarse to a finer feature map. 'deconv' = ConvTranspose
    (checkerboard artefacts, the weak choice); 'resize'/'nearest' = interpolate + 3x3
    conv (artefact-free, the strong choice used by modern restoration nets)."""
    mode = str(mode).lower()
    if mode == "deconv":
        return nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1)
    interp = "nearest" if mode == "nearest" else "bilinear"
    return _ResizeConv(in_c, out_c, interp)


class ArchDeblurNet(nn.Module):
    """Encoder-decoder with PARAMETERISED ResBlocks -- used ONLY by the architecture
    surfaces. Takes a 3-ch image, returns a 3-ch image at the same resolution. Every
    architecture choice comes from `arch`; with the default arch it is the strong
    reference net. A dilated bottleneck pair widens the receptive field (dilation surface)."""

    def __init__(self, in_ch: int, base: int, arch: dict):
        super().__init__()
        arch = {**default_arch(), **(arch or {})}
        self.arch = arch
        n = max(1, int(arch.get("n_resblocks", 2)))
        base = int(max(8, min(64, int(arch.get("width", base)))))   # width surface
        self.use_skip = bool(arch.get("skip", True))
        up = arch.get("upsample", "resize")
        self.act = _act(arch.get("act", "leaky"))
        d = max(1, min(4, int(arch.get("dilation", 1))))

        self.inc = nn.Conv2d(in_ch, base, 3, padding=1)
        self.enc1 = nn.Sequential(*[_ArchResBlock(base, arch) for _ in range(n)])
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(*[_ArchResBlock(base * 2, arch) for _ in range(n)])
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.mid = nn.Sequential(*[_ArchResBlock(base * 4, arch) for _ in range(n + 1)])
        # dilated bottleneck pair -> receptive field (dilation surface)
        self.middil = nn.Sequential(
            nn.Conv2d(base * 4, base * 4, 3, padding=d, dilation=d),
            _act(arch.get("act", "leaky")),
            nn.Conv2d(base * 4, base * 4, 3, padding=d, dilation=d),
        )
        self.up2 = _upsample_block(base * 4, base * 2, up)
        self.dec2 = nn.Sequential(*[_ArchResBlock(base * 2, arch) for _ in range(n)])
        self.up1 = _upsample_block(base * 2, base, up)
        self.dec1 = nn.Sequential(*[_ArchResBlock(base, arch) for _ in range(n)])
        self.outc = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x):
        h0 = self.act(self.inc(x))
        e1 = self.enc1(h0)
        e2 = self.enc2(self.act(self.down1(e1)))
        m = self.mid(self.act(self.down2(e2)))
        m = m + 0.1 * self.middil(m)          # scaled dilated residual (stability)
        d2i = self.act(self.up2(m))
        d2 = self.dec2(d2i + e2 if self.use_skip else d2i)
        d1i = self.act(self.up1(d2))
        d1 = self.dec1(d1i + e1 if self.use_skip else d1i)
        return self.outc(d1)


# --------------------------------------------------------------------------- #
# Metrics: PSNR + a simple SSIM, on images in [0,1].
# --------------------------------------------------------------------------- #
def psnr_batch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = pred.clamp(0, 1).float()
    gt = gt.clamp(0, 1).float()
    mse = ((pred - gt) ** 2).reshape(pred.shape[0], -1).mean(1)
    mse = mse.clamp_min(1e-10)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return float(psnr.mean())


def _gaussian_window(ch, ks=11, sigma=1.5, device="cpu"):
    coords = torch.arange(ks, dtype=torch.float32, device=device) - (ks - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    w = (g[:, None] * g[None, :])[None, None]
    return w.expand(ch, 1, ks, ks).contiguous()


def ssim_batch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = pred.clamp(0, 1).float()
    gt = gt.clamp(0, 1).float()
    ch = pred.shape[1]
    w = _gaussian_window(ch, device=pred.device)
    pad = w.shape[-1] // 2
    mu1 = F.conv2d(pred, w, padding=pad, groups=ch)
    mu2 = F.conv2d(gt, w, padding=pad, groups=ch)
    mu1_2, mu2_2, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(pred * pred, w, padding=pad, groups=ch) - mu1_2
    s2 = F.conv2d(gt * gt, w, padding=pad, groups=ch) - mu2_2
    s12 = F.conv2d(pred * gt, w, padding=pad, groups=ch) - mu12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_2 + mu2_2 + C1) * (s1 + s2 + C2))
    return float(ssim_map.mean())


# --------------------------------------------------------------------------- #
# Loss builders. The `loss` surface picks among these.
# --------------------------------------------------------------------------- #
def _charbonnier(pred, gt, eps=1e-3):
    return torch.sqrt((pred - gt) ** 2 + eps ** 2).mean()


def _grad(x):
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    return gx, gy


def _edge_loss(pred, gt, eps=1e-3):
    px, py = _grad(pred)
    gx, gy = _grad(gt)
    return (torch.sqrt((px - gx) ** 2 + eps ** 2).mean()
            + torch.sqrt((py - gy) ** 2 + eps ** 2).mean())


def _gaussian_blur(x, sigma):
    """Low-pass (Gaussian) filter an image batch -- used to build an OVER-SMOOTHED target."""
    if sigma <= 0:
        return x
    ks = int(2 * round(3 * sigma) + 1)
    ch = x.shape[1]
    w = _gaussian_window(ch, ks=ks, sigma=sigma, device=x.device)
    return F.conv2d(x, w, padding=ks // 2, groups=ch)


def build_loss(cfg: dict):
    """cfg keys:
      kind        : 'l2' (MSE) or 'charbonnier' (robust L1-like)   [reconstruction loss]
      edge_weight : weight of an extra image-gradient (edge) term  [0 disables]
      target_smooth: sigma of a Gaussian low-pass applied to the GT BEFORE the loss.
                     >0 makes the network optimise toward an OVER-SMOOTHED target (the
                     classic L2-conditional-mean failure mode: it deliberately blurs away
                     the high-frequency detail it should restore, so deblur PSNR drops).
                     0 = optimise toward the true sharp GT (correct -> higher PSNR).
    Returns fn(pred, gt)."""
    kind = str(cfg.get("kind", "charbonnier")).lower()
    ew = float(cfg.get("edge_weight", 0.0))
    ts = float(cfg.get("target_smooth", 0.0))

    def loss_fn(pred, gt):
        tgt = _gaussian_blur(gt, ts) if ts > 0 else gt
        if kind == "l2":
            base = F.mse_loss(pred, tgt)
        else:
            base = _charbonnier(pred, tgt)
        if ew > 0.0:
            base = base + ew * _edge_loss(pred, tgt)
        return base

    return loss_fn


# --------------------------------------------------------------------------- #
# Defaults (fallbacks) = the STRONG reference config for each surface.
# --------------------------------------------------------------------------- #
def default_residual():
    return dict(global_residual=True)


def default_loss():
    # STRONG reference = optimise toward the TRUE SHARP target (no over-smoothing).
    return dict(kind="charbonnier", edge_weight=0.1, target_smooth=0.0)


def default_scale():
    return dict(scales=3)


def default_recurrence():
    # STRONG reference = 3 within-scale full-res refinement passes (SRN recurrence).
    return dict(n_recurrence=3)


# ---- architecture surfaces: each toggles ONE key of the arch_cfg (default_arch()) ----
# The hook name and the key it controls (everything else stays at default_arch()):
#   depth       -> get_arch_config()['n_resblocks']   (int 1..4)
#   attention   -> get_arch_config()['attention']     (bool)
#   upsample    -> get_arch_config()['upsample']      ('resize'|'deconv'|'nearest')
#   norm        -> get_arch_config()['norm']          ('none'|'batch')
#   activation  -> get_arch_config()['act']           ('leaky'|'gelu'|'relu')
#   skip        -> get_arch_config()['skip']          (bool)
#   dilation    -> get_arch_config()['dilation']      (int 1..4)
ARCH_SURFACES = ("depth", "width", "attention", "upsample", "norm", "activation", "skip", "dilation")

# Which arch key each architecture surface owns + the validator for its value.
_ARCH_KEY = {
    "depth": "n_resblocks", "width": "width", "attention": "attention", "upsample": "upsample",
    "norm": "norm", "activation": "act", "skip": "skip", "dilation": "dilation",
}


def _sanitize_arch(cfg: dict) -> dict:
    """Clamp/validate an arch_cfg to the strong-reference defaults + safe ranges."""
    out = default_arch()
    if not isinstance(cfg, dict):
        return out
    if "n_resblocks" in cfg:
        out["n_resblocks"] = int(max(1, min(4, int(cfg["n_resblocks"]))))
    if "attention" in cfg:
        out["attention"] = bool(cfg["attention"])
    if "upsample" in cfg and str(cfg["upsample"]).lower() in ("resize", "deconv", "nearest"):
        out["upsample"] = str(cfg["upsample"]).lower()
    if "norm" in cfg and str(cfg["norm"]).lower() in ("none", "batch"):
        out["norm"] = str(cfg["norm"]).lower()
    if "act" in cfg and str(cfg["act"]).lower() in ("leaky", "gelu", "relu"):
        out["act"] = str(cfg["act"]).lower()
    if "skip" in cfg:
        out["skip"] = bool(cfg["skip"])
    if "dilation" in cfg:
        out["dilation"] = int(max(1, min(4, int(cfg["dilation"]))))
    if "width" in cfg:
        out["width"] = int(max(8, min(64, int(cfg["width"]))))
    return out



# --------------------------------------------------------------------------- #
# Load agent surface
# --------------------------------------------------------------------------- #
def load_surface(sol_path: Path):
    spec = importlib.util.spec_from_file_location("agent_surface", str(sol_path))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(sol_path.parent))
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# --------------------------------------------------------------------------- #
# Forward: apply the deblur net at a single scale, honouring global residual.
# --------------------------------------------------------------------------- #
def _forward_single(model, blur, global_residual):
    out = model(blur)
    if global_residual:
        out = blur + out
    return out


def _forward_multiscale(model, blur, global_residual, scales):
    """Coarse-to-fine SCALE-RECURRENT deblurring (SRN, Tao et al. CVPR 2018; cf.
    DeepDeblur, Nah et al. CVPR 2017). ``model`` is the SHARED-weight 6-ch-in deblur net:
    at every scale it takes [ blurry_at_scale , upsampled_coarser_estimate ] concatenated
    on the channel axis and outputs the deblurred image at that scale.

    Coarse -> fine: at the coarsest scale the blur spans fewer pixels and is easy to
    invert; the prior channel there is just the (coarse) blurry image itself. Each FINER
    level still sees the PRISTINE blurry image at its OWN resolution (so full-resolution
    high-frequency detail is NEVER thrown away) PLUS the upsampled coarser deblurred
    estimate as guidance, and refines. The global-residual skip is anchored on the pristine
    blurry image at that scale (sharp = blurry + net([blurry, prior])). Because the finest
    level keeps the full-res pristine blur and merely GAINS the coarse guidance channel, a
    3-scale pyramid restores large blur better than a single full-res pass while never
    losing detail. With scales=1 the prior channel equals the blurry image, so this reduces
    to a single full-res residual pass on a duplicated input."""
    H, W = blur.shape[-2:]
    est = None            # running deblurred estimate carried up from the coarser scale
    for s in reversed(range(scales)):
        factor = 2 ** s
        h, w = H // factor, W // factor
        b_s = F.interpolate(blur, size=(h, w), mode="bilinear", align_corners=False)
        # prior guidance channel: upsampled coarser estimate, or the blurry image at the
        # coarsest scale. The blurry image itself is ALWAYS kept as the detail-carrying input.
        prior = b_s if est is None else \
            F.interpolate(est, size=(h, w), mode="bilinear", align_corners=False)
        out = model(torch.cat([b_s, prior], dim=1))     # 6-ch in: [pristine blur, prior]
        base = b_s if global_residual else torch.zeros_like(b_s)
        est = base + out
    # Scale-recurrent refinement (SRN): with >1 scale, run ONE extra full-res recurrence
    # using the current estimate as the prior, so the net gets a second, better-initialised
    # pass at the hard full-resolution deblur. This is the recurrence that makes the
    # coarse-to-fine pyramid clearly beat a single full-res pass on heavy blur (scales=1
    # skips it, so it is a strict extra refinement that grows the multi>single margin).
    if scales > 1:
        base = blur if global_residual else torch.zeros_like(blur)
        est = base + model(torch.cat([blur, est], dim=1))
    return est


def _forward_recurrent(model, blur, global_residual, n_recurrence):
    """WITHIN-SCALE recurrent refinement at full resolution (the SRN recurrence, Tao et al.
    CVPR 2018, applied WITHOUT the multi-scale pyramid). ``model`` is the SHARED-weight
    6-ch-in net: at each pass it takes [ pristine_blur , current_estimate ] and outputs a
    refined deblurred image. The first pass seeds the estimate from the blurry image; each
    later pass RE-reads the pristine blur plus its own previous estimate and sharpens
    further. With n_recurrence=1 this is a single full-res pass (weak on heavy blur); more
    passes progressively remove larger blur (strong), at IDENTICAL parameter count (weights
    shared across passes)."""
    est = blur
    for _ in range(max(1, n_recurrence)):
        base = blur if global_residual else torch.zeros_like(blur)
        est = base + model(torch.cat([blur, est], dim=1))
    return est


# --------------------------------------------------------------------------- #
# Train + eval
# --------------------------------------------------------------------------- #
def run(surface, mod, data_root, device, iters, seed, blur_type="medium"):
    set_all_seeds(seed)
    b_tr, s_tr = load_split(data_root, "train", blur_type)
    b_va, s_va = load_split(data_root, "val", blur_type)
    b_tr, s_tr = b_tr.to(device), s_tr.to(device)
    b_va, s_va = b_va.to(device), s_va.to(device)
    print(f"DATA blur={blur_type} train={b_tr.shape[0]} val={b_va.shape[0]} "
          f"img={tuple(b_tr.shape[-2:])}", flush=True)

    # ---- resolve surface configs (all default to the strong reference) ----
    res_cfg = default_residual()
    loss_cfg = default_loss()
    scale_cfg = default_scale()
    arch_cfg = default_arch()
    rec_cfg = default_recurrence()

    if surface == "residual":
        try:
            cand = mod.get_residual_config()
            assert isinstance(cand, dict) and "global_residual" in cand
            res_cfg = {**default_residual(), **cand}
            print(f"RESIDUAL_APPLIED {res_cfg}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"RESIDUAL_FALLBACK reason={e!r}", flush=True)
    else:
        print(f"RESIDUAL_FIXED {res_cfg}", flush=True)

    # loss surface AND the 'edge' surface both edit get_loss_config(); loss keys are
    # merged into the strong-reference default_loss(), so 'edge' (which varies edge_weight)
    # and 'loss' (which varies target_smooth) share one clean loss builder.
    if surface in ("loss", "edge"):
        try:
            cand = mod.get_loss_config()
            assert isinstance(cand, dict) and "kind" in cand
            loss_cfg = {**default_loss(), **cand}
            # clamp target_smooth to a sane range (sigma of the GT low-pass)
            loss_cfg["target_smooth"] = float(max(0.0, min(4.0, loss_cfg.get("target_smooth", 0.0))))
            loss_cfg["edge_weight"] = float(max(0.0, min(2.0, loss_cfg.get("edge_weight", 0.0))))
            print(f"LOSS_APPLIED {loss_cfg}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"LOSS_FALLBACK reason={e!r}", flush=True)
    else:
        print(f"LOSS_FIXED {loss_cfg}", flush=True)

    if surface == "multiscale":
        try:
            cand = mod.get_scale_config()
            assert isinstance(cand, dict) and "scales" in cand
            sc = int(cand["scales"])
            assert 1 <= sc <= 4
            scale_cfg = dict(scales=sc)
            print(f"SCALE_APPLIED {scale_cfg}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"SCALE_FALLBACK reason={e!r}", flush=True)
    else:
        print(f"SCALE_FIXED {scale_cfg}", flush=True)

    if surface == "recurrence":
        try:
            cand = mod.get_recurrence_config()
            assert isinstance(cand, dict) and "n_recurrence" in cand
            nr = int(max(1, min(4, int(cand["n_recurrence"]))))
            rec_cfg = dict(n_recurrence=nr)
            print(f"RECUR_APPLIED {rec_cfg}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"RECUR_FALLBACK reason={e!r}", flush=True)
    elif surface in ARCH_SURFACES:
        try:
            cand = mod.get_arch_config()
            assert isinstance(cand, dict)
            arch_cfg = _sanitize_arch(cand)
            print(f"ARCH_APPLIED surface={surface} {arch_cfg}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ARCH_FALLBACK reason={e!r}", flush=True)
    else:
        print(f"ARCH_FIXED {arch_cfg}", flush=True)

    global_residual = bool(res_cfg.get("global_residual", True))
    scales = int(scale_cfg.get("scales", 3))
    n_recurrence = int(rec_cfg.get("n_recurrence", 3))
    loss_fn = build_loss(loss_cfg)

    # The multiscale surface uses the SCALE-RECURRENT 6-ch net ([blur, prior]) for BOTH
    # scales=1 and scales>1, so single- vs multi-scale is an apples-to-apples comparison at
    # identical parameter count (only the number of coarse-to-fine passes changes). The
    # recurrence surface likewise uses the 6-ch net ([blur, estimate]) run n_recurrence
    # times at full resolution. The architecture surfaces use the parameterised ArchDeblurNet
    # (3-ch). The residual/loss/edge surfaces use the plain 3-ch single-scale DeblurNet.
    multiscale_surface = (surface == "multiscale")
    recurrence_surface = (surface == "recurrence")
    arch_surface = (surface in ARCH_SURFACES)
    in_ch = 6 if (multiscale_surface or recurrence_surface) else 3

    def forward(blur):
        if multiscale_surface:
            return _forward_multiscale(model, blur, global_residual, max(scales, 1))
        if recurrence_surface:
            return _forward_recurrent(model, blur, global_residual, n_recurrence)
        return _forward_single(model, blur, global_residual)

    # ---- build model + optimiser (FIXED) ----
    if arch_surface:
        model = ArchDeblurNet(in_ch=in_ch, base=BASE, arch=arch_cfg).to(device).train()
    else:
        model = DeblurNet(in_ch=in_ch, base=BASE).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    N = b_tr.shape[0]
    for it in range(iters):
        sel = torch.randint(0, N, (BS,), device=device)
        blur = b_tr[sel]; sharp = s_tr[sel]
        out = forward(blur)
        loss = loss_fn(out, sharp)
        if not torch.isfinite(loss):
            loss = out.float().pow(2).mean() * 0.0 + 1.0
        opt.zero_grad(); loss.backward(); opt.step()
        if it % max(1, iters // 5) == 0 or it == iters - 1:
            print(f"train it={it} loss={loss.detach().item():.5f}", flush=True)

    # ---- eval on held-out val ----
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, b_va.shape[0], 64):
            preds.append(forward(b_va[i:i + 64]).clamp(0, 1).cpu())
    pred = torch.cat(preds, 0)
    sharp = s_va.cpu()
    blur = b_va.cpu()

    psnr = psnr_batch(pred, sharp)
    blurry_psnr = psnr_batch(blur, sharp)     # identity floor (do-nothing)
    ssim = ssim_batch(pred, sharp)
    mse = float(((pred.clamp(0, 1) - sharp) ** 2).mean())
    return dict(psnr=psnr, blurry_psnr=blurry_psnr, psnr_gain=psnr - blurry_psnr,
                ssim=ssim, mse=mse)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--surface", required=True,
                    choices=(["residual", "loss", "multiscale", "recurrence", "edge"]
                             + list(ARCH_SURFACES)))
    ap.add_argument("--blur-type", default="medium",
                    choices=["small", "medium", "large",   # loss / edge / arch bands
                             "rs", "rm", "rl",             # residual / mild band
                             "ms", "mm", "ml",             # multiscale / heavy band
                             "es", "em", "el",             # easy band (skip task)
                             "hs", "hm", "hl"],            # very-heavy band (recur/dilation)
                    help="which motion-blur severity setting to train+eval on")
    ap.add_argument("--label", default="run")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE {device} torch {torch.__version__}", flush=True)

    mod = load_surface(Path(args.solution))
    m = run(args.surface, mod, args.data_root, device, args.iters, args.seed,
            blur_type=args.blur_type)

    print(f"DEBLUR_METRICS surface={args.surface} setting={args.label} "
          f"psnr={m['psnr']:.4f} psnr_gain={m['psnr_gain']:.4f} "
          f"blurry_psnr={m['blurry_psnr']:.4f} ssim={m['ssim']:.4f} "
          f"mse={m['mse']:.6f}", flush=True)


if __name__ == "__main__":
    main()
