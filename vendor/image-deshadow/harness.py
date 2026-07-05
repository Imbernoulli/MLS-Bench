#!/usr/bin/env python3
"""Image shadow-removal (deshadowing) harness (self-contained, MULTI-SURFACE).

SHADOW REMOVAL: recover a clean, shadow-free image from one on which a CAST SHADOW darkens a
known region. This is a genuinely new restoration direction with a DISTINCT degradation --
a spatially-localised MULTIPLICATIVE illumination attenuation over a soft-edged region --
separate from atmospheric-scattering haze (image-dehaze), rain streaks (image-derain),
motion blur (image-deblur), low-light global under-exposure (low-light-enhance),
inpainting (image-inpainting), colorization (image-colorization) and harmonization. The
reference formulations are the physics-based linear illumination model of Shadow Image
Decomposition / SP+M-Net (Le et al., "Shadow Removal via Shadow Image Decomposition",
ICCV 2019; "Physics-based Shadow Image Decomposition for Shadow Removal", TPAMI 2021), the
multi-context deep net DeshadowNet (Qu et al., CVPR 2017) and the joint detect-and-remove
ST-CGAN (Wang et al., CVPR 2018).

THE SHADOW ILLUMINATION MODEL (linear attenuation; Le et al. ICCV 2019):

    I_shadow(x) = a(x) * J(x)          with   a(x) = 1 - (1 - att) * m(x)

where J is the clean shadow-free scene (the GT), m(x) in [0,1] is the (soft) shadow matte /
mask (1 = deep umbra, 0 = fully lit, fractional = penumbra) and `att` in (0,1) is the
per-shadow attenuation factor (how dark the umbra is: smaller = darker shadow). Given the
mask and the attenuation the clean image is recovered EXACTLY by inverting the model:
J = I_shadow / a. Equivalently the ILLUMINATED (relit) image is a linear function of the
shadowed pixels, I_lit = w * I_shadow (+ b), which is precisely the SP+M-Net parameterisation
that takes the shadow MASK as an input to predict the illumination parameters.

A compact conv net is trained a few hundred steps on a TINY fixed set of shadowed->clean
patch pairs and evaluated by SHADOW-REGION PSNR on a held-out split. Data are REAL (shadow,
shadow-free, mask) triplets from ISTD (Wang, Li & Yang, "Stacked Conditional Generative
Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal", CVPR 2018):
the same static outdoor scene photographed twice, with and without a physical object casting
a real cast shadow, so the "clean" target is an authentic photograph of the unshadowed scene
(not a synthetic composite) and the mask is the corresponding ground-truth shadow region.
Each 640x480 triplet is center-cropped to a square and resized to 64x64 by prepare_data.py
(no download at run time -- data is prepared once and cached to npz). The shadow MASK is
provided as an extra input channel. Train and val use ISTD's own DISJOINT-SCENE train/test
split.

The agent edits ONE design surface (chosen by --surface); everything else (data, backbone
width/depth, optimiser, iterations, seed, eval split, degradation, the metric) is FIXED, so
any change in the score is attributable to the edited surface.

=========================================================================================
EDITABLE SURFACES (one per task). Every surface plugs into the SAME validated mask-guided
residual deshadower harness (same data, base width/depth, optimiser, iters, seed, eval
split, shadow-region-PSNR metric); the residual FORMULATION and the mask-as-4th-channel
input are FIXED for every surface except `network`/`mask` (which vary exactly that).

  network   -> get_network_config() : the deshadower BACKBONE + whether it uses the mask.
     {'arch':'copy'}       = pass the shadowed input straight through (NO removal): the
                             do-nothing floor.
     {'arch':'unet_nomask'}= a BLIND U-Net that sees only the 3-ch shadowed image and must
                             both LOCATE and correct the shadow from RGB alone (DeshadowNet-
                             style multi-context net without the mask prior). Removes some
                             shadow but leaks / over-corrects at the soft boundary.
     {'arch':'unet_mask'}  = the MASK-GUIDED U-Net: the soft shadow mask is concatenated as a
                             4th input channel, so the net knows exactly WHERE and HOW MUCH
                             to brighten -- the SP+M-Net physically-parameterised recovery
                             that fits the multiplicative attenuation. The strong answer.

  mask      -> get_mask_config() : {'use_mask': True|False} -- feed the soft shadow mask as
              a 4th input channel or not (a focused re-framing of the `network` blind-vs-
              mask-guided comparison, SP+M-Net vs DeshadowNet).

  architecture -> get_arch_config() : SHALLOW 1-level U-Net vs the DEEPER 2-level encoder-
              decoder (more downsampling stages -> larger receptive field to cover big
              shadows). {'depth': 1|2}.

  loss      -> get_loss_config() : the reconstruction loss composition -- plain L1, +SSIM,
              +color-consistency, +composition (re-shadow) consistency. Richer, physically-
              grounded loss terms sharpen the recovered penumbra / colour.

  attention -> get_attention_config() : squeeze-excite CHANNEL ATTENTION on the trunk
              (RCAN-style) so the net emphasises the shadow-carrying channels.

  dilation  -> get_dilation_config() : per-block DILATION schedule -> receptive field. A
              dilated trunk covers large shadows the small-RF trunk cannot in one pass.

  normalization -> get_norm_config() : 'none' | 'bn' | 'in' normalisation in the blocks.

  multiscale -> get_multiscale_config() : single-scale vs a COARSE-TO-FINE pyramid that
              relights at half resolution first (captures big soft shadows) then refines.

  fusion    -> get_fusion_config() : last-block features only vs DENSE multi-level feature
              fusion (DenseNet/RDN-style) across all trunk blocks.

  physics   -> get_physics_config() : predict a free 3-ch residual (unconstrained) vs the
              PHYSICS-PARAMETERISED SP+M-Net illumination model -- the net predicts per-pixel
              affine relighting params (w,b) and outputs J = w*I + b, guaranteeing a valid
              multiplicative-illumination inverse. The physical parameterisation.

  upsampling -> get_upsampling_config() : transpose-conv (checkerboard) vs bilinear-resize +
              conv decoder upsampler.

  residual  -> get_residual_config() : direct clean-image regression vs RESIDUAL learning
              (clean = shadowed + net(.)) -- predict only the correction, an easier target
              for the near-multiplicative degradation.

Metric line (one per run):
  DESHADOW_METRICS surface=<S> setting=<L> psnr=<..> psnr_gain=<..> shadow_psnr=<..> \
      ssim=<..> mse=<..> full_psnr=<..>
`psnr` is the SHADOW-REGION PSNR of the DESHADOWED output vs the clean GT (dB, HIGHER better,
computed ONLY over pixels the shadow touches, m>0) and is the PRIMARY metric -- so a method
that merely copies the LIT region cannot win, it must actually brighten the shadow.
`shadow_psnr` is the PSNR of the *shadowed input* over the shadow region vs the clean GT --
the identity ("do-nothing" / copy) floor. `psnr_gain = psnr - shadow_psnr` is reported so it
is explicit that the deshadowed output must BEAT passing the shadowed input through: a
degenerate net that copies its input scores psnr==shadow_psnr (gain 0), and a constant /
all-white / all-black output scores far BELOW the shadowed-input floor. `full_psnr` (whole
image), `ssim` and `mse` are diagnostics.

Every hook is wrapped so a malformed / crashing return falls back to a sane default (the
strong reference config) rather than aborting the run.
"""
from __future__ import annotations

import argparse
import importlib.util
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
IMG = 64                 # patch size (ISTD real photo, center-cropped + resized)
BASE = 32                # deshadower base width (FIXED)
BS = 32                  # batch size (FIXED)

# The full set of editable SURFACES that were DESIGNED on this harness. Each is a single
# design lever on the SAME fixed shadow-removal pipeline (same data, base width/depth,
# optimiser, iters, seed, eval split, shadow-region-PSNR metric), so any score change is
# attributable to the edited lever. All 12 are RUNNABLE.
#
# The data pipeline was swapped from a SYNTHETIC linear-illumination cast shadow (CIFAR-10
# patches) to REAL ISTD (Wang et al. CVPR 2018) shadow/shadow-free/mask photo triplets (see
# this file's data section + vendor/data_scripts/image-deshadow/prepare_data.py). A CPU
# smoke-test re-check on the real data (BASE shrunk 32->8..14 + reduced iters, purely for CPU
# tractability -- ORDERING signal only, not final score_spec numbers; 2 seeds; full provenance
# in vendor/image-deshadow/anchors/real_istd_cpu_smoke.log) re-validated all 6 surfaces that
# were monotone on the OLD synthetic set, but ONE surface (`physics`) that was aggregate-
# monotone on synthetic data INVERTS at the aggregate level on real data and has been
# re-classified DROPPED (see tasks/deshadow-physics/DROPPED.md for the real-data numbers). A
# full GPU re-anchor of the 6 surviving SHIPPED tasks' score_spec anchors on real ISTD data is
# still pending (tracked, not yet done by this pass -- the pinned _WEAK/_STRONG numbers in
# score_spec.py are still the OLD synthetic-GPU numbers).
#
#   SHIPPED (robustly monotone on synthetic GPU anchors AND real-ISTD CPU smoke re-check;
#   task in tasks/deshadow-<s>; GPU re-anchor on real data still pending):
#     network       # copy floor / blind U-Net / mask-guided U-Net (SP+M-Net)   [validated]
#     mask          # mask as 4th input channel off vs on (SP+M-Net vs blind)   [validated]
#     dilation      # dilation [1,1] small RF vs [2,4] dilated large RF          [validated]
#     fusion        # last-block features vs dense multi-level fusion (RDN)      [validated]
#     upsampling    # transpose-conv vs bilinear-resize+conv decoder      [agg-monotone, weak]
#   DROPPED (not shipped):
#     physics       # free residual vs SP+M-Net affine J=w*I+b -- was agg-monotone on
#                    # SYNTHETIC data, INVERTS (weak>strong aggregate, both seeds) on REAL
#                    # ISTD data: the hard affine constraint is a misspecified prior once the
#                    # degradation is not an exact linear model. See tasks/deshadow-physics/
#                    # DROPPED.md.
#     architecture, loss, attention, normalization, multiscale, residual
#                    # non-monotone / not cross-seed robust on the ORIGINAL synthetic set;
#                    # NOT re-validated on real data (kept for provenance only).
SURFACES = (
    "network",        # copy / blind U-Net / mask-guided U-Net (SP+M-Net)          [SHIPPED]
    "mask",           # mask as 4th input channel: off vs on (SP+M-Net vs blind)   [SHIPPED]
    "architecture",   # shallow 1-level U-Net vs deeper 2-level encoder-decoder    [DROPPED]
    "loss",           # L1 vs +SSIM +color +composition (re-shadow) consistency    [DROPPED]
    "attention",      # no gating vs squeeze-excite channel attention (RCAN)       [DROPPED]
    "dilation",       # dilation=1 (small RF) vs dilated trunk (large RF)          [SHIPPED]
    "normalization",  # no norm vs instance/batch norm                             [DROPPED]
    "multiscale",     # single-scale vs coarse-to-fine relight pyramid             [DROPPED]
    "fusion",         # last-block features only vs dense multi-level fusion       [SHIPPED]
    "physics",        # free residual vs SP+M-Net affine illumination model        [DROPPED]
    "upsampling",     # transpose-conv vs bilinear-resize + conv decoder           [SHIPPED]
    "residual",       # direct clean regression vs residual correction learning    [DROPPED]
)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------- #
# Data: shadowed->clean pairs + soft shadow mask. Prepared offline
# (prepare_data.py) into npz with
#   shad (N,3,H,W) in [0,1]   clean (N,3,H,W) in [0,1]   mask (N,1,H,W) in [0,1]
# train / val splits use DISJOINT clean patches AND disjoint shadow RNG (fixed).
# --------------------------------------------------------------------------- #
def load_split(root: str, split: str):
    arr = np.load(os.path.join(root, f"{split}.npz"))
    shad = torch.from_numpy(arr["shad"].astype(np.float32))
    clean = torch.from_numpy(arr["clean"].astype(np.float32))
    mask = torch.from_numpy(arr["mask"].astype(np.float32))
    return shad, clean, mask


# --------------------------------------------------------------------------- #
# Building blocks. A ResBlock trunk shared by every backbone/surface. Design knobs
# (dilation / norm / activation) default to the plain 3x3 / no-norm / ReLU block so an
# untouched surface leaves the trunk unchanged.
# --------------------------------------------------------------------------- #
def _make_norm(kind: str, c: int):
    kind = str(kind).lower()
    if kind in ("bn", "batch", "batchnorm"):
        return nn.BatchNorm2d(c)
    if kind in ("in", "instance", "instancenorm"):
        return nn.InstanceNorm2d(c, affine=True)
    return nn.Identity()  # 'none'


class _ResBlock(nn.Module):
    def __init__(self, c, dilation: int = 1, norm: str = "none"):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        d = max(1, int(dilation))
        self.c2 = nn.Conv2d(c, c, 3, padding=d, dilation=d)
        self.n1 = _make_norm(norm, c)
        self.n2 = _make_norm(norm, c)

    def forward(self, x):
        y = F.relu(self.n1(self.c1(x)), inplace=True)
        y = self.n2(self.c2(y))
        return x + y


class _ChannelAttention(nn.Module):
    """Squeeze-excite channel gate (the `attention` surface)."""

    def __init__(self, c, r=4):
        super().__init__()
        self.fc1 = nn.Conv2d(c, max(4, c // r), 1)
        self.fc2 = nn.Conv2d(max(4, c // r), c, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class UNetDeshadow(nn.Module):
    """U-Net encoder-decoder with ResBlocks + skip connections (multi-scale receptive
    field). Takes an `in_ch`-ch input (3 = shadowed RGB, or 4 = shadowed RGB + shadow
    mask) and returns an `out_ch`-ch field.

    Design knobs used by the editable SURFACES (all default to the plain 2-level, no-norm,
    transpose-conv, no-attention, no-fusion, dilation=1 net so an untouched surface leaves
    the backbone unchanged):
      depth      : 1 or 2 encoder/decoder levels (the `architecture` surface).
      attention  : squeeze-excite channel gate before the output conv (`attention`).
      dilations  : per-mid-block dilation schedule -> receptive field (`dilation`).
      norm       : 'none'|'bn'|'in' normalisation in the blocks (`normalization`).
      up         : 'transpose'|'bilinear' decoder upsampler (`upsampling`).
      fusion     : concat every decoder-level feature before the output conv (`fusion`)."""

    def __init__(self, in_ch: int = 3, base: int = BASE, out_ch: int = 3,
                 depth: int = 2, attention: bool = False, dilations=None,
                 norm: str = "none", up: str = "transpose", fusion: bool = False):
        super().__init__()
        self.depth = 2 if int(depth) >= 2 else 1
        self.norm = norm
        self.up_kind = str(up).lower()
        self.fusion = bool(fusion)
        if dilations is None:
            dilations = [1, 1]
        dilations = list(dilations) + [1] * max(0, 2 - len(dilations))

        self.inc = nn.Conv2d(in_ch, base, 3, padding=1)
        self.enc1 = _ResBlock(base, norm=norm)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc2 = _ResBlock(base * 2, norm=norm)
        if self.depth >= 2:
            self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
            self.mid = nn.Sequential(
                _ResBlock(base * 4, dilation=dilations[0], norm=norm),
                _ResBlock(base * 4, dilation=dilations[1], norm=norm))
            self.up2 = self._make_up(base * 4, base * 2)
            self.dec2 = _ResBlock(base * 2, norm=norm)
        else:
            self.mid = nn.Sequential(
                _ResBlock(base * 2, dilation=dilations[0], norm=norm),
                _ResBlock(base * 2, dilation=dilations[1], norm=norm))
        self.up1 = self._make_up(base * 2, base)
        self.dec1 = _ResBlock(base, norm=norm)
        self.att = _ChannelAttention(base) if attention else None
        if self.fusion:
            # fuse [dec1 feats (base ch)] (+ upsampled dec2 feats (base*2 ch) if depth>=2)
            # -> base channels. depth1: base; depth2: base + base*2 = base*3.
            fuse_in = base + (base * 2 if self.depth >= 2 else 0)
            self.fuse = nn.Conv2d(fuse_in, base, 1)
        self.outc = nn.Conv2d(base, out_ch, 3, padding=1)

    def _make_up(self, cin, cout):
        if self.up_kind in ("bilinear", "resize"):
            return nn.Conv2d(cin, cout, 3, padding=1)   # applied after F.interpolate
        return nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1)

    def _up(self, mod, x, size):
        if self.up_kind in ("bilinear", "resize"):
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            return mod(x)
        u = mod(x)
        if u.shape[-2:] != size:
            u = F.interpolate(u, size=size, mode="nearest")
        return u

    def forward(self, x):
        h0 = F.relu(self.inc(x), inplace=True)
        e1 = self.enc1(h0)
        e2 = self.enc2(F.relu(self.down1(e1), inplace=True))
        if self.depth >= 2:
            m = self.mid(F.relu(self.down2(e2), inplace=True))
            d2 = self.dec2(self._up(self.up2, m, e2.shape[-2:]) + e2)
        else:
            d2 = self.mid(e2)
        d1 = self.dec1(self._up(self.up1, d2, e1.shape[-2:]) + e1)
        if self.fusion:
            feats = [d1]
            if self.depth >= 2:
                feats.append(F.interpolate(d2, size=d1.shape[-2:], mode="nearest"))
            d1 = self.fuse(torch.cat(feats, dim=1))
        if self.att is not None:
            d1 = self.att(d1)
        return self.outc(d1)


class MultiScaleDeshadow(nn.Module):
    """Coarse-to-fine PYRAMID deshadower (the `multiscale` surface). The input is relit at
    HALF resolution first (larger effective receptive field -> captures big soft shadows)
    and the coarse estimate is fused into the fine branch. Same per-branch width."""

    def __init__(self, in_ch: int = 4, base: int = BASE, out_ch: int = 3,
                 up: str = "bilinear", norm: str = "none"):
        super().__init__()
        self.coarse = UNetDeshadow(in_ch, base, out_ch, depth=1, norm=norm, up=up)
        self.fine = UNetDeshadow(in_ch + out_ch, base, out_ch, depth=2, norm=norm, up=up)

    def forward(self, x):
        H, W = x.shape[-2:]
        x_lo = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        y_lo = self.coarse(x_lo)
        y_up = F.interpolate(y_lo, size=(H, W), mode="bilinear", align_corners=False)
        return self.fine(torch.cat([x, y_up], dim=1))


def build_backbone(arch: str) -> nn.Module | None:
    """`copy` -> None (identity, handled in the forward wrapper). `unet_nomask` -> 3-ch
    input U-Net (blind). `unet_mask` -> 4-ch input U-Net (shadowed RGB + mask)."""
    if arch == "copy":
        return None
    if arch == "unet_mask":
        return UNetDeshadow(in_ch=4, base=BASE, out_ch=3)
    return UNetDeshadow(in_ch=3, base=BASE, out_ch=3)


# --------------------------------------------------------------------------- #
# Metrics: PSNR (full and MASKED shadow-region) + a simple SSIM, on images in [0,1].
# --------------------------------------------------------------------------- #
def psnr_batch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = pred.clamp(0, 1).float()
    gt = gt.clamp(0, 1).float()
    mse = ((pred - gt) ** 2).reshape(pred.shape[0], -1).mean(1)
    mse = mse.clamp_min(1e-10)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return float(psnr.mean())


def psnr_masked(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
                thresh: float = 0.05) -> float:
    """PSNR computed ONLY over the shadow region (pixels where the soft mask m > thresh).
    Per-image mean-squared error is averaged over the masked pixels of all 3 channels, so a
    method that only fixes the LIT region (m<=thresh) gains nothing here -- it must actually
    brighten the shadowed pixels to raise this number."""
    pred = pred.clamp(0, 1).float()
    gt = gt.clamp(0, 1).float()
    sel = (mask > thresh).float()                       # (N,1,H,W)
    sel3 = sel.expand_as(pred)                          # (N,3,H,W)
    se = ((pred - gt) ** 2) * sel3
    denom = sel3.reshape(sel3.shape[0], -1).sum(1).clamp_min(1.0)
    mse = se.reshape(se.shape[0], -1).sum(1) / denom     # per-image MSE over shadow px
    mse = mse.clamp_min(1e-10)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return float(psnr.mean())


def _gaussian_window(ch, ks=11, sigma=1.5, device="cpu"):
    coords = torch.arange(ks, dtype=torch.float32, device=device) - (ks - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    w = (g[:, None] * g[None, :])[None, None]
    return w.expand(ch, 1, ks, ks).contiguous()


def _ssim_map(pred, gt):
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
    return ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_2 + mu2_2 + C1) * (s1 + s2 + C2))


def ssim_batch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return float(_ssim_map(pred, gt).mean())


# --------------------------------------------------------------------------- #
# Loss builders. The FIXED base loss (all non-`loss` surfaces) is robust L1 + SSIM,
# up-weighted inside the shadow region (matches the shadow-region PSNR metric). The `loss`
# surface additionally toggles a COLOR-consistency term (chroma of the recovered region)
# and a COMPOSITION / re-shadow consistency term (re-apply the shadow model to the recovery
# and match the input) -- physically-grounded terms that sharpen the penumbra / colour.
# --------------------------------------------------------------------------- #
def _color_loss(pred, gt, mask):
    """Per-pixel chroma (mean-subtracted RGB) L1, up-weighted in the shadow -- penalises the
    hue/colour cast a blind brightening leaves in the recovered umbra."""
    pm = pred - pred.mean(dim=1, keepdim=True)
    gm = gt - gt.mean(dim=1, keepdim=True)
    w = 1.0 + 4.0 * mask
    return (((pm - gm).abs()) * w).mean()


def _composition_loss(pred, shad, mask):
    """Re-shadow consistency: re-apply the (known) soft-mask attenuation to the recovered
    clean estimate and require it to reproduce the shadowed INPUT in the lit / penumbra
    region. A cheap self-supervised physical consistency (SP+M-Net decomposition idea)."""
    # estimate a per-image umbra gain from the mask=1 region is not available here; instead
    # require the recovery to AGREE with the input where the shadow is weak (m small), i.e.
    # the net must not corrupt the already-lit pixels.
    keep = (1.0 - mask)                       # lit / weakly-shadowed region weight
    return (((pred - shad).abs()) * keep).mean()


def build_loss(cfg: dict):
    """cfg keys (defaults = the STRONG composite loss):
      ssim   : include the SSIM structural term (bool)
      color  : include the chroma-consistency term (bool)
      comp   : include the re-shadow composition-consistency term (bool)
    L1 (shadow-up-weighted) is ALWAYS present. Returns fn(pred, gt, mask, shad)."""
    use_ssim = bool(cfg.get("ssim", True))
    use_color = bool(cfg.get("color", True))
    use_comp = bool(cfg.get("comp", True))

    def loss_fn(pred, gt, mask, shad):
        pred = pred.clamp(0, 1)
        w = 1.0 + 4.0 * mask                             # up-weight the shadow region
        loss = (((pred - gt).abs()) * w).mean()          # L1 (always)
        if use_ssim:
            loss = loss + 0.2 * (1.0 - _ssim_map(pred, gt).mean())
        if use_color:
            loss = loss + 0.2 * _color_loss(pred, gt, mask)
        if use_comp:
            loss = loss + 0.1 * _composition_loss(pred, shad, mask)
        return loss

    return loss_fn


# --------------------------------------------------------------------------- #
# FIXED reconstruction loss used by every NON-`loss` surface: robust L1 + SSIM, weighted
# UP inside the shadow region (this is the validated strong base loss).
# --------------------------------------------------------------------------- #
def recon_loss(pred, gt, mask):
    pred = pred.clamp(0, 1)
    w = 1.0 + 4.0 * mask                                 # up-weight the shadow region
    l1 = (((pred - gt).abs()) * w).mean()
    ssim = 1.0 - _ssim_map(pred, gt).mean()
    return l1 + 0.2 * ssim


# --------------------------------------------------------------------------- #
# Deshadowers built on the mask-guided U-Net. `free` = predict a 3-ch RESIDUAL added to the
# shadowed input (the validated strong formulation). `physics` = SP+M-Net: predict per-pixel
# affine relighting params (w,b) from the masked input and output J = w*I + b (a valid
# multiplicative-illumination inverse). `direct` = regress the clean image directly.
# --------------------------------------------------------------------------- #
class ConfigurableDeshadow(nn.Module):
    """Mask-guided (4-ch) residual deshadower whose SINGLE U-Net trunk is configured by ONE
    editable design knob (depth / attention / dilation / norm / up / fusion / multiscale).
    The mask-as-4th-channel input and the residual FORMULATION are FIXED, so a change in
    shadow-region PSNR is attributable to the one knob."""

    def __init__(self, use_mask=True, depth=2, attention=False, dilations=None,
                 norm="none", up="transpose", fusion=False, multiscale=False,
                 mode="residual"):
        super().__init__()
        self.use_mask = bool(use_mask)
        self.mode = str(mode)            # 'residual' | 'direct' | 'physics'
        in_ch = 4 if self.use_mask else 3
        out_ch = 6 if self.mode == "physics" else 3    # physics: per-pixel (w,b) x3
        if multiscale:
            self.net = MultiScaleDeshadow(in_ch=in_ch, base=BASE, out_ch=out_ch,
                                          up=up, norm=norm)
        else:
            self.net = UNetDeshadow(in_ch=in_ch, base=BASE, out_ch=out_ch, depth=depth,
                                    attention=attention, dilations=dilations, norm=norm,
                                    up=up, fusion=fusion)

    def forward(self, shad, mask):
        inp = torch.cat([shad, mask], dim=1) if self.use_mask else shad
        y = self.net(inp)
        if self.mode == "physics":
            w, b = y[:, :3], y[:, 3:]
            # init near identity (w~1, b~0): center w around 1.
            return (1.0 + w) * shad + 0.1 * b
        if self.mode == "direct":
            return y
        return shad + y                  # residual (default strong)


# --------------------------------------------------------------------------- #
# Defaults (fallbacks) = the STRONG reference config for each surface.
# --------------------------------------------------------------------------- #
def default_network():
    return dict(arch="unet_mask")


def default_mask():
    return dict(use_mask=True)


def default_arch():
    return dict(depth=2)


def default_loss():
    return dict(ssim=True, color=True, comp=True)


def default_attention():
    return dict(attention=True)


def default_dilation():
    return dict(dilations=[2, 4])


def default_norm():
    return dict(norm="in")


def default_multiscale():
    return dict(multiscale=True)


def default_fusion():
    return dict(fusion=True)


def default_physics():
    return dict(mode="physics")


def default_upsampling():
    return dict(up="bilinear")


def default_residual():
    return dict(mode="residual")


# Weak (default-stub) configs shipped in the agent solution for each NEW surface.
_WEAK = {
    "mask":          dict(use_mask=False),
    "architecture":  dict(depth=1),
    "loss":          dict(ssim=False, color=False, comp=False),
    "attention":     dict(attention=False),
    "dilation":      dict(dilations=[1, 1]),
    "normalization": dict(norm="none"),
    "multiscale":    dict(multiscale=False),
    "fusion":        dict(fusion=False),
    "physics":       dict(mode="residual"),
    "upsampling":    dict(up="transpose"),
    "residual":      dict(mode="direct"),
}

# hook name + strong default for each NEW surface.
_HOOK = {
    "mask":          ("get_mask_config", default_mask),
    "architecture":  ("get_arch_config", default_arch),
    "loss":          ("get_loss_config", default_loss),
    "attention":     ("get_attention_config", default_attention),
    "dilation":      ("get_dilation_config", default_dilation),
    "normalization": ("get_norm_config", default_norm),
    "multiscale":    ("get_multiscale_config", default_multiscale),
    "fusion":        ("get_fusion_config", default_fusion),
    "physics":       ("get_physics_config", default_physics),
    "upsampling":    ("get_upsampling_config", default_upsampling),
    "residual":      ("get_residual_config", default_residual),
}


# --------------------------------------------------------------------------- #
# Load agent surface
# --------------------------------------------------------------------------- #
def load_surface(sol_path: Path):
    spec = importlib.util.spec_from_file_location("agent_surface", str(sol_path))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(sol_path.parent))
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _resolve_hook(surface, mod):
    """Resolve the config dict for a NEW surface. Call its hook; a missing/broken hook falls
    back to the surface's WEAK default (so an untouched stub == the weak baseline). The
    returned dict is merged onto the STRONG-reference default so a partial return is full."""
    hook_name, strong_default = _HOOK[surface]
    strong = strong_default()
    try:
        fn = getattr(mod, hook_name)
        cand = fn()
        assert isinstance(cand, dict)
        cfg = {**strong, **cand}
        print(f"{surface.upper()}_APPLIED {cfg}", flush=True)
        return cfg
    except Exception as e:  # noqa: BLE001
        cfg = dict(_WEAK[surface])
        print(f"{surface.upper()}_FALLBACK reason={e!r} -> weak {cfg}", flush=True)
        return cfg


def _build_configured(surface, cfg):
    """Build the mask-guided residual deshadower for a NEW surface from its cfg."""
    kw = dict(use_mask=True, depth=2, attention=False, dilations=None, norm="none",
              up="transpose", fusion=False, multiscale=False, mode="residual")
    if surface == "mask":
        kw["use_mask"] = bool(cfg.get("use_mask", True))
    elif surface == "architecture":
        kw["depth"] = int(cfg.get("depth", 2))
    elif surface == "attention":
        kw["attention"] = bool(cfg.get("attention", True))
    elif surface == "dilation":
        kw["dilations"] = cfg.get("dilations", [2, 4])
    elif surface == "normalization":
        kw["norm"] = str(cfg.get("norm", "in"))
    elif surface == "multiscale":
        kw["multiscale"] = bool(cfg.get("multiscale", True))
    elif surface == "fusion":
        kw["fusion"] = bool(cfg.get("fusion", True))
    elif surface == "physics":
        kw["mode"] = str(cfg.get("mode", "physics"))
    elif surface == "upsampling":
        kw["up"] = str(cfg.get("up", "bilinear"))
    elif surface == "residual":
        kw["mode"] = str(cfg.get("mode", "residual"))
    return ConfigurableDeshadow(**kw)


# --------------------------------------------------------------------------- #
# Forward for the `network` surface (copy / blind / mask-guided named archs).
# --------------------------------------------------------------------------- #
def _deshadow_named(model, arch, shad, mask):
    if arch == "copy" or model is None:
        return shad
    if arch == "unet_mask":
        inp = torch.cat([shad, mask], dim=1)             # (B,4,H,W)
    else:
        inp = shad                                       # (B,3,H,W) blind
    return shad + model(inp)


# --------------------------------------------------------------------------- #
# Train + eval
# --------------------------------------------------------------------------- #
def run(surface, mod, data_root, device, iters, seed):
    set_all_seeds(seed)
    s_tr, c_tr, m_tr = load_split(data_root, "train")
    s_va, c_va, m_va = load_split(data_root, "val")
    s_tr, c_tr, m_tr = s_tr.to(device), c_tr.to(device), m_tr.to(device)
    s_va, c_va, m_va = s_va.to(device), c_va.to(device), m_va.to(device)
    print(f"DATA train={s_tr.shape[0]} val={s_va.shape[0]} img={tuple(s_tr.shape[-2:])}",
          flush=True)

    named_arch = None
    loss_fn = None
    if surface == "network":
        # ---- named-arch surface: copy / unet_nomask / unet_mask ----
        net_cfg = default_network()
        try:
            cand = mod.get_network_config()
            assert isinstance(cand, dict) and "arch" in cand
            a = str(cand["arch"]).lower()
            assert a in ("copy", "unet_nomask", "unet_mask")
            net_cfg = dict(arch=a)
            print(f"NETWORK_APPLIED {net_cfg}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"NETWORK_FALLBACK reason={e!r}", flush=True)
        named_arch = str(net_cfg.get("arch", "unet_mask"))
        model = build_backbone(named_arch)
    elif surface == "loss":
        # ---- loss surface: FIXED mask-guided residual deshadower, vary the loss ----
        loss_cfg = _resolve_hook("loss", mod)
        loss_fn = build_loss(loss_cfg)
        model = ConfigurableDeshadow(use_mask=True, mode="residual")
    else:
        # ---- every other NEW surface: FIXED mask-guided residual deshadower, one knob ----
        cfg = _resolve_hook(surface, mod)
        model = _build_configured(surface, cfg)

    if loss_fn is None:
        loss_fn = lambda p, g, m, s: recon_loss(p, g, m)  # noqa: E731

    # ---- build optimiser + train (FIXED) ----
    if model is not None:
        model = model.to(device).train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        N = s_tr.shape[0]
        for it in range(iters):
            sel = torch.randint(0, N, (BS,), device=device)
            shad = s_tr[sel]; clean = c_tr[sel]; mask = m_tr[sel]
            if surface == "network":
                out = _deshadow_named(model, named_arch, shad, mask)
            else:
                out = model(shad, mask)
            loss = loss_fn(out, clean, mask, shad)
            if not torch.isfinite(loss):
                loss = out.float().pow(2).mean() * 0.0 + 1.0
            opt.zero_grad(); loss.backward(); opt.step()
            if it % max(1, iters // 5) == 0 or it == iters - 1:
                print(f"train it={it} loss={float(loss):.5f}", flush=True)
        model.eval()

    # ---- eval on held-out val ----
    preds = []
    with torch.no_grad():
        for i in range(0, s_va.shape[0], 64):
            sh = s_va[i:i + 64]; mk = m_va[i:i + 64]
            if surface == "network":
                out = _deshadow_named(model, named_arch, sh, mk)
            else:
                out = model(sh, mk)
            preds.append(out.clamp(0, 1).cpu())
    pred = torch.cat(preds, 0)
    clean = c_va.cpu()
    shad = s_va.cpu()
    mask = m_va.cpu()

    psnr = psnr_masked(pred, clean, mask)                # PRIMARY: shadow-region PSNR
    shadow_psnr = psnr_masked(shad, clean, mask)         # identity floor (copy input)
    full_psnr = psnr_batch(pred, clean)                  # diagnostic (whole image)
    ssim = ssim_batch(pred, clean)
    mse = float(((pred.clamp(0, 1) - clean) ** 2).mean())
    return dict(psnr=psnr, shadow_psnr=shadow_psnr, psnr_gain=psnr - shadow_psnr,
                ssim=ssim, mse=mse, full_psnr=full_psnr)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--surface", required=True, choices=list(SURFACES))
    ap.add_argument("--label", default="run")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE {device} torch {torch.__version__}", flush=True)

    mod = load_surface(Path(args.solution))
    m = run(args.surface, mod, args.data_root, device, args.iters, args.seed)

    print(f"DESHADOW_METRICS surface={args.surface} setting={args.label} "
          f"psnr={m['psnr']:.4f} psnr_gain={m['psnr_gain']:.4f} "
          f"shadow_psnr={m['shadow_psnr']:.4f} ssim={m['ssim']:.4f} "
          f"mse={m['mse']:.6f} full_psnr={m['full_psnr']:.4f}", flush=True)


if __name__ == "__main__":
    main()
