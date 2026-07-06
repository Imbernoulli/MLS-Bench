#!/usr/bin/env python3
"""Image-harmonization harness (self-contained, config-driven, multi-surface).

IMAGE HARMONIZATION: given a COMPOSITE image (a foreground region pasted into a
background whose colour/brightness does NOT match) and the foreground MASK, RECOLOUR
the foreground so it is photometrically consistent with the background. This is a
genuinely distinct restoration direction: the foreground content is already present
and structurally correct -- only its colour STATISTICS are wrong -- so it is separate
from inpainting (fill a hole), matting (extract an alpha), colorization (grey->colour)
and dehazing (remove a global scattering degradation). The reference methods are
DoveNet (Cong et al., CVPR 2020, which introduced the iHarmony4 benchmark, a plain
encoder-decoder U-Net trained to regress the harmonized image) and RainNet (Ling et
al., "Region-aware Adaptive Instance Normalization for Image Harmonization", CVPR
2021), which adds a REGION-AWARE normalization (RAIN) module that transfers the
BACKGROUND feature statistics onto the FOREGROUND features -- the current SOTA design.

THE COMPOSITE MODEL (see prepare_data.py) -- REAL iHarmony4 data (Cong et al., DoveNet,
CVPR 2020, https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4):
    composite(x) = m(x) * T(J(x))  +  (1 - m(x)) * J(x)
where J is the REAL photo (the EXACT harmonized ground truth), m is the foreground mask
and T is a REAL colour-transfer artifact (applied by the iHarmony4 authors using colour-
transfer methods against real reference photos, NOT a synthetic knob) that only affects
the foreground. The background is UNTOUCHED (already correct); the harmonizer must
recover J inside the foreground. The three severities (mild/medium/strong) are three of
iHarmony4's four real photographic sub-datasets (HCOCO / Hday2night / HFlickr), ordered by
their MEASURED foreground composite-vs-GT PSNR floor at this harness's fixed 64x64 working
resolution (see prepare_data.py for the exact mapping and provenance).

The agent edits ONE design surface (chosen by --surface); everything else (data, base
width/depth, optimiser, iterations, seed, eval split, degradation, the metric) is FIXED,
so any change in the score is attributable to the edited surface. Each surface exposes
ONE hook the agent designs; every other axis is fixed at a sane default from
`default_config()` (which is the STRONG reference config -- a mask-conditioned residual
U-Net with skips, batchnorm, transpose upsampling, L1+foreground-emphasis loss).

EDITABLE SURFACES (one task per surface, --surface):

  network      -> get_network_config(): {'arch': 'copy'|'blind'|'mask'|'rain'}
                  HOW MUCH REGION INFORMATION the harmonizer uses. copy=do-nothing floor;
                  blind=mask-BLIND 3-ch U-Net (region-agnostic, under-corrects); mask=
                  mask-CONDITIONED 4-ch U-Net (recolours only the foreground); rain=+RAIN.
                  (The ORIGINAL validated surface -- unchanged.)
  maskcond     -> get_mask_conditioning(): 'none'|'concat'|'gated'  HOW the mask is FED to
                  the net. none=mask-blind (region-agnostic); concat=mask concatenated as a
                  4th input channel (DoveNet); gated=concat PLUS a mask-gated output blend
                  that hard-restricts edits to the foreground (background provably preserved).
  normalization-> get_normalization(): 'none'|'batch'|'instance'|'rain'  the normalization
                  layer in the harmonizer. rain=RainNet region-aware AdaIN (Ling et al. CVPR
                  2021, transfers BACKGROUND stats onto FOREGROUND features) is the SOTA; a
                  plain global InstanceNorm ERASES the very foreground/background statistic
                  gap the harmonizer must model -> worst.
  loss         -> get_loss_config(): {mode:'bg'|'global'|'fg', ...}  WHERE the reconstruction
                  loss is applied. bg=supervise the (already-correct) BACKGROUND only -> the
                  net learns to copy the composite through (degenerate, no foreground signal);
                  global=whole-image L1; fg=whole-image L1 + FOREGROUND emphasis (the region
                  that actually needs correcting) -> best.
  fusion       -> get_fusion_config(): {skips: bool}  U-Net encoder->decoder SKIP connections.
                  Without skips the bottleneck loses the high-frequency foreground detail ->
                  blurred recolour; with skips the detail is re-injected -> sharp, accurate.
  colorhead    -> get_color_head(): 'residual'|'affine_global'|'affine_spatial'  the OUTPUT
                  parameterization. residual=predict a full-res RGB residual (DoveNet). Since
                  the degradation IS a per-channel affine, an AFFINE-parametric head that
                  predicts per-channel (gain,bias) -- global or spatially-varying -- matches
                  the true inverse and is more sample-efficient (colour-transform head, cf.
                  learnable colour curves / 3D-LUT harmonizers).
  upsampling   -> get_upsampling(): 'transpose'|'nearest'|'bilinear'  decoder upsampling.
                  nearest=blocky reconstruction; transpose/bilinear=smooth, higher PSNR.
  dilation     -> get_dilation(): bottleneck dilation rate (1..8). A larger dilated receptive
                  field lets the bottleneck SEE the background context surrounding the
                  foreground blob (needed to infer the correct target colour); rate 1 = no
                  extra context.
  attention    -> get_attention_config(): {enabled: bool}  a squeeze-excite channel-attention
                  gate on the bottleneck that recalibrates the per-channel appearance
                  correction. Off = plain conv bottleneck.
  activation   -> get_activation(): 'relu'|'identity'|'gelu'  the conv nonlinearity. identity
                  collapses the net toward a linear map (under-fits the clamp/tint) -> worst;
                  relu/gelu give the needed nonlinearity.
  bgstats      -> get_bgstats_config(): {enabled: bool}  a FOREGROUND-BACKGROUND STATISTICS
                  MATCHING input pre-conditioning: append the per-channel BACKGROUND mean
                  (broadcast) as extra input channels so the net is handed the target colour
                  statistics it must match the foreground to (an explicit statistics-matching
                  prior, cf. classic Reinhard colour transfer + RAIN's stats transfer).
  inputnorm    -> get_input_norm(): 'none'|'bg_whiten'  whether to apply a fixed (non-learned)
                  BACKGROUND-referenced input normalization. 'bg_whiten' whitens the whole
                  image by the BACKGROUND per-channel mean/std (and un-whitens the output); at
                  this scale this naive transform corrupts the input colour scale and the
                  reconstruction collapses -> 'none' (raw composite) is the robust, stronger
                  choice.

Metric line (one per run):
  HARMONY_METRICS surface=<S> setting=<L> fg_psnr=<..> fg_psnr_gain=<..> \
      comp_fg_psnr=<..> fg_mse=<..> fg_ssim=<..>
`fg_psnr` is the FOREGROUND-region PSNR of the HARMONIZED output vs the real GT (dB,
HIGHER better) -- the PRIMARY metric, measured ONLY inside the foreground mask (the
background is already correct, so it is excluded). `comp_fg_psnr` is the
foreground-region PSNR of the *composite input* vs the GT -- the identity ("do-nothing")
floor a harmonizer must beat. `fg_psnr_gain = fg_psnr - comp_fg_psnr` is reported so it
is explicit that the output must BEAT copying the composite through: a copy-composite
degenerate scores fg_psnr == comp_fg_psnr (gain 0). `fg_mse` and `fg_ssim` are
diagnostics.

Every hook is wrapped so a malformed / crashing return falls back to a sane default
(the strong reference config).
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
IMG = 64
BASE = 32                # harmonizer base width (FIXED)
BS = 32                  # batch size (FIXED)

SURFACES = ["network", "maskcond", "normalization", "loss", "fusion", "colorhead",
            "upsampling", "dilation", "attention", "activation", "bgstats", "inputnorm"]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------- #
# Data: REAL (comp, real, mask) triples from iHarmony4. Prepared offline
# (prepare_data.py) into npz. comp (N,3,H,W)  real (N,3,H,W) in [0,1]   mask (N,1,H,W)
# in {0,1}. train / val use each sub-dataset's OFFICIAL disjoint train/test id split.
# --------------------------------------------------------------------------- #
def load_split(root: str, severity: str, split: str):
    arr = np.load(os.path.join(root, f"{split}_{severity}.npz"))
    comp = torch.from_numpy(arr["comp"].astype(np.float32))
    real = torch.from_numpy(arr["real"].astype(np.float32))
    mask = torch.from_numpy(arr["mask"].astype(np.float32))
    return comp, real, mask


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def default_config() -> dict:
    """The STRONG reference config: a mask-conditioned residual U-Net with skip
    connections, batchnorm, transpose upsampling, ReLU, dilation 1, and an
    L1+foreground-emphasis loss. Every surface's default matches this so an untouched
    surface reproduces the strong reference."""
    return dict(
        arch="mask",              # network surface: copy | blind | mask | rain
        mask_cond="concat",       # maskcond surface: none | concat | gated
        norm="batch",             # normalization surface: none | batch | instance | rain
        loss_mode="fg",           # loss surface: bg | global | fg
        skips=True,               # fusion surface: encoder->decoder skips
        color_head="residual",    # colorhead surface: residual | affine_global | affine_spatial
        upsampling="transpose",   # upsampling surface: transpose | nearest | bilinear
        dilation=1,               # dilation surface: bottleneck dilation rate
        attention=False,          # attention surface: squeeze-excite bottleneck gate
        activation="relu",        # activation surface: relu | identity | gelu
        bgstats=False,            # bgstats surface: append background-mean input channels
        input_norm="none",        # inputnorm surface: none | bg_whiten
    )


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def _act(name: str):
    if name == "identity":
        return nn.Identity()
    if name == "gelu":
        return nn.GELU()
    return nn.ReLU(inplace=True)


def _norm(name: str, c: int):
    if name == "batch":
        return nn.BatchNorm2d(c)
    if name == "instance":
        return nn.InstanceNorm2d(c, affine=True)
    return nn.Identity()          # 'none' or 'rain' (rain handled separately)


class _ResBlock(nn.Module):
    def __init__(self, c, act="relu", norm="none"):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n1 = _norm(norm, c)
        self.a1 = _act(act)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = _norm(norm, c)

    def forward(self, x):
        y = self.a1(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return x + y


class _SEGate(nn.Module):
    """Squeeze-excite channel attention (Hu et al. CVPR 2018): global-avg-pool ->
    2-layer MLP -> per-channel sigmoid gate. Recalibrates the per-channel correction."""

    def __init__(self, c, r=4):
        super().__init__()
        self.fc1 = nn.Conv2d(c, max(4, c // r), 1)
        self.fc2 = nn.Conv2d(max(4, c // r), c, 1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


def _region_stats(feat: torch.Tensor, region: torch.Tensor, eps: float = 1e-5):
    """Per-channel (mean, std) of `feat` (B,C,h,w) over the pixels where region==1
    (region is (B,1,h,w) in {0,1}). Returns (B,C,1,1) tensors. Falls back to the whole
    frame if a region is empty."""
    area = region.sum(dim=(2, 3), keepdim=True)               # (B,1,1,1)
    area = area.clamp_min(1.0)
    mean = (feat * region).sum(dim=(2, 3), keepdim=True) / area
    var = ((feat - mean) ** 2 * region).sum(dim=(2, 3), keepdim=True) / area
    std = (var + eps).sqrt()
    return mean, std


class RAIN(nn.Module):
    """Region-aware Adaptive Instance Normalization (RainNet, Ling et al. CVPR 2021).

    Re-normalizes the FOREGROUND features to the BACKGROUND feature statistics: the
    foreground is standardized by its own (mean,std) then re-scaled to the background's
    (mean,std), so its style is explicitly aligned to the background. The background
    features are passed through untouched. `mask` is (B,1,H,W) at the feature
    resolution (1 = foreground)."""

    def forward(self, feat, mask):
        fg = mask
        bg = 1.0 - mask
        f_mean, f_std = _region_stats(feat, fg)
        b_mean, b_std = _region_stats(feat, bg)
        normed = (feat - f_mean) / f_std * b_std + b_mean     # align FG style -> BG style
        return bg * feat + fg * normed


# --------------------------------------------------------------------------- #
# Harmonizer backbone. The config selects: input channels (mask concat / bg-stats),
# normalization (incl. RAIN), skips, upsampling, dilation, activation, attention,
# output head (residual RGB or per-channel affine correction), and mask-gated blend.
# --------------------------------------------------------------------------- #
class UNetHarmonizer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        base = BASE
        self.cfg = cfg
        arch = cfg["arch"]
        # channels the mask is fed as input: 'mask'/'rain' arch and concat/gated maskcond
        # all concatenate the mask; 'blind' arch and 'none' maskcond do not.
        self.use_mask_in = (arch in ("mask", "rain")) and cfg["mask_cond"] != "none"
        self.gated = cfg["mask_cond"] == "gated"
        self.use_rain = (arch == "rain") or (cfg["norm"] == "rain")
        norm = "none" if cfg["norm"] == "rain" else cfg["norm"]
        act = cfg["activation"]
        self.upsampling = cfg["upsampling"]
        self.skips = cfg["skips"]
        self.color_head = cfg["color_head"]
        self.bgstats = cfg["bgstats"]

        in_ch = 3
        if self.use_mask_in:
            in_ch += 1
        if self.bgstats:
            in_ch += 3                                        # broadcast background mean

        self.inc = nn.Conv2d(in_ch, base, 3, padding=1)
        self.inc_act = _act(act)
        self.enc1 = _ResBlock(base, act, norm)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc2 = _ResBlock(base * 2, act, norm)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        dil = int(cfg["dilation"])
        self.mid1 = _ResBlock(base * 4, act, norm)
        # a dilated conv in the bottleneck to widen the receptive field (context)
        self.mid_dil = nn.Conv2d(base * 4, base * 4, 3, padding=dil, dilation=dil)
        self.mid_dil_act = _act(act)
        self.mid2 = _ResBlock(base * 4, act, norm)
        self.attn = _SEGate(base * 4) if cfg["attention"] else nn.Identity()

        self.up2 = self._make_up(base * 4, base * 2)
        self.dec2 = _ResBlock(base * 2, act, norm)
        self.up1 = self._make_up(base * 2, base)
        self.dec1 = _ResBlock(base, act, norm)
        self.up_act = _act(act)

        if self.color_head == "affine_global":
            self.outc = nn.Conv2d(base, 6, 3, padding=1)      # -> pooled per-channel g,b
        elif self.color_head == "affine_spatial":
            self.outc = nn.Conv2d(base, 6, 3, padding=1)      # per-pixel per-channel g,b
        else:
            self.outc = nn.Conv2d(base, 3, 3, padding=1)      # residual RGB

        if self.use_rain:
            self.rain_mid = RAIN(); self.rain_s2 = RAIN(); self.rain_s1 = RAIN()

    def _make_up(self, cin, cout):
        if self.upsampling == "transpose":
            return nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1)
        mode = self.upsampling                                # 'nearest' | 'bilinear'
        return nn.Sequential(_Upsample(mode), nn.Conv2d(cin, cout, 3, padding=1))

    def forward(self, comp, mask, bg_mean=None):
        parts = [comp]
        if self.use_mask_in:
            parts.append(mask)
        if self.bgstats and bg_mean is not None:
            parts.append(bg_mean.expand(-1, -1, comp.shape[-2], comp.shape[-1]))
        x = torch.cat(parts, dim=1) if len(parts) > 1 else comp

        h0 = self.inc_act(self.inc(x))
        e1 = self.enc1(h0)
        e2 = self.enc2(self.inc_act(self.down1(e1)))
        m = self.mid1(self.inc_act(self.down2(e2)))
        m = self.mid_dil_act(self.mid_dil(m))
        m = self.mid2(m)
        m = self.attn(m)
        if self.use_rain:
            mm = F.interpolate(mask, size=m.shape[-2:], mode="nearest")
            m = self.rain_mid(m, mm)
            m2 = F.interpolate(mask, size=e2.shape[-2:], mode="nearest")
            e2 = self.rain_s2(e2, m2)
            m1 = F.interpolate(mask, size=e1.shape[-2:], mode="nearest")
            e1 = self.rain_s1(e1, m1)
        d2 = self.up_act(self.up2(m))
        d2 = self.dec2(d2 + e2 if self.skips else d2)
        d1 = self.up_act(self.up1(d2))
        d1 = self.dec1(d1 + e1 if self.skips else d1)
        raw = self.outc(d1)

        if self.color_head == "residual":
            out = comp + raw
        elif self.color_head == "affine_global":
            gb = raw.mean(dim=(2, 3), keepdim=True)            # (B,6,1,1)
            gain = 1.0 + gb[:, :3]; bias = gb[:, 3:]
            out = gain * comp + bias
        else:  # affine_spatial
            gain = 1.0 + raw[:, :3]; bias = raw[:, 3:]
            out = gain * comp + bias

        if self.gated:
            # hard-restrict the edit to the foreground: background provably preserved.
            out = mask * out + (1.0 - mask) * comp
        return out.clamp(0, 1)


class _Upsample(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == "bilinear":
            return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return F.interpolate(x, scale_factor=2, mode="nearest")


def build_backbone(cfg: dict) -> nn.Module | None:
    if cfg["arch"] == "copy":
        return None
    return UNetHarmonizer(cfg)


# --------------------------------------------------------------------------- #
# Metrics: FOREGROUND-region PSNR / MSE / SSIM (measured ONLY where mask==1).
# --------------------------------------------------------------------------- #
def fg_psnr_batch(pred, gt, mask) -> float:
    pred = pred.clamp(0, 1).float(); gt = gt.clamp(0, 1).float()
    se = ((pred - gt) ** 2) * mask
    denom = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    mse = se.sum(dim=(1, 2, 3)) / denom
    mse = mse.clamp_min(1e-10)
    return float((10.0 * torch.log10(1.0 / mse)).mean())


def fg_mse_batch(pred, gt, mask) -> float:
    pred = pred.clamp(0, 1).float(); gt = gt.clamp(0, 1).float()
    se = ((pred - gt) ** 2) * mask
    denom = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return float((se.sum(dim=(1, 2, 3)) / denom).mean())


def _gaussian_window(ch, ks=11, sigma=1.5, device="cpu"):
    coords = torch.arange(ks, dtype=torch.float32, device=device) - (ks - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    return w.expand(ch, 1, ks, ks).contiguous()


def _ssim_map(pred, gt):
    pred = pred.clamp(0, 1).float(); gt = gt.clamp(0, 1).float()
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


def fg_ssim_batch(pred, gt, mask) -> float:
    smap = _ssim_map(pred, gt).mean(dim=1, keepdim=True)      # (B,1,H,W)
    denom = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return float(((smap * mask).sum(dim=(1, 2, 3)) / denom).mean())


# --------------------------------------------------------------------------- #
def load_surface(sol_path: Path):
    spec = importlib.util.spec_from_file_location("agent_surface", str(sol_path))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(sol_path.parent))
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# --------------------------------------------------------------------------- #
# Resolve config from the active surface's hook (all others FIXED to the strong default).
# --------------------------------------------------------------------------- #
def resolve_config(surface, mod) -> dict:
    cfg = default_config()

    def note(tag, val):
        print(f"{tag}_APPLIED {val}", flush=True)

    def fb(tag, e):
        print(f"{tag}_FALLBACK reason={e!r}", flush=True)

    if surface == "network":
        try:
            cand = mod.get_network_config()
            a = str(cand["arch"]).lower(); assert a in ("copy", "blind", "mask", "rain")
            cfg["arch"] = a
            if a == "blind":
                cfg["mask_cond"] = "none"
            note("NETWORK", {"arch": a})
        except Exception as e:  # noqa: BLE001
            fb("NETWORK", e)
    elif surface == "maskcond":
        try:
            mc = str(mod.get_mask_conditioning()).lower()
            assert mc in ("none", "concat", "gated")
            cfg["mask_cond"] = mc
            cfg["arch"] = "blind" if mc == "none" else "mask"
            note("MASKCOND", mc)
        except Exception as e:
            fb("MASKCOND", e)
    elif surface == "normalization":
        try:
            n = str(mod.get_normalization()).lower()
            assert n in ("none", "batch", "instance", "rain")
            cfg["norm"] = n; note("NORM", n)
        except Exception as e:
            fb("NORM", e)
    elif surface == "loss":
        try:
            lc = mod.get_loss_config()
            m = str(lc["mode"]).lower(); assert m in ("bg", "global", "fg")
            cfg["loss_mode"] = m; note("LOSS", lc)
        except Exception as e:
            fb("LOSS", e)
    elif surface == "fusion":
        try:
            fc = mod.get_fusion_config(); cfg["skips"] = bool(fc["skips"])
            note("FUSION", fc)
        except Exception as e:
            fb("FUSION", e)
    elif surface == "colorhead":
        try:
            ch = str(mod.get_color_head()).lower()
            assert ch in ("residual", "affine_global", "affine_spatial")
            cfg["color_head"] = ch; note("COLORHEAD", ch)
        except Exception as e:
            fb("COLORHEAD", e)
    elif surface == "upsampling":
        try:
            u = str(mod.get_upsampling()).lower()
            assert u in ("transpose", "nearest", "bilinear")
            cfg["upsampling"] = u; note("UPSAMPLING", u)
        except Exception as e:
            fb("UPSAMPLING", e)
    elif surface == "dilation":
        try:
            d = int(mod.get_dilation()); assert 1 <= d <= 8
            cfg["dilation"] = d; note("DILATION", d)
        except Exception as e:
            fb("DILATION", e)
    elif surface == "attention":
        try:
            ac = mod.get_attention_config(); cfg["attention"] = bool(ac["enabled"])
            note("ATTENTION", ac)
        except Exception as e:
            fb("ATTENTION", e)
    elif surface == "activation":
        try:
            a = str(mod.get_activation()).lower()
            assert a in ("relu", "identity", "gelu")
            cfg["activation"] = a; note("ACTIVATION", a)
        except Exception as e:
            fb("ACTIVATION", e)
    elif surface == "bgstats":
        try:
            bc = mod.get_bgstats_config(); cfg["bgstats"] = bool(bc["enabled"])
            note("BGSTATS", bc)
        except Exception as e:
            fb("BGSTATS", e)
    elif surface == "inputnorm":
        try:
            ic = str(mod.get_input_norm()).lower(); assert ic in ("none", "bg_whiten")
            cfg["input_norm"] = ic; note("INPUTNORM", ic)
        except Exception as e:
            fb("INPUTNORM", e)
    else:
        print(f"SURFACE_FIXED {cfg}", flush=True)
    return cfg


# --------------------------------------------------------------------------- #
# Background per-channel mean/std (for bgstats + inputnorm). Computed from the composite
# in the BACKGROUND region only (which is the untouched, already-correct image).
# --------------------------------------------------------------------------- #
def _bg_mean_std(comp, mask):
    bg = 1.0 - mask
    mean, std = _region_stats(comp, bg)
    return mean, std


# --------------------------------------------------------------------------- #
# Train + eval
# --------------------------------------------------------------------------- #
def run(surface, mod, data_root, severity, device, iters, seed):
    set_all_seeds(seed)
    c_tr, r_tr, m_tr = load_split(data_root, severity, "train")
    c_va, r_va, m_va = load_split(data_root, severity, "val")
    c_tr, r_tr, m_tr = c_tr.to(device), r_tr.to(device), m_tr.to(device)
    c_va, r_va, m_va = c_va.to(device), r_va.to(device), m_va.to(device)
    print(f"DATA severity={severity} train={c_tr.shape[0]} val={c_va.shape[0]} "
          f"img={tuple(c_tr.shape[-2:])}", flush=True)

    cfg = resolve_config(surface, mod)
    print(f"CONFIG {cfg}", flush=True)

    model = build_backbone(cfg)

    if model is None:
        # 'copy' identity: no training; the harmonized output IS the composite.
        print("NETWORK_COPY identity (no harmonization)", flush=True)
        pred = c_va
    else:
        model = model.to(device).train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        N = c_tr.shape[0]
        use_whiten = cfg["input_norm"] == "bg_whiten"
        use_bgstats = cfg["bgstats"]
        for it in range(iters):
            sel = torch.randint(0, N, (BS,), device=device)
            comp = c_tr[sel]; real = r_tr[sel]; mask = m_tr[sel]
            bg_mean = None
            net_in = comp
            if use_whiten or use_bgstats:
                bmean, bstd = _bg_mean_std(comp, mask)
                if use_bgstats:
                    bg_mean = bmean
                if use_whiten:
                    net_in = (comp - bmean) / (bstd + 1e-3)
            out = model(net_in, mask, bg_mean)
            if use_whiten:
                out = (out * (bstd + 1e-3) + bmean)           # invert whitening for the target
            out = out.clamp(0, 1)
            loss = _recon_loss(out, real, mask, cfg["loss_mode"])
            if not torch.isfinite(loss):
                loss = out.float().pow(2).mean() * 0.0 + 1.0
            opt.zero_grad(); loss.backward(); opt.step()
            if it % max(1, iters // 5) == 0 or it == iters - 1:
                print(f"train it={it} loss={float(loss):.5f}", flush=True)
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, c_va.shape[0], 64):
                comp = c_va[i:i + 64]; mask = m_va[i:i + 64]
                bg_mean = None; net_in = comp
                if use_whiten or use_bgstats:
                    bmean, bstd = _bg_mean_std(comp, mask)
                    if use_bgstats:
                        bg_mean = bmean
                    if use_whiten:
                        net_in = (comp - bmean) / (bstd + 1e-3)
                o = model(net_in, mask, bg_mean)
                if use_whiten:
                    o = o * (bstd + 1e-3) + bmean
                preds.append(o.clamp(0, 1))
        pred = torch.cat(preds, 0)

    pred = pred.clamp(0, 1).cpu()
    real = r_va.cpu(); comp = c_va.cpu(); mask = m_va.cpu()

    fg_psnr = fg_psnr_batch(pred, real, mask)
    comp_fg_psnr = fg_psnr_batch(comp, real, mask)      # do-nothing floor (composite input)
    fg_mse = fg_mse_batch(pred, real, mask)
    fg_ssim = fg_ssim_batch(pred, real, mask)
    return dict(fg_psnr=fg_psnr, comp_fg_psnr=comp_fg_psnr,
                fg_psnr_gain=fg_psnr - comp_fg_psnr, fg_mse=fg_mse, fg_ssim=fg_ssim)


def _recon_loss(out, real, mask, mode: str):
    """WHERE the reconstruction L1 is applied.
      bg     = supervise the (already-correct) BACKGROUND only -> no foreground signal
               (the net learns to copy the composite through: degenerate).
      global = whole-image L1.
      fg     = whole-image L1 + a FOREGROUND emphasis (the region that needs correcting).
    """
    bg = 1.0 - mask
    l_all = (out - real).abs().mean()
    l_fg = ((out - real).abs() * mask).sum() / mask.sum().clamp_min(1.0)
    l_bg = ((out - real).abs() * bg).sum() / bg.sum().clamp_min(1.0)
    if mode == "bg":
        return l_bg
    if mode == "global":
        return l_all
    return l_all + l_fg               # 'fg' (default strong)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--surface", required=True, choices=SURFACES)
    ap.add_argument("--severity", required=True, choices=["mild", "medium", "strong"])
    ap.add_argument("--label", default="run")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE {device} torch {torch.__version__}", flush=True)

    mod = load_surface(Path(args.solution))
    m = run(args.surface, mod, args.data_root, args.severity, device, args.iters, args.seed)

    print(f"HARMONY_METRICS surface={args.surface} setting={args.label} "
          f"fg_psnr={m['fg_psnr']:.4f} fg_psnr_gain={m['fg_psnr_gain']:.4f} "
          f"comp_fg_psnr={m['comp_fg_psnr']:.4f} fg_mse={m['fg_mse']:.6f} "
          f"fg_ssim={m['fg_ssim']:.4f}", flush=True)


if __name__ == "__main__":
    main()
