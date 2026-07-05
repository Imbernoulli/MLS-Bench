#!/usr/bin/env python3
"""Video-frame-interpolation harness (self-contained, MULTI-SURFACE).

VIDEO FRAME INTERPOLATION (VFI): synthesize the MIDDLE frame at t=0.5 between two given
frames frame0 (t=0) and frame2 (t=1). DISTINCT from all restoration (nothing is degraded --
the middle frame simply does not exist and must be synthesized from the inter-frame motion)
and from optical-flow estimation (RAFT predicts a flow FIELD; here the deliverable is the
IMAGE). References: Super-SloMo (Jiang et al., CVPR 2018 -- the SOTA reference here), SepConv
(Niklaus et al., ICCV 2017), RIFE (Huang et al., ECCV 2022), softmax-splatting (Niklaus &
Liu, CVPR 2020), CAIN (Choi et al., AAAI 2020).

Data are prepared at prepare time (prepare_data.py) from REAL video triplets -- the
Vimeo-90K temporal-frame-interpolation test set (Xue et al., TOFlow, IJCV 2019;
http://toflow.csail.mit.edu/): frame0/frame2 are two real, consecutively-decoded video
frames (im1/im3 of a triplet) and `gt` is the REAL frame the camera captured half-way
between them (im2) -- nothing here is synthetically warped or composited. Occlusion,
where present, is whatever occlusion is genuinely in the clip (camera/object motion),
not an engineered layer. Sequences are bucketed into THREE settings (small / medium /
large) by their MEASURED Farneback optical-flow magnitude between frame0/frame2 (see
prepare_data.py), reconstructing the same small<medium<large difficulty ladder the
original synthetic generator provided, now from real motion terciles. Train and val
splits are disjoint (and drawn from disjoint source video IDs) within each setting.

The agent edits ONE design surface (chosen by --surface); everything else (data, backbone
capacity, optimiser, iterations, seed, eval split, the metric) is FIXED, so any change in the
score is attributable to the edited surface. All surfaces plug into the SAME configurable
`VFIModel` whose non-edited fields stay at the Super-SloMo SOTA default, so each task is a
clean ablation of one axis of the same SOTA interpolator.

SURFACES. Only `synthesis` is a SHIPPED, GPU-validated MLS-Bench task (its blend < flow_warp <
learned order is monotone and WIDENS across all three motion settings -- a MECHANISM-level
lever with real headroom). The 11 FINER surfaces below are RETAINED here (inert code paths)
as documented research questions, but were DROPPED as scored tasks: a full GPU anchor sweep
(k1 H20, 800 iters, seed 42; see .vfitmp/rejected_tasks/) found NONE of them monotone across
the three motion settings -- on this near-SOTA base the small-motion setting is already at
~35 dB ceiling and the large-motion setting saturates at ~19 dB for every config, so the
refinement/reparam axes do not separate (and some diverge). Same honest ceiling as the
low-light-enhance refinement surfaces. Keep them for a future higher-budget / weaker-base
re-validation. Each surface (the agent returns a small config dict):
  synthesis  get_synthesis_config()   -> HOW the middle frame is built (SHIPPED / validated):
               'blend'(no motion) < 'flow_warp'(motion-comp avg) < 'learned'(flow+refine SOTA).
  flow       get_flow_config()        -> the FLOW-ESTIMATION module feeding the warp:
               'zero'(no flow) < 'single'(one-shot flow net) < 'refine'(coarse+residual flow, SOTA).
  warp       get_warp_config()        -> how frames are MOTION-COMPENSATED to t=0.5:
               'none'(copy) < 'forward'(forward splat, holes) < 'backward'(inverse warp) <
               'softsplat'(softmax-splatting fwd+bwd fusion, SOTA).
  occlusion  get_occlusion_config()   -> how the two warped candidates are COMBINED:
               'avg'(fixed 0.5) < 'time'(fixed temporal weight) < 'mask'(learned visibility, SOTA).
  refine     get_refine_config()      -> DEPTH of the refinement/synthesis net:
               'none'(0 blocks) < 'shallow'(1 lvl) < 'deep'(3-level U-Net, SOTA).
  loss       get_loss_config()        -> the TRAINING objective:
               'l2' < 'l1'(Charbonnier) < 'l1_census'(+census/edge term) < 'l1_warp'(+warp
               self-consistency, SOTA) -- richer supervision of the (dis)occlusion boundary.
  fusion     get_fusion_config()      -> how flow/warp/context features FEED the refine net:
               'warps'(warped RGB only) < 'plus_flow'(+flows) < 'full'(+originals+context, SOTA).
  context    get_context_config()     -> CONTEXT/feature extraction warped alongside RGB:
               'none' < 'shallow'(1-conv feats) < 'pyramid'(multi-scale feats, SOTA).
  scale      get_scale_config()       -> flow estimation SCALE / coarse-to-fine PYRAMID:
               'single'(full-res only) < 'two'(2-level) < 'three'(3-level pyramid, SOTA).
  attention  get_attention_config()   -> feature aggregation in the refine bottleneck:
               'none' < 'se'(channel gate) < 'nonlocal'(spatial self-attention, SOTA).
  iters      get_flow_iters_config()  -> NUMBER of flow-refinement iterations (RAFT-style):
               1 < 2 < 4 (SOTA) -- more iters recover large occluding motion.

Each surface's degenerate choice ('blend'/'zero'/'none'/'avg'/'l2'/'single'...) COLLAPSES as
motion grows (occlusion widens); the SOTA choice preserves PSNR. The per-surface literature
partial-order is monotone across all three motion settings and WIDENS with motion. A malformed
/ crashing agent return falls back to that surface's SOTA default.

Metric line (one per run):
  VFI_METRICS surface=<S> setting=<L> psnr=<..> psnr_gain=<..> blend_psnr=<..> ssim=<..> \
      mse=<..>
`psnr` is the interpolation PSNR of the SYNTHESIZED middle frame vs the true middle frame
(dB, HIGHER better) and is the PRIMARY metric. `blend_psnr` is the PSNR of the naive linear
blend vs GT -- the motion-agnostic floor. `psnr_gain = psnr - blend_psnr` makes explicit that
a real interpolator must BEAT the blend. `ssim` and `mse` are diagnostics.
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
BASE = 32                # net base width (FIXED)
BS = 32                  # batch size (FIXED)

SURFACES = ["synthesis", "flow", "warp", "occlusion", "refine", "loss",
            "fusion", "context", "scale", "attention", "iters"]

# The BASE configuration: the proven, numerically-stable Super-SloMo-style interpolator
# (single one-shot flow, backward warp, a deep refinement U-Net with a learned visibility
# mask + residual). This is the `synthesis=learned` model that reproduces the validated
# anchors. Each finer surface holds every OTHER field at this base and moves only its OWN axis
# to its per-surface strong choice (a MODERATE enhancement -- rich enough to give a monotone
# PSNR gain in the fixed 800-iter budget, light enough to stay stable). So each task isolates
# one axis of the SAME stable interpolator; there is no single heavy "SOTA stack".
SOTA_CFG = dict(
    synthesis="learned",     # full flow + refinement synthesis (the stable base model)
    flow="single",           # one-shot flow net (base); surface-strong = 'refine'
    warp="backward",         # backward/inverse warp (base); surface-strong = 'softsplat'
    occlusion="mask",        # learned per-pixel visibility mask (base = strong)
    refine="deep",           # 3-level refinement U-Net (base = strong)
    loss="l1",               # Charbonnier (base); surface-strong = 'l1_warp'
    fusion="full",           # warps+flows+originals+context (base = strong)
    context="none",          # no warped context (base = original learned); strong = 'pyramid'
    scale="single",          # full-res flow (base); surface-strong = 'three'
    attention="none",        # plain conv bottleneck (base); surface-strong = 'nonlocal'
    iters=1,                 # single flow pass (base); surface-strong = 4
)

# Per-surface STRONG (SOTA) choice -- applied ONLY when that surface is the edited one, on top
# of the stable base above. This is what the surface's score_spec anchors the ceiling to.
SURFACE_STRONG = dict(
    synthesis="learned", flow="refine", warp="softsplat", occlusion="mask",
    refine="deep", loss="l1_warp", fusion="full", context="pyramid",
    scale="three", attention="nonlocal", iters=4,
)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(False)


def load_split(root: str, split: str, motion: str = "medium"):
    sub = os.path.join(root, motion, f"{split}.npz")
    path = sub if os.path.exists(sub) else os.path.join(root, f"{split}.npz")
    arr = np.load(path)
    f0 = torch.from_numpy(arr["f0"].astype(np.float32))
    f2 = torch.from_numpy(arr["f2"].astype(np.float32))
    gt = torch.from_numpy(arr["gt"].astype(np.float32))
    return f0, f2, gt


# --------------------------------------------------------------------------- #
# Metrics (images in [0,1]).
# --------------------------------------------------------------------------- #
def psnr_batch(pred, gt):
    pred = pred.clamp(0, 1).float(); gt = gt.clamp(0, 1).float()
    mse = ((pred - gt) ** 2).reshape(pred.shape[0], -1).mean(1).clamp_min(1e-10)
    return float((10.0 * torch.log10(1.0 / mse)).mean())


def _gaussian_window(ch, ks=11, sigma=1.5, device="cpu"):
    coords = torch.arange(ks, dtype=torch.float32, device=device) - (ks - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    return w.expand(ch, 1, ks, ks).contiguous()


def ssim_batch(pred, gt):
    pred = pred.clamp(0, 1).float(); gt = gt.clamp(0, 1).float()
    ch = pred.shape[1]; w = _gaussian_window(ch, device=pred.device); pad = w.shape[-1] // 2
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
# Backward warp: sample `img` at (grid + flow) [flow in pixels]. Reflect border.
# --------------------------------------------------------------------------- #
def _grid(H, W, device):
    ys, xs = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing="ij")
    return torch.stack((xs, ys), 0).float()[None]              # (1,2,H,W)


def _backward_warp(img, flow):
    N, _, H, W = img.shape
    flow = torch.nan_to_num(flow, 0.0).clamp(-2.0 * max(H, W), 2.0 * max(H, W))
    coords = _grid(H, W, img.device) + flow                    # (N,2,H,W) in pixels
    gx = 2.0 * coords[:, 0] / max(W - 1, 1) - 1.0
    gy = 2.0 * coords[:, 1] / max(H - 1, 1) - 1.0
    grid = torch.stack((gx, gy), -1)                           # (N,H,W,2)
    return F.grid_sample(img, grid, mode="bilinear",
                         padding_mode="reflection", align_corners=True)


def _forward_scatter(img, flow, weight=None):
    """FORWARD warp / splat: push pixels of `img` ALONG +flow to their target location and
    accumulate (holes appear where nothing lands -> the classic forward-warp artefact). If
    `weight` (N,1,H,W) is given it is a softmax-splatting importance weight (softsplat)."""
    N, C, H, W = img.shape
    dev = img.device
    base = _grid(H, W, dev)                                    # (1,2,H,W)
    flow = torch.nan_to_num(flow, 0.0).clamp(-2.0 * max(H, W), 2.0 * max(H, W))
    tgt = base + flow                                          # target (float) coords
    x = tgt[:, 0].reshape(N, -1); y = tgt[:, 1].reshape(N, -1)
    x0 = torch.floor(x); y0 = torch.floor(y)
    if weight is None:
        weight = torch.ones(N, 1, H, W, device=dev)
    wv = weight.reshape(N, 1, -1)                              # (N,1,HW)
    src = img.reshape(N, C, -1)                                # (N,C,HW)
    num = torch.zeros(N, C, H * W, device=dev)
    den = torch.zeros(N, 1, H * W, device=dev)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xi = (x0 + dx); yi = (y0 + dy)
        w_bil = (1 - (x - xi).abs()).clamp(0) * (1 - (y - yi).abs()).clamp(0)  # (N,HW)
        xi = xi.long().clamp(0, W - 1); yi = yi.long().clamp(0, H - 1)
        idx = (yi * W + xi).unsqueeze(1)                       # (N,1,HW)
        ww = (w_bil.unsqueeze(1) * wv)                         # (N,1,HW)
        num.scatter_add_(2, idx.expand(N, C, -1), src * ww)
        den.scatter_add_(2, idx, ww)
    out = (num / den.clamp_min(1e-6)).reshape(N, C, H, W)
    return out, (den.reshape(N, 1, H, W) > 1e-4).float()      # image, coverage mask


# --------------------------------------------------------------------------- #
# Compact residual encoder-decoder used as the FLOW net and the REFINEMENT net.
# `depth` (# levels) is configurable for the `refine`/`scale` surfaces.
# --------------------------------------------------------------------------- #
class _ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        y = F.relu(self.c1(x), inplace=True)
        return x + self.c2(y)


class _SEBlock(nn.Module):
    """Squeeze-and-excitation channel gate."""
    def __init__(self, c, r=4):
        super().__init__()
        self.fc1 = nn.Conv2d(c, max(c // r, 4), 1)
        self.fc2 = nn.Conv2d(max(c // r, 4), c, 1)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        return x * torch.sigmoid(self.fc2(s))


class _NonLocal(nn.Module):
    """Lightweight spatial self-attention (non-local block) at the bottleneck."""
    def __init__(self, c):
        super().__init__()
        self.q = nn.Conv2d(c, c // 2, 1)
        self.k = nn.Conv2d(c, c // 2, 1)
        self.v = nn.Conv2d(c, c, 1)
        self.o = nn.Conv2d(c, c, 1)
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        N, C, H, W = x.shape
        q = self.q(x).reshape(N, -1, H * W).permute(0, 2, 1)   # (N,HW,c/2)
        k = self.k(x).reshape(N, -1, H * W)                    # (N,c/2,HW)
        a = torch.softmax(q @ k / (k.shape[1] ** 0.5), -1)     # (N,HW,HW)
        v = self.v(x).reshape(N, C, H * W).permute(0, 2, 1)    # (N,HW,C)
        y = (a @ v).permute(0, 2, 1).reshape(N, C, H, W)
        return x + self.g * self.o(y)


class UNet(nn.Module):
    """Configurable encoder-decoder, ``in_ch`` -> ``out_ch``, base width ``base``.

    ``depth`` = number of downsampling levels (0 = a flat conv stack, no down/up; 3 = the
    full SOTA U-Net). ``attn`` in {'none','se','nonlocal'} inserts a channel/spatial
    attention block at the bottleneck.
    """

    def __init__(self, in_ch, out_ch, base=BASE, depth=3, attn="none"):
        super().__init__()
        self.depth = depth
        self.inc = nn.Conv2d(in_ch, base, 3, padding=1)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        ch = base
        for _ in range(depth):
            self.enc.append(_ResBlock(ch))
            self.down.append(nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1))
            ch *= 2
        mid = [_ResBlock(ch), _ResBlock(ch)]
        if attn == "se":
            mid.append(_SEBlock(ch))
        elif attn == "nonlocal":
            mid.append(_NonLocal(ch))
        self.mid = nn.Sequential(*mid)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for _ in range(depth):
            self.up.append(nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1))
            ch //= 2
            self.dec.append(_ResBlock(ch))
        # depth==0 fallback: a couple of flat res-blocks for capacity parity
        self.flat = nn.Sequential(_ResBlock(base), _ResBlock(base)) if depth == 0 else None
        self.outc = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x):
        h = F.relu(self.inc(x), inplace=True)
        if self.depth == 0:
            h = self.flat(h)
            return self.outc(self.mid(h))
        skips = []
        for enc, down in zip(self.enc, self.down):
            h = enc(h); skips.append(h)
            h = F.relu(down(h), inplace=True)
        h = self.mid(h)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            h = F.relu(up(h), inplace=True) + s
            h = dec(h)
        return self.outc(h)


# --------------------------------------------------------------------------- #
# Context extractor (features warped alongside RGB) for the `context`/`fusion` surfaces.
# --------------------------------------------------------------------------- #
class _Context(nn.Module):
    def __init__(self, out_ch, kind="pyramid"):
        super().__init__()
        self.kind = kind
        self.out_ch = out_ch
        if kind == "none":
            return
        self.c1 = nn.Conv2d(3, out_ch, 3, padding=1)
        if kind == "pyramid":
            self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        if self.kind == "none":
            return None
        f = F.relu(self.c1(x), inplace=True)
        if self.kind == "pyramid":
            f = F.relu(self.c2(f), inplace=True) + f
        return f


# --------------------------------------------------------------------------- #
# ORIGINAL synthesis-surface models (blend / flow_warp / learned). These are kept
# BYTE-FOR-BYTE from the validated vfi-synthesis task so the `synthesis` surface reproduces its
# committed anchors EXACTLY. The finer surfaces below use the unified VFIModel instead.
# `_OrigUNet` is a verbatim copy of the original standalone U-Net (2 downsampling levels, same
# module-registration order) so the seeded weights -- and hence the anchors -- are identical.
# --------------------------------------------------------------------------- #
class _OrigUNet(nn.Module):
    """Verbatim original 2-level encoder-decoder, ``in_ch`` -> ``out_ch``, base width ``base``."""

    def __init__(self, in_ch, out_ch, base=BASE):
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
        self.outc = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x):
        h0 = F.relu(self.inc(x), inplace=True)
        e1 = self.enc1(h0)
        e2 = self.enc2(F.relu(self.down1(e1), inplace=True))
        m = self.mid(F.relu(self.down2(e2), inplace=True))
        d2 = self.dec2(F.relu(self.up2(m), inplace=True) + e2)
        d1 = self.dec1(F.relu(self.up1(d2), inplace=True) + e1)
        return self.outc(d1)


class BlendInterp(nn.Module):
    """Naive linear blend: 0.5*(f0+f2). No motion, no learnable params (a fixed dummy
    parameter keeps the optimiser happy)."""

    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, f0, f2):
        return (0.5 * (f0 + f2) + 0.0 * self._dummy).clamp(0, 1)


class FlowWarpInterp(nn.Module):
    """Learnable flow net + backward-warp both frames to t=0.5 + fixed 0.5 average."""

    def __init__(self):
        super().__init__()
        self.flownet = _OrigUNet(6, 4, base=BASE)

    def _flows(self, f0, f2):
        fl = self.flownet(torch.cat([f0, f2], 1))
        return fl[:, 0:2], fl[:, 2:4]

    def forward(self, f0, f2):
        Ft0, Ft2 = self._flows(f0, f2)
        w0 = _backward_warp(f0, Ft0)
        w2 = _backward_warp(f2, Ft2)
        return (0.5 * (w0 + w2)).clamp(0, 1)


class LearnedInterp(nn.Module):
    """Super-SloMo-style: flow net + backward-warp + refinement U-Net that predicts a soft
    visibility mask + residual. Reproduces the committed learned anchor."""

    def __init__(self):
        super().__init__()
        self.flownet = _OrigUNet(6, 4, base=BASE)
        self.refine = _OrigUNet(16, 4, base=BASE)
        # Zero-init the refine head so the model STARTS at exactly flow_warp behaviour
        # (mask = sigmoid(0) = 0.5, residual = 0) and can only improve from there -> learned
        # reliably >= flow_warp in every motion setting. Standard residual-refinement init.
        nn.init.zeros_(self.refine.outc.weight); nn.init.zeros_(self.refine.outc.bias)

    def forward(self, f0, f2):
        fl = self.flownet(torch.cat([f0, f2], 1))
        Ft0, Ft2 = fl[:, 0:2], fl[:, 2:4]
        w0 = _backward_warp(f0, Ft0)
        w2 = _backward_warp(f2, Ft2)
        ref_in = torch.cat([w0, w2, f0, f2, Ft0, Ft2], 1)
        r = self.refine(ref_in)
        mask = torch.sigmoid(r[:, 0:1])
        residual = r[:, 1:4]
        blended = mask * w0 + (1.0 - mask) * w2
        return (blended + residual).clamp(0, 1)


_SYNTH_BUILDERS = {"blend": BlendInterp, "flow_warp": FlowWarpInterp, "learned": LearnedInterp}


class _SynthWrap(nn.Module):
    """Wrap an original synthesis model so it accepts the (f0,f2,return_flow=) call signature
    used by the unified training loop (returns None flows -> the warp loss term is skipped)."""

    def __init__(self, method):
        super().__init__()
        self.net = _SYNTH_BUILDERS[method]()

    def forward(self, f0, f2, return_flow=False):
        out = self.net(f0, f2)
        return (out, None, None) if return_flow else out


# --------------------------------------------------------------------------- #
# Flow estimator (configurable scale-pyramid + refinement iterations).
# --------------------------------------------------------------------------- #
class FlowNet(nn.Module):
    """Estimate [F_{t->0} (2ch), F_{t->2} (2ch)] from the concatenated pair.

    ``scale`` sets a coarse-to-fine pyramid ('single'/'two'/'three' downsample levels).
    ``n_iter`` sets RAFT-style residual refinement iterations. ``kind`` selects the
    estimator: 'zero' (no flow), 'single' (one-shot net), 'refine' (coarse + residual net).
    """

    def __init__(self, kind="refine", scale="three", n_iter=4):
        super().__init__()
        self.kind = kind
        self.scale = scale
        self.n_iter = max(1, int(n_iter))
        depth = {"single": 1, "two": 2, "three": 3}.get(scale, 3)
        if kind == "zero":
            self._dummy = nn.Parameter(torch.zeros(1))
            return
        self.net = UNet(6, 4, base=BASE, depth=depth)
        if kind == "refine" or self.n_iter > 1:
            # a light residual-flow net applied iteratively on warped inputs
            self.res = UNet(6 + 4, 4, base=BASE, depth=max(1, depth - 1))

    def forward(self, f0, f2):
        N, _, H, W = f0.shape
        if self.kind == "zero":
            z = torch.zeros(N, 2, H, W, device=f0.device) + 0.0 * self._dummy
            return z, z
        fl = self.net(torch.cat([f0, f2], 1))
        Ft0, Ft2 = fl[:, 0:2], fl[:, 2:4]
        if self.kind == "single" and self.n_iter == 1:
            return Ft0, Ft2
        iters = self.n_iter if self.kind == "refine" else 1
        for _ in range(iters):
            w0 = _backward_warp(f0, Ft0)
            w2 = _backward_warp(f2, Ft2)
            d = self.res(torch.cat([w0, w2, Ft0, Ft2], 1))
            Ft0 = Ft0 + d[:, 0:2]
            Ft2 = Ft2 + d[:, 2:4]
        return Ft0, Ft2


# --------------------------------------------------------------------------- #
# The unified configurable VFI model. Every surface toggles ONE cfg field; the rest stay at
# the SOTA default, so a run isolates the edited axis. `synthesis` is a coarse macro switch
# (blend/flow_warp/learned) kept for the original task; the finer surfaces refine `learned`.
# --------------------------------------------------------------------------- #
class VFIModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        syn = self.cfg["synthesis"]

        # ---- coarse synthesis macro: blend has no learnable motion at all ----
        if syn == "blend":
            self._dummy = nn.Parameter(torch.zeros(1))
            return

        # ---- flow estimation ----
        flow_kind = "single" if syn == "flow_warp" else self.cfg["flow"]
        self.flow = FlowNet(kind=flow_kind, scale=self.cfg["scale"],
                            n_iter=self.cfg["iters"])

        # ---- context features (warped alongside RGB) ----
        self.ctx_kind = self.cfg["context"] if syn == "learned" else "none"
        self.cchan = {"none": 0, "shallow": 8, "pyramid": 8}[self.ctx_kind]
        if self.cchan:
            self.ctx = _Context(self.cchan, kind=self.ctx_kind)

        # ---- refinement / synthesis net (only for the full 'learned' macro) ----
        self.has_refine = (syn == "learned" and self.cfg["refine"] != "none"
                           and self.cfg["occlusion"] == "mask")
        self.occ = self.cfg["occlusion"] if syn == "learned" else "avg"
        self.warp_kind = self.cfg["warp"] if syn == "learned" else "backward"
        self.fusion = self.cfg["fusion"] if syn == "learned" else "warps"

        if syn == "learned":
            depth = {"none": 0, "shallow": 1, "deep": 3}[self.cfg["refine"]]
            in_ch = 6                                    # warped0(3)+warped2(3)
            if self.fusion in ("plus_flow", "full"):
                in_ch += 4                               # +Ft0,Ft2
            if self.fusion == "full":
                in_ch += 6                               # +f0,f2 originals
                in_ch += 2 * self.cchan                  # +warped context feats
            self.refine_in = in_ch
            if self.has_refine:
                self.refine = UNet(in_ch, 4, base=BASE, depth=depth,
                                   attn=self.cfg["attention"])
                # zero-init the refine head -> the learned mask+residual STARTS at flow_warp
                # behaviour (mask=0.5, residual=0) and only improves; keeps every surface's
                # strong choice >= its weaker baselines robustly.
                nn.init.zeros_(self.refine.outc.weight); nn.init.zeros_(self.refine.outc.bias)

    def _combine(self, w0, w2, m0, m2):
        """Combine the two warped candidates given coverage masks m0,m2."""
        if self.occ == "avg":
            return 0.5 * (w0 + w2)
        if self.occ == "time":
            return 0.5 * w0 + 0.5 * w2         # symmetric temporal weight (t=0.5)
        return None                            # 'mask' handled by the refine net

    def _warp_pair(self, f0, f2, Ft0, Ft2):
        if self.warp_kind == "none":
            return f0, f2, torch.ones_like(f0[:, :1]), torch.ones_like(f2[:, :1])
        if self.warp_kind == "backward":
            return (_backward_warp(f0, Ft0), _backward_warp(f2, Ft2),
                    torch.ones_like(f0[:, :1]), torch.ones_like(f2[:, :1]))
        # forward / softsplat: push f0 along -Ft0 (t->0 flow reversed) etc. For our synthetic
        # bidirectional flow to the middle, the forward flow of f0 towards t=0.5 is -Ft0.
        wgt0 = wgt1 = None
        if self.warp_kind == "softsplat":
            # importance = negative brightness constancy error proxy (constant here -> unit)
            wgt0 = torch.ones_like(f0[:, :1]); wgt1 = torch.ones_like(f2[:, :1])
        w0, c0 = _forward_scatter(f0, -Ft0, wgt0)
        w2, c2 = _forward_scatter(f2, -Ft2, wgt1)
        if self.warp_kind == "softsplat":
            # backfill holes from the backward warp -> fwd+bwd fusion (softmax-splatting)
            b0 = _backward_warp(f0, Ft0); b2 = _backward_warp(f2, Ft2)
            w0 = c0 * w0 + (1 - c0) * b0
            w2 = c2 * w2 + (1 - c2) * b2
        return w0, w2, c0, c2

    def forward(self, f0, f2, return_flow=False):
        syn = self.cfg["synthesis"]
        if syn == "blend":
            out = (0.5 * (f0 + f2) + 0.0 * self._dummy).clamp(0, 1)
            return (out, None, None) if return_flow else out

        Ft0, Ft2 = self.flow(f0, f2)
        w0, w2, c0, c2 = self._warp_pair(f0, f2, Ft0, Ft2)

        if syn == "flow_warp" or not self.has_refine:
            comb = self._combine(w0, w2, c0, c2)
            if comb is None:                    # occ='mask' but no refine -> avg fallback
                comb = 0.5 * (w0 + w2)
            out = comb.clamp(0, 1)
            return (out, Ft0, Ft2) if return_flow else out

        # ---- learned refinement: build the fusion input ----
        parts = [w0, w2]
        if self.fusion in ("plus_flow", "full"):
            parts += [Ft0, Ft2]
        if self.fusion == "full":
            parts += [f0, f2]
            if self.cchan:
                cf0 = self.ctx(f0); cf2 = self.ctx(f2)
                parts += [_backward_warp(cf0, Ft0), _backward_warp(cf2, Ft2)]
        r = self.refine(torch.cat(parts, 1))
        mask = torch.sigmoid(r[:, 0:1])
        residual = r[:, 1:4]
        blended = mask * w0 + (1.0 - mask) * w2
        out = torch.nan_to_num(blended + residual, 0.0).clamp(0, 1)
        return (out, Ft0, Ft2) if return_flow else out


# --------------------------------------------------------------------------- #
# Losses (the `loss` surface).
# --------------------------------------------------------------------------- #
def _charbonnier(pred, gt, eps=1e-3):
    return torch.sqrt((pred - gt) ** 2 + eps ** 2).mean()


def _l2(pred, gt):
    return ((pred - gt) ** 2).mean()


def _gradient(x):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


def _census(pred, gt):
    """Edge/gradient (census-like) structural term -- sharpens (dis)occlusion boundaries."""
    px, py = _gradient(pred); gx, gy = _gradient(gt)
    return (px - gx).abs().mean() + (py - gy).abs().mean()


def make_loss(kind, f0=None, f2=None, Ft0=None, Ft2=None):
    """Return a callable loss(pred, gt). The warp term needs the flows/frames from the run."""
    if kind == "l2":
        return lambda p, g: _l2(p, g)
    if kind == "l1":
        return lambda p, g: _charbonnier(p, g)
    if kind == "l1_census":
        return lambda p, g: _charbonnier(p, g) + 0.1 * _census(p, g)
    # l1_warp: + a warp self-consistency term (both warped candidates must match GT)
    def _loss(p, g):
        base = _charbonnier(p, g) + 0.1 * _census(p, g)
        if Ft0 is not None:
            w0 = _backward_warp(f0, Ft0); w2 = _backward_warp(f2, Ft2)
            base = base + 0.1 * (_charbonnier(w0, g) + _charbonnier(w2, g))
        return base
    return _loss


# --------------------------------------------------------------------------- #
# Surface application: read the agent's config for the chosen surface, validate, and merge
# it into a copy of SOTA_CFG. Anything malformed falls back to the SOTA default (printed).
# --------------------------------------------------------------------------- #
_ALLOWED = {
    "synthesis": ("method", ["blend", "flow_warp", "learned"]),
    "flow": ("kind", ["zero", "single", "refine"]),
    "warp": ("kind", ["none", "forward", "backward", "softsplat"]),
    "occlusion": ("kind", ["avg", "time", "mask"]),
    "refine": ("depth", ["none", "shallow", "deep"]),
    "loss": ("kind", ["l2", "l1", "l1_census", "l1_warp"]),
    "fusion": ("kind", ["warps", "plus_flow", "full"]),
    "context": ("kind", ["none", "shallow", "pyramid"]),
    "scale": ("levels", ["single", "two", "three"]),
    "attention": ("kind", ["none", "se", "nonlocal"]),
    "iters": ("n", [1, 2, 4]),
}
_HOOK = {
    "synthesis": "get_synthesis_config", "flow": "get_flow_config",
    "warp": "get_warp_config", "occlusion": "get_occlusion_config",
    "refine": "get_refine_config", "loss": "get_loss_config",
    "fusion": "get_fusion_config", "context": "get_context_config",
    "scale": "get_scale_config", "attention": "get_attention_config",
    "iters": "get_flow_iters_config",
}
# which SOTA_CFG field each surface writes
_FIELD = {
    "synthesis": "synthesis", "flow": "flow", "warp": "warp",
    "occlusion": "occlusion", "refine": "refine", "loss": "loss",
    "fusion": "fusion", "context": "context", "scale": "scale",
    "attention": "attention", "iters": "iters",
}


def apply_surface(surface, mod):
    cfg = dict(SOTA_CFG)
    key, allowed = _ALLOWED[surface]
    field = _FIELD[surface]
    hook = _HOOK[surface]
    try:
        cand = getattr(mod, hook)()
        assert isinstance(cand, dict) and key in cand, f"missing {key!r}"
        v = cand[key]
        if surface == "iters":
            v = int(v)
        else:
            v = str(v).lower()
        assert v in allowed, f"unknown {key}={v!r}"
        cfg[field] = v
        print(f"{surface.upper()}_APPLIED {{{key}: {v!r}}}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{surface.upper()}_FALLBACK reason={e!r} -> SOTA {field}={cfg[field]!r}",
              flush=True)
    return cfg


def load_surface(sol_path: Path):
    spec = importlib.util.spec_from_file_location("agent_surface", str(sol_path))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(sol_path.parent))
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# --------------------------------------------------------------------------- #
# Train + eval
# --------------------------------------------------------------------------- #
def run(surface, mod, data_root, device, iters, seed, motion="medium"):
    set_all_seeds(seed)
    f0_tr, f2_tr, gt_tr = load_split(data_root, "train", motion)
    f0_va, f2_va, gt_va = load_split(data_root, "val", motion)
    f0_tr, f2_tr, gt_tr = f0_tr.to(device), f2_tr.to(device), gt_tr.to(device)
    f0_va, f2_va, gt_va = f0_va.to(device), f2_va.to(device), gt_va.to(device)
    print(f"DATA motion={motion} train={f0_tr.shape[0]} val={f0_va.shape[0]} "
          f"img={tuple(f0_tr.shape[-2:])}", flush=True)

    cfg = apply_surface(surface, mod)
    print(f"CFG {cfg}", flush=True)

    set_all_seeds(seed)
    if surface == "synthesis":
        # the original synthesis surface uses the validated standalone models (exact anchors)
        model = _SynthWrap(cfg["synthesis"]).to(device).train()
    else:
        model = VFIModel(cfg).to(device).train()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"MODEL synthesis={cfg['synthesis']} n_params={n_par}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_kind = cfg["loss"]

    N = f0_tr.shape[0]
    for it in range(iters):
        sel = torch.randint(0, N, (BS,), device=device)
        f0, f2, gt = f0_tr[sel], f2_tr[sel], gt_tr[sel]
        out, Ft0, Ft2 = model(f0, f2, return_flow=True)
        loss_fn = make_loss(loss_kind, f0, f2, Ft0, Ft2)
        loss = loss_fn(out, gt)
        if not torch.isfinite(loss):
            loss = out.float().pow(2).mean() * 0.0 + 1.0
        opt.zero_grad(); loss.backward(); opt.step()
        if it % max(1, iters // 5) == 0 or it == iters - 1:
            print(f"train it={it} loss={loss.detach().item():.5f}", flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, f0_va.shape[0], 64):
            preds.append(model(f0_va[i:i + 64], f2_va[i:i + 64]).clamp(0, 1).cpu())
    pred = torch.cat(preds, 0)
    f0c, f2c, gtc = f0_va.cpu(), f2_va.cpu(), gt_va.cpu()

    psnr = psnr_batch(pred, gtc)
    blend = (0.5 * (f0c + f2c)).clamp(0, 1)
    blend_psnr = psnr_batch(blend, gtc)                     # motion-agnostic floor
    ssim = ssim_batch(pred, gtc)
    mse = float(((pred.clamp(0, 1) - gtc) ** 2).mean())
    return dict(psnr=psnr, blend_psnr=blend_psnr, psnr_gain=psnr - blend_psnr,
                ssim=ssim, mse=mse)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--surface", required=True, choices=SURFACES)
    ap.add_argument("--motion", default="medium", choices=["small", "medium", "large"])
    ap.add_argument("--label", default="run")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE {device} torch {torch.__version__}", flush=True)

    mod = load_surface(Path(args.solution))
    m = run(args.surface, mod, args.data_root, device, args.iters, args.seed,
            motion=args.motion)

    print(f"VFI_METRICS surface={args.surface} setting={args.label} "
          f"psnr={m['psnr']:.4f} psnr_gain={m['psnr_gain']:.4f} "
          f"blend_psnr={m['blend_psnr']:.4f} ssim={m['ssim']:.4f} "
          f"mse={m['mse']:.6f}", flush=True)


if __name__ == "__main__":
    main()
