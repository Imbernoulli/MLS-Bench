#!/usr/bin/env python3
"""Trimap-guided image-matting harness (self-contained).

Predict a SOFT foreground alpha matte in [0,1] per pixel (NOT a hard mask) for a
trimap-guided composite, on a SYNTHETIC composite dataset so the ground-truth
alpha is EXACT: each image is I = alpha*F + (1-alpha)*B with a rendered soft alpha
(anti-aliased + blurred shape, plus fine hair-like structure) and a derived trimap
(definite-fg / definite-bg / UNKNOWN band). The standard matting metrics (SAD / MSE
/ gradient error, all computed ONLY in the trimap UNKNOWN region, LOWER is better)
are exact and clean.

THREE VAL TRIMAP-WIDTH SETTINGS (the >=3 validation settings; the baseline
partial-order must hold across all three). The trimap is RE-DERIVED from the exact
stored GT alpha at eval time by eroding the solid-fg / solid-bg regions by a chosen
width, so we score the SAME images under three difficulties:
  medium  -> moderate UNKNOWN band  (band width 6, unk_frac ~0.39)
  wide    -> wider band             (band width 9, unk_frac ~0.47)
  xwide   -> thick UNKNOWN band      (band width 12, largest soft transition to
             solve, hardest; unk_frac ~0.54)
The training trimap width is FIXED (medium); only the SCORED val band changes. (Two
regimes were excluded after measurement: a NARROW band (width 2) is too easy — every
method saturates and second-order design choices invert; and an EXTREME band
(width >=16, unk_frac >0.65) is so hard that refinement/attention add noise and the
order flattens/inverts. The order holds cleanly across these three moderate-to-wide
bands.)

Surfaces (one per task, --surface). Each surface is an editable HOOK into the SAME
fixed harness/data/eval; only the named component is swapped, everything else is
FIXED. The `arch` surface replaces the WHOLE network (build_net); the `refine`
surface wraps a fixed coarse stage with an agent second stage; every other surface
plugs ONE component into a FIXED configurable matting U-Net (ConfigMattingNet) so
the RQ is isolated.

  arch    -> build_net(in_ch) : the WHOLE matting network (RGB+trimap -> 1-ch alpha).
             copy-trimap / constant (degenerate) < plain encoder-decoder <
             DIM deep-matting (encoder-decoder + skip + refinement, Xu et al. 2017 =
             SOTA). This is the STRICT-BAR eval task.

  loss    -> get_matting_loss() : the ALPHA-MATTE TRAINING LOSS.
             whole-image L1 < unknown-band L1 < unknown + composition + gradient
             (Deep Image Matting, Xu et al. 2017 = SOTA).

  trimap  -> encode_trimap(trimap) : how the trimap becomes extra input channels.
             all-zero (trimap-blind) < raw channel < 3-plane one-hot fg/unk/bg.

  decoder -> build_decoder(enc_channels) : the decoder head mapping encoder features
             to a 1-ch alpha. deepest-feature bilinear < U-Net skip-connection decoder.

  refine  -> refine(coarse_alpha, x, trimap) : a SECOND-STAGE alpha refinement of a
             fixed coarse matte. identity < shallow refine < full refinement stage
             (Deep Image Matting stage-2, Xu et al. 2017 = SOTA).

  skip    -> fuse(dec_up, skip) : how a decoder feature is FUSED with its encoder
             skip. drop-skip < half-strength skip < full concat skip (U-Net).

  attention -> build_attention(ch) : a module at the bottleneck (same-shape).
             global-avg-pool (destroys spatial context) < local 3x3 conv <
             non-local self-attention / guided context aggregation (SOTA).

  dilation -> build_dilation(ch) : the bottleneck receptive-field block.
             single 3x3 (over-processes) < pointwise < dilated multi-rate block
             (ASPP-style context, cf. Chen et al. 2017 / Iizuka 2017).

  norm    -> make_norm(num_ch) : the normalisation layer after each conv.
             identity < instance-norm < batch-norm (SOTA on this recurring data).

  upsampling -> build_upsampler(cin) : the decoder UPSAMPLING operator.
             nearest (blocky, loses soft edges) < bilinear < learned transposed-conv
             / guided upsample (SOTA — sharpest soft matte).

  propagation -> propagate(alpha, image, trimap) : a post-decoder alpha-PROPAGATION
             module that refines the matte using image affinity (guided-filter /
             matting-Laplacian style, Levin et al. 2008).
             identity (no propagation) < box-smooth < image-guided filter (SOTA).

  fgpred  -> build_head(cin) : the OUTPUT HEAD. alpha-only 1-ch head <
             joint alpha + foreground-colour prediction head (auxiliary FG
             supervision regularises the matte, Context-Aware Matting Hou 2019 =
             SOTA). Only the alpha channel is scored; the FG head is an aux head.

Metric line (one per run):
    MATTING_METRICS surface=<S> setting=<L> sad=<..> mse=<..> grad=<..> unk_frac=<..>
sad (sum-of-absolute-alpha-differences /1000, in the unknown band) is the PRIMARY
metric, LOWER is better. mse (alpha MSE *1e3) and grad (alpha gradient error /1000)
are secondary. All are measured in the trimap UNKNOWN region only.

A DEGENERATE predictor (constant 0.5, or copy-the-trimap unknown value 0.5) scores
CONST_HALF_SAD; the per-image mean-alpha predictor scores MEAN_ALPHA_SAD (both
printed by the harness). Because the GT alpha in the unknown band is genuinely soft
and spans a range, both are far above any real matting net -> the metric is monotone
in matting quality and a trivial output is clearly beaten.

Every hook is wrapped so a malformed / crashing return falls back to a sane default.
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

# stored channel layout (see data_scripts/image-matting/prepare_data.py)
C_I = slice(0, 3)
C_TRI = 3
C_ALPHA = 4
C_F = slice(5, 8)
C_B = slice(8, 11)

# fixed protocol
TRAIN_TRIMAP_WIDTH = 6          # the FIXED training trimap erosion width (medium)
TRIMAP_WIDTHS = {"medium": 6, "wide": 9, "xwide": 12}
BASE = 32                       # encoder base channels

# surfaces that replace the WHOLE net vs. plug ONE component into ConfigMattingNet
ARCH_SURFACES = {"arch"}
LOSS_SURFACES = {"loss"}
TRIMAP_SURFACES = {"trimap"}
DECODER_SURFACES = {"decoder"}
REFINE_SURFACES = {"refine"}
COMPONENT_SURFACES = {"skip", "attention", "dilation", "norm", "upsampling",
                      "propagation", "fgpred"}
ALL_SURFACES = (ARCH_SURFACES | LOSS_SURFACES | TRIMAP_SURFACES | DECODER_SURFACES
                | REFINE_SURFACES | COMPONENT_SURFACES)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------- #
# Trimap re-derivation (band width -> difficulty setting)
# --------------------------------------------------------------------------- #
def _binary_dilate_t(mask: torch.Tensor, iters: int) -> torch.Tensor:
    """4-neighbour binary dilation of a (H,W) bool tensor, `iters` steps."""
    m = mask.clone()
    for _ in range(iters):
        s = m.clone()
        s[1:, :] |= m[:-1, :]
        s[:-1, :] |= m[1:, :]
        s[:, 1:] |= m[:, :-1]
        s[:, :-1] |= m[:, 1:]
        m = s
    return m


def derive_trimap(alpha: torch.Tensor, width: int) -> torch.Tensor:
    """Re-derive a trimap in {0,0.5,1} from an exact GT alpha (H,W) by eroding the
    solid-fg / solid-bg regions with the given band width. Larger width -> thicker
    UNKNOWN band -> harder setting."""
    solid_fg = alpha >= 0.999
    solid_bg = alpha <= 0.001
    not_solid = ~(solid_fg | solid_bg)
    band = _binary_dilate_t(not_solid, width)
    tri = torch.full_like(alpha, 0.5)
    tri[solid_fg & ~band] = 1.0
    tri[solid_bg & ~band] = 0.0
    # guarantee a non-trivial unknown region
    if (tri == 0.5).sum() < 50:
        band = _binary_dilate_t(not_solid, max(width, 8))
        tri = torch.full_like(alpha, 0.5)
        tri[solid_fg & ~band] = 1.0
        tri[solid_bg & ~band] = 0.0
    return tri


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_split(root: str, split: str, trimap_width: int):
    """Return list of dicts with tensors: image(3,H,W), trimap(H,W), alpha(H,W),
    fg(3,H,W), bg(3,H,W), unknown(bool H,W). The trimap/unknown are re-derived at
    the requested `trimap_width` from the exact stored GT alpha."""
    d = os.path.join(root, split)
    with open(os.path.join(d, "manifest.json")) as f:
        items = json.load(f)
    out = []
    for it in items:
        arr = np.load(os.path.join(d, it["img"])).astype(np.float32)
        alpha = torch.from_numpy(arr[C_ALPHA].copy())
        tri = derive_trimap(alpha, trimap_width)
        out.append(dict(
            image=torch.from_numpy(arr[C_I].copy()),
            trimap=tri,
            alpha=alpha,
            fg=torch.from_numpy(arr[C_F].copy()),
            bg=torch.from_numpy(arr[C_B].copy()),
            unknown=(tri == 0.5),
        ))
    return out


def load_surface(sol_path: Path, attr: str = None):
    spec = importlib.util.spec_from_file_location("agent_surface", str(sol_path))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(sol_path.parent))
    spec.loader.exec_module(mod)  # type: ignore
    return mod if attr is None else getattr(mod, attr)


# --------------------------------------------------------------------------- #
# Metrics (in the UNKNOWN band only) — LOWER is better
# --------------------------------------------------------------------------- #
def _grad_mag(a):
    ax = a[:, 1:] - a[:, :-1]
    ay = a[1:, :] - a[:-1, :]
    gx = F.pad(ax, (0, 1, 0, 0))
    gy = F.pad(ay, (0, 0, 0, 1))
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def eval_metrics(pred_alpha, gt_alpha, unknown):
    u = unknown
    if u.sum() < 5:
        return dict(sad=0.0, mse=0.0, grad=0.0, unk_frac=float(u.float().mean()))
    p = pred_alpha.clamp(0, 1)
    diff = (p - gt_alpha).abs()
    sad = float(diff[u].sum().item()) / 1000.0
    mse = float(((p - gt_alpha) ** 2)[u].mean().item()) * 1e3
    gp = _grad_mag(p); gg = _grad_mag(gt_alpha)
    grad = float((gp - gg).abs()[u].sum().item()) / 1000.0
    return dict(sad=sad, mse=mse, grad=grad, unk_frac=float(u.float().mean()))


# --------------------------------------------------------------------------- #
# Fixed encoder (small U-Net encoder). SHARED by ConfigMattingNet surfaces.
# --------------------------------------------------------------------------- #
def _cbr(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                         nn.ReLU(True), nn.Conv2d(cout, cout, 3, padding=1),
                         nn.BatchNorm2d(cout), nn.ReLU(True))


class Encoder(nn.Module):
    """U-Net encoder: 3 downsampling stages. Returns skip features [e0,e1,e2,e3]."""

    def __init__(self, cin, norm_fn=None):
        super().__init__()
        self.e0 = _cbr_norm(cin, 32, norm_fn)     # H
        self.e1 = _cbr_norm(32, 64, norm_fn)      # H/2
        self.e2 = _cbr_norm(64, 96, norm_fn)      # H/4
        self.e3 = _cbr_norm(96, 128, norm_fn)     # H/8 (bottleneck)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e0 = self.e0(x)
        e1 = self.e1(self.pool(e0))
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        return [e0, e1, e2, e3]

    @property
    def channels(self):
        return [32, 64, 96, 128]


def _cbr_norm(cin, cout, norm_fn=None):
    """conv-norm-relu-conv-norm-relu with a pluggable norm (default BatchNorm)."""
    def _n(ch):
        if norm_fn is None:
            return nn.BatchNorm2d(ch)
        try:
            m = norm_fn(ch)
            return m if isinstance(m, nn.Module) else nn.BatchNorm2d(ch)
        except Exception:  # noqa: BLE001
            return nn.BatchNorm2d(ch)
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), _n(cout),
                         nn.ReLU(True), nn.Conv2d(cout, cout, 3, padding=1),
                         _n(cout), nn.ReLU(True))


# --------------------------------------------------------------------------- #
# ConfigMattingNet: fixed encoder + strong U-Net decoder, with pluggable hooks.
#   fuse       : callable(dec_up, skip) -> fused tensor (skip surface)
#   attention  : nn.Module at the bottleneck (same shape) (attention surface)
#   dilation   : nn.Module at the bottleneck (same shape) (dilation surface)
#   norm       : callable(ch) -> nn.Module used inside every conv block (norm surface)
#   upsample   : callable(cin) -> nn.Module upsampling operator (upsampling surface)
#   propagate  : callable(alpha, image, trimap) -> refined alpha (propagation surface)
#   head       : callable(cin) -> nn.Module output head (fgpred surface)
# When a hook is absent the component is the fixed strong default.
# --------------------------------------------------------------------------- #
class _Identity(nn.Module):
    def forward(self, x):
        return x


class _DefaultUpsampler(nn.Module):
    """Default decoder upsampler: bilinear to a reference size."""
    def forward(self, x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear",
                             align_corners=False)


class _DefaultHead(nn.Module):
    """Default output head: 1x1 conv -> 1-channel alpha logit."""
    def __init__(self, cin):
        super().__init__()
        self.conv = nn.Conv2d(cin, 1, 1)

    def forward(self, x):
        return self.conv(x)   # returns (B,1,H,W) logits


class ConfigMattingNet(nn.Module):
    def __init__(self, cin, hooks=None):
        super().__init__()
        hooks = hooks or {}
        norm_fn = hooks.get("norm")
        self.enc = Encoder(cin, norm_fn=norm_fn)
        c0, c1, c2, c3 = self.enc.channels

        self.up3 = _cbr_norm(c3 + c2, c2, norm_fn)
        self.up2 = _cbr_norm(c2 + c1, c1, norm_fn)
        self.up1 = _cbr_norm(c1 + c0, c0, norm_fn)

        # ---- pluggable components (fixed strong defaults) ----
        att = hooks.get("attention")
        self.attention = att if isinstance(att, nn.Module) else _Identity()
        dil = hooks.get("dilation")
        self.dilation = dil if isinstance(dil, nn.Module) else _Identity()

        self._fuse = hooks.get("fuse")            # callable(dec_up, skip)->tensor

        up_fn = hooks.get("upsample")
        self.upsampler = _resolve_upsampler(up_fn)   # nn.ModuleDict-ish
        self._propagate = hooks.get("propagate")  # callable(alpha,image,trimap)->alpha

        head_fn = hooks.get("head")
        self.head, self._head_multi = _resolve_head(head_fn, c0)

    def _do_fuse(self, dec_up, skip):
        if self._fuse is None:
            return torch.cat([dec_up, skip], 1)
        try:
            out = self._fuse(dec_up, skip)
            assert torch.is_tensor(out)
            return out
        except Exception:  # noqa: BLE001
            return torch.cat([dec_up, skip], 1)

    def _up(self, x, ref):
        return self.upsampler(x, ref)

    def forward(self, x, image=None, trimap=None):
        feats = self.enc(x)
        e0, e1, e2, e3 = feats
        m = self.attention(e3)
        m = self.dilation(m)
        d = self.up3(self._do_fuse(self._up(m, e2), e2))
        d = self.up2(self._do_fuse(self._up(d, e1), e1))
        d = self.up1(self._do_fuse(self._up(d, e0), e0))
        out = self.head(d)                      # (B,>=1,H,W) logits
        # stash raw head output so the harness can add an aux FG loss (fgpred surface)
        self.last_head = out
        alpha = torch.sigmoid(out[:, 0])        # (B,H,W)
        if self._propagate is not None and image is not None:
            try:
                a2 = self._propagate(alpha, image, trimap)
                assert torch.is_tensor(a2) and a2.shape == alpha.shape
                alpha = a2.clamp(0, 1)
            except Exception:  # noqa: BLE001
                pass
        return alpha


# --------------------------------------------------------------------------- #
# Default (strong) network used as the `arch`-surface fallback and coarse stage.
# --------------------------------------------------------------------------- #
class UNetMatting(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.enc = Encoder(cin)
        c0, c1, c2, c3 = self.enc.channels
        self.up3 = _cbr(c3 + c2, c2)
        self.up2 = _cbr(c2 + c1, c1)
        self.up1 = _cbr(c1 + c0, c0)
        self.out = nn.Conv2d(c0, 1, 1)

    def _up(self, x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x, image=None, trimap=None):
        e0, e1, e2, e3 = self.enc(x)
        d = self.up3(torch.cat([self._up(e3, e2), e2], 1))
        d = self.up2(torch.cat([self._up(d, e1), e1], 1))
        d = self.up1(torch.cat([self._up(d, e0), e0], 1))
        return torch.sigmoid(self.out(d)).squeeze(1)


def default_build_net(cin):
    return UNetMatting(cin)


# --------------------------------------------------------------------------- #
# Default decoder (for the decoder surface fallback) — full U-Net skip decoder.
# --------------------------------------------------------------------------- #
class UNetDecoder(nn.Module):
    def __init__(self, ch):
        super().__init__()
        c0, c1, c2, c3 = ch
        self.up3 = _cbr(c3 + c2, c2)
        self.up2 = _cbr(c2 + c1, c1)
        self.up1 = _cbr(c1 + c0, c0)
        self.out = nn.Conv2d(c0, 1, 1)

    def _up(self, x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, feats):
        e0, e1, e2, e3 = feats
        d = self.up3(torch.cat([self._up(e3, e2), e2], 1))
        d = self.up2(torch.cat([self._up(d, e1), e1], 1))
        d = self.up1(torch.cat([self._up(d, e0), e0], 1))
        return torch.sigmoid(self.out(d)).squeeze(1)


class DecoderMattingNet(nn.Module):
    """Fixed encoder + the agent's decoder head (for the decoder surface)."""
    def __init__(self, cin, decoder):
        super().__init__()
        self.enc = Encoder(cin)
        self.dec = decoder

    def forward(self, x, image=None, trimap=None):
        return self.dec(self.enc(x))


# --------------------------------------------------------------------------- #
# Component hook resolvers
# --------------------------------------------------------------------------- #
class _UpsamplerWrapper(nn.Module):
    def __init__(self, fns):
        super().__init__()
        # fns: nn.ModuleList of per-stage upsamplers OR a single module
        self.default = _DefaultUpsampler()
        self.mod = fns

    def forward(self, x, ref):
        if self.mod is None:
            return self.default(x, ref)
        try:
            out = self.mod(x)
            if out.shape[-2:] != ref.shape[-2:]:
                out = F.interpolate(out, size=ref.shape[-2:], mode="bilinear",
                                    align_corners=False)
            return out
        except Exception:  # noqa: BLE001
            return self.default(x, ref)


def _resolve_upsampler(up_fn):
    if up_fn is None:
        return _DefaultUpsampler()
    try:
        # up_fn(cin) -> nn.Module; but channel count varies per stage, so we build a
        # channel-agnostic wrapper: probe with a dummy conv is impossible, so we
        # require the module to preserve channels. If it can't, fall back.
        probe = up_fn(96)
        assert isinstance(probe, nn.Module)
        out = probe(torch.zeros(1, 96, 8, 8))
        assert torch.is_tensor(out)
        return _StageUpsampler(up_fn)
    except Exception:  # noqa: BLE001
        return _DefaultUpsampler()


class _StageUpsampler(nn.Module):
    """Builds a per-channel upsampler lazily so each decoder stage gets its own."""
    def __init__(self, up_fn):
        super().__init__()
        self.up_fn = up_fn
        self.cache = nn.ModuleDict()
        self.default = _DefaultUpsampler()

    def forward(self, x, ref):
        key = str(x.shape[1])
        if key not in self.cache:
            try:
                m = self.up_fn(x.shape[1]).to(x.device)
                assert isinstance(m, nn.Module)
                self.cache[key] = m
            except Exception:  # noqa: BLE001
                return self.default(x, ref)
        try:
            out = self.cache[key](x)
            if out.shape[-2:] != ref.shape[-2:]:
                out = F.interpolate(out, size=ref.shape[-2:], mode="bilinear",
                                    align_corners=False)
            return out
        except Exception:  # noqa: BLE001
            return self.default(x, ref)


def _resolve_head(head_fn, cin):
    if head_fn is None:
        return _DefaultHead(cin), False
    try:
        m = head_fn(cin)
        assert isinstance(m, nn.Module)
        out = m(torch.zeros(1, cin, 8, 8))
        assert torch.is_tensor(out) and out.shape[1] >= 1
        return m, (out.shape[1] >= 4)
    except Exception:  # noqa: BLE001
        return _DefaultHead(cin), False


# --------------------------------------------------------------------------- #
# Default matting loss (FIXED for non-loss surfaces)
# --------------------------------------------------------------------------- #
def default_matting_loss(pred, gt, image, fg, bg, trimap, unknown):
    """alpha-L1 + composition-L1 in the unknown band (Deep Image Matting)."""
    u = unknown.float()
    eps = 1e-6
    denom = u.sum(dim=(-2, -1)).clamp(min=1.0)
    al = ((pred - gt).abs() * u).sum(dim=(-2, -1)) / denom
    comp = pred.unsqueeze(1) * fg + (1 - pred.unsqueeze(1)) * bg
    cl = ((comp - image).abs().mean(1) * u).sum(dim=(-2, -1)) / denom
    return (al + 0.5 * cl).mean() + eps * 0.0


def default_trimap_encode(trimap):
    """FIXED default for the non-trimap surfaces: the raw trimap as 1 channel."""
    return trimap.unsqueeze(1)


# --------------------------------------------------------------------------- #
# Refine wrapper: fixed coarse ConfigMattingNet + agent refine() second stage.
# --------------------------------------------------------------------------- #
class _RefineWrapper(nn.Module):
    def __init__(self, coarse, refine_fn):
        super().__init__()
        self.coarse = coarse
        self._refine = refine_fn

    def forward(self, x, image=None, trimap=None):
        a = self.coarse(x)
        if self._refine is None:
            return a
        try:
            out = self._refine(a, x, trimap if trimap is not None else x[:, 3])
            assert torch.is_tensor(out) and out.shape == a.shape
            return out.clamp(0, 1)
        except Exception:  # noqa: BLE001
            return a


# --------------------------------------------------------------------------- #
# Build the network for a surface
# --------------------------------------------------------------------------- #
def _component_hooks(surface, mod, cin):
    """Load an agent COMPONENT hook and package it for ConfigMattingNet."""
    hooks = {}
    if surface == "skip":
        try:
            fn = mod.fuse
            dec = torch.zeros(1, 8, 4, 4); skip = torch.zeros(1, 4, 4, 4)
            out = fn(dec, skip)
            assert torch.is_tensor(out) and out.shape[0] == 1 and out.shape[-2:] == (4, 4)
            hooks["fuse"] = fn
            print("SKIP_APPLIED custom skip fusion", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"SKIP_FALLBACK reason={e!r}", flush=True)
    elif surface == "attention":
        try:
            fn = mod.build_attention
            m = fn(128)
            assert isinstance(m, nn.Module)
            probe = m(torch.zeros(1, 128, 16, 16))
            assert probe.shape == (1, 128, 16, 16)
            hooks["attention"] = fn(128)
            print("ATTENTION_APPLIED custom bottleneck", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ATTENTION_FALLBACK reason={e!r}", flush=True)
    elif surface == "dilation":
        try:
            fn = mod.build_dilation
            m = fn(128)
            assert isinstance(m, nn.Module)
            probe = m(torch.zeros(1, 128, 16, 16))
            assert probe.shape == (1, 128, 16, 16)
            hooks["dilation"] = fn(128)
            print("DILATION_APPLIED custom dilation block", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"DILATION_FALLBACK reason={e!r}", flush=True)
    elif surface == "norm":
        try:
            fn = mod.make_norm
            m = fn(32)
            assert isinstance(m, nn.Module)
            _ = m(torch.zeros(2, 32, 8, 8))
            hooks["norm"] = fn
            print("NORM_APPLIED custom norm layer", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"NORM_FALLBACK reason={e!r}", flush=True)
    elif surface == "upsampling":
        try:
            fn = mod.build_upsampler
            m = fn(64)
            assert isinstance(m, nn.Module)
            out = m(torch.zeros(1, 64, 8, 8))
            assert torch.is_tensor(out) and out.shape[1] == 64
            hooks["upsample"] = fn
            print("UPSAMPLING_APPLIED custom upsampler", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"UPSAMPLING_FALLBACK reason={e!r}", flush=True)
    elif surface == "propagation":
        try:
            fn = mod.propagate
            a = torch.rand(1, 32, 32); im = torch.rand(1, 3, 32, 32); tr = torch.rand(1, 32, 32)
            out = fn(a, im, tr)
            assert torch.is_tensor(out) and out.shape == a.shape
            hooks["propagate"] = fn
            print("PROPAGATION_APPLIED custom propagation", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"PROPAGATION_FALLBACK reason={e!r}", flush=True)
    elif surface == "fgpred":
        try:
            fn = mod.build_head
            m = fn(32)
            assert isinstance(m, nn.Module)
            out = m(torch.zeros(1, 32, 8, 8))
            assert torch.is_tensor(out) and out.shape[1] >= 1
            hooks["head"] = fn
            print("FGPRED_APPLIED custom head", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"FGPRED_FALLBACK reason={e!r}", flush=True)
    return hooks


def _stack(samples, idxs, device):
    b = {}
    for k in ("image", "trimap", "alpha", "fg", "bg", "unknown"):
        b[k] = torch.stack([samples[i][k] for i in idxs]).to(device)
    return b


def train_and_eval(surface, mod, train, val, device, iters, seed, batch=8):
    set_all_seeds(seed)

    # ---- trimap encoder: agent's for trimap surface, default (raw) otherwise ----
    trimap_encode = default_trimap_encode
    if surface == "trimap":
        try:
            fn = mod.encode_trimap
            assert callable(fn)
            dummy = torch.full((1, 8, 8), 0.5)
            enc = fn(dummy)
            assert torch.is_tensor(enc) and enc.dim() == 4 and enc.shape[0] == 1
            trimap_encode = fn
            print(f"TRIMAP_APPLIED custom encode -> {enc.shape[1]} channel(s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"TRIMAP_FALLBACK reason={e!r}", flush=True)
            trimap_encode = default_trimap_encode

    probe = trimap_encode(train[0]["trimap"].unsqueeze(0))
    k = int(probe.shape[1])
    cin = 3 + k
    print(f"INPUT_CHANNELS rgb=3 + trimap_enc={k} -> cin={cin}", flush=True)

    # ---- loss: agent's for loss surface, default otherwise ----
    loss_fn = default_matting_loss
    if surface == "loss":
        try:
            loss_fn = mod.get_matting_loss()
            assert callable(loss_fn)
            print("LOSS_APPLIED custom loss", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"LOSS_FALLBACK reason={e!r}", flush=True)
            loss_fn = default_matting_loss

    # ---- build the network per surface ----
    aux_fg = False
    if surface == "arch":
        build_net = default_build_net
        try:
            build_net = mod.build_net
            probe_net = build_net(cin)
            assert isinstance(probe_net, nn.Module)
            print("ARCH_APPLIED custom net", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ARCH_FALLBACK reason={e!r}", flush=True)
            build_net = default_build_net
        try:
            net = build_net(cin)
            assert isinstance(net, nn.Module)
        except Exception as e:  # noqa: BLE001
            print(f"ARCH_FALLBACK(build) reason={e!r}", flush=True)
            net = default_build_net(cin)
    elif surface == "decoder":
        enc_ch = Encoder(cin).channels
        try:
            decoder = mod.build_decoder(enc_ch)
            assert isinstance(decoder, nn.Module)
            print("DECODER_APPLIED custom decoder", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"DECODER_FALLBACK reason={e!r}", flush=True)
            decoder = UNetDecoder(enc_ch)
        net = DecoderMattingNet(cin, decoder)
    elif surface == "refine":
        refine_fn = None
        try:
            refine_fn = mod.refine
            ca = torch.rand(1, 8, 8); xin = torch.rand(1, cin, 8, 8); tr = torch.rand(1, 8, 8)
            out = refine_fn(ca, xin, tr)
            assert torch.is_tensor(out) and out.shape == ca.shape
            print("REFINE_APPLIED custom refine stage", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"REFINE_FALLBACK reason={e!r}", flush=True)
            refine_fn = None
        net = _RefineWrapper(ConfigMattingNet(cin), refine_fn)
    elif surface in COMPONENT_SURFACES:
        hooks = _component_hooks(surface, mod, cin)
        net = ConfigMattingNet(cin, hooks=hooks)
        aux_fg = (surface == "fgpred" and net._head_multi)
    else:  # loss, trimap -> fixed ConfigMattingNet
        net = ConfigMattingNet(cin)

    net = net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def _forward(batch_dict):
        enc = trimap_encode(batch_dict["trimap"]).to(device)
        if enc.shape[1] != k:
            enc = enc[:, :k] if enc.shape[1] > k else F.pad(enc, (0, 0, 0, 0, 0, k - enc.shape[1]))
        x = torch.cat([batch_dict["image"], enc], 1)
        pred = net(x, image=batch_dict["image"], trimap=batch_dict["trimap"])
        if pred.dim() == 3 and pred.shape[-2:] != batch_dict["alpha"].shape[-2:]:
            pred = F.interpolate(pred[:, None], size=batch_dict["alpha"].shape[-2:],
                                 mode="bilinear", align_corners=False)[:, 0]
        return pred

    n = len(train)
    order = list(range(n))
    for it in range(iters):
        if it % (n // batch + 1) == 0:
            random.shuffle(order)
        idxs = [order[(it * batch + j) % n] for j in range(batch)]
        b = _stack(train, idxs, device)
        pred = _forward(b)
        try:
            loss = loss_fn(pred, b["alpha"], b["image"], b["fg"], b["bg"],
                           b["trimap"], b["unknown"])
        except Exception as e:  # noqa: BLE001
            if it == 0:
                print(f"LOSS_CALL_FALLBACK reason={e!r}", flush=True)
            loss = default_matting_loss(pred, b["alpha"], b["image"], b["fg"],
                                        b["bg"], b["trimap"], b["unknown"])
        # aux FG-prediction supervision (fgpred surface, joint head): if the head
        # emits >=4 channels, supervise channels 1:4 as a foreground-colour prediction
        # in the unknown band. This auxiliary task regularises the shared decoder
        # features (Context-Aware Matting, Hou 2019) and improves the matte.
        if aux_fg:
            try:
                head_out = net.last_head            # (B,>=4,H,W)
                if head_out.shape[1] >= 4:
                    fg_pred = torch.sigmoid(head_out[:, 1:4])
                    u = b["unknown"].float().unsqueeze(1)
                    denom = u.sum(dim=(-2, -1, -3)).clamp(min=1.0)
                    fg_l = ((fg_pred - b["fg"]).abs() * u).sum(dim=(-2, -1, -3)) / denom
                    loss = loss + 0.25 * fg_l.mean()
            except Exception:  # noqa: BLE001
                pass
        if not torch.isfinite(loss):
            loss = default_matting_loss(pred, b["alpha"], b["image"], b["fg"],
                                        b["bg"], b["trimap"], b["unknown"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if it % max(1, iters // 5) == 0 or it == iters - 1:
            print(f"train it={it} loss={float(loss):.4f}", flush=True)

    net.eval()
    accs = []
    with torch.no_grad():
        for s in val:
            b = _stack([s], [0], device)
            pred = _forward(b)[0]
            accs.append(eval_metrics(pred, b["alpha"][0], b["unknown"][0]))
    return _avg(accs)


def _avg(accs):
    if not accs:
        return dict(sad=999.0, mse=999.0, grad=999.0, unk_frac=0.0)
    keys = accs[0].keys()
    return {k: float(np.mean([a[k] for a in accs])) for k in keys}


def degenerate_sads(val):
    const_half, mean_alpha = [], []
    for s in val:
        u = s["unknown"]; g = s["alpha"]
        gu = g[u]
        if gu.numel() == 0:
            continue
        const_half.append(float((0.5 - gu).abs().sum().item()) / 1000.0)
        mean_alpha.append(float((gu.mean() - gu).abs().sum().item()) / 1000.0)
    return float(np.mean(const_half)), float(np.mean(mean_alpha))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--surface", required=True, choices=sorted(ALL_SURFACES))
    ap.add_argument("--trimap-width", default="medium",
                    choices=sorted(TRIMAP_WIDTHS), help="scored val trimap-band width")
    ap.add_argument("--label", default="run")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE {device} torch {torch.__version__}", flush=True)

    # training always uses the FIXED medium band; only the SCORED val band varies.
    train = load_split(args.data_root, "train", TRAIN_TRIMAP_WIDTH)
    val = load_split(args.data_root, "val", TRIMAP_WIDTHS[args.trimap_width])
    ch, ma = degenerate_sads(val)
    uf = float(np.mean([float(s["unknown"].float().mean()) for s in val]))
    print(f"DATA train={len(train)} val={len(val)} trimap_width={args.trimap_width} "
          f"unk_frac={uf:.3f} CONST_HALF_SAD={ch:.3f} MEAN_ALPHA_SAD={ma:.3f}", flush=True)

    mod = load_surface(Path(args.solution))
    m = train_and_eval(args.surface, mod, train, val, device, args.iters, args.seed)

    print(f"MATTING_METRICS surface={args.surface} setting={args.label} "
          f"sad={m['sad']:.4f} mse={m['mse']:.4f} grad={m['grad']:.4f} "
          f"unk_frac={m['unk_frac']:.4f}", flush=True)


if __name__ == "__main__":
    main()
