"""FIXED small analysis/synthesis transforms + entropy-model model shells for the
compress-* tasks. Kept intentionally small (few-hundred-step trainable in ~1-2 min).

These wrap CompressAI's own primitives (GDN, conv/deconv, EntropyBottleneck,
GaussianConditional, ResidualBlock*, AttentionBlock, MaskedConv2d, ...) so the
tasks exercise the real learned-compression stack.

Two API layers:
  - g_a/g_s/h_a/h_s_scale/h_s_meanscale(N, M): the ORIGINAL fixed 4-layer,
    stride-2 transforms used by the 3 pre-existing tasks (compress-entropy-model,
    compress-quantization-surrogate, compress-rd-target). Byte-identical to
    before -- unchanged.
  - build_transform(...) / build_hyper_*(...): a CONFIG-DRIVEN builder used by the
    new expansion tasks (activation, width, latent-channels, hyperprior depth,
    context model, upsampling, normalization, residual-block count). Each knob
    defaults to the value that reproduces the ORIGINAL transforms above, so a
    surface that leaves every other knob at its default recovers the original
    small codec exactly.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import (
    GDN,
    AttentionBlock,
    MaskedConv2d,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv1x1,
    conv3x3,
    subpel_conv3x3,
)
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv


# --------------------------------------------------------------------------- #
# ORIGINAL fixed transforms (unchanged; used by the 3 pre-existing tasks)      #
# --------------------------------------------------------------------------- #

def g_a(N, M):
    """Analysis transform g_a: 3 -> M, 4x downsample (bmshj2018 style)."""
    return nn.Sequential(
        conv(3, N), GDN(N),
        conv(N, N), GDN(N),
        conv(N, N), GDN(N),
        conv(N, M),
    )


def g_s(N, M):
    """Synthesis transform g_s: M -> 3."""
    return nn.Sequential(
        deconv(M, N), GDN(N, inverse=True),
        deconv(N, N), GDN(N, inverse=True),
        deconv(N, N), GDN(N, inverse=True),
        deconv(N, 3),
    )


def h_a(N, M):
    return nn.Sequential(
        conv(M, N, stride=1, kernel_size=3), nn.ReLU(inplace=True),
        conv(N, N), nn.ReLU(inplace=True),
        conv(N, N),
    )


def h_s_scale(N, M):
    """Hyper-synthesis producing scales only (mean-free hyperprior)."""
    return nn.Sequential(
        deconv(N, N), nn.ReLU(inplace=True),
        deconv(N, N), nn.ReLU(inplace=True),
        conv(N, M, stride=1, kernel_size=3), nn.ReLU(inplace=True),
    )


def h_s_meanscale(N, M):
    """Hyper-synthesis producing both means and scales (mean-scale hyperprior)."""
    return nn.Sequential(
        deconv(N, M), nn.LeakyReLU(inplace=True),
        deconv(M, M * 3 // 2), nn.LeakyReLU(inplace=True),
        conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
    )


# --------------------------------------------------------------------------- #
# CONFIG-DRIVEN transform builder (new expansion surfaces)                    #
# --------------------------------------------------------------------------- #

_ACTS = {
    "gdn": None,       # handled specially (needs in/inverse channel count)
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(inplace=True),
    "identity": nn.Identity,
}


def _act_layer(kind: str, ch: int, inverse: bool = False):
    """Build one activation layer for the given kind on `ch` channels."""
    if kind == "gdn":
        return GDN(ch, inverse=inverse)
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    if kind == "identity":
        return nn.Identity()
    return GDN(ch, inverse=inverse)


def _norm_layer(kind: str, ch: int):
    """Optional extra normalization layer inserted after each conv (before act)."""
    if kind == "batchnorm":
        return nn.BatchNorm2d(ch)
    if kind == "none":
        return nn.Identity()
    return nn.Identity()


class _NearestResizeConv(nn.Module):
    """2x nearest-neighbour upsample + 3x3 conv (the classic "resize-conv"
    alternative to a learned transposed conv; avoids checkerboard artifacts but
    has no learned upsampling kernel)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = conv3x3(in_ch, out_ch)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class _NormActConv(nn.Module):
    """One down/up-sampling stage: conv/deconv or residual block, optional extra
    norm, then the configured activation. Used by build_transform below."""

    def __init__(self, in_ch, out_ch, up: bool, act: str, norm: str,
                 use_residual: bool, inverse_act: bool, upsample_mode: str = "deconv"):
        super().__init__()
        if use_residual:
            self.body = (
                ResidualBlockUpsample(in_ch, out_ch, upsample=2)
                if up else ResidualBlockWithStride(in_ch, out_ch, stride=2)
            )
            # ResidualBlock* variants already bake in a GDN internally; only
            # add the configured norm/act as an extra head so "act" still
            # controls the *visible* nonlinearity for non-gdn choices.
            self.norm = _norm_layer(norm, out_ch)
            self.act = (nn.Identity() if act == "gdn"
                        else _act_layer(act, out_ch, inverse=inverse_act))
        elif up and upsample_mode == "subpel":
            self.body = subpel_conv3x3(in_ch, out_ch, 2)
            self.norm = _norm_layer(norm, out_ch)
            self.act = _act_layer(act, out_ch, inverse=inverse_act)
        elif up and upsample_mode == "nearest":
            self.body = _NearestResizeConv(in_ch, out_ch)
            self.norm = _norm_layer(norm, out_ch)
            self.act = _act_layer(act, out_ch, inverse=inverse_act)
        else:
            conv_fn = deconv if up else conv
            self.body = conv_fn(in_ch, out_ch)
            self.norm = _norm_layer(norm, out_ch)
            self.act = _act_layer(act, out_ch, inverse=inverse_act)

    def forward(self, x):
        x = self.body(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def build_transform(
    in_ch: int,
    out_ch: int,
    N: int,
    M: int,
    up: bool,
    depth: int = 4,
    activation: str = "gdn",
    norm: str = "none",
    residual: bool = False,
    attention: bool = False,
    upsample_mode: str = "deconv",
) -> nn.Sequential:
    """Build a g_a-style (up=False) or g_s-style (up=True) transform.

    depth=4, activation="gdn", norm="none", residual=False, attention=False,
    upsample_mode="deconv" reproduces the ORIGINAL g_a/g_s exactly (4 stride-2
    stages, GDN between, learned transposed-conv upsampling in g_s).
    Knobs:
      depth       number of stride-2 stages (>=2). Channel width is N for all
                  interior stages, M at the latent end (matches the original).
      activation  "gdn" | "relu" | "leaky_relu" | "identity" (identity = linear,
                  a degenerate transform with no nonlinearity -- should lose).
      norm        "none" | "batchnorm" extra normalization after each conv.
      residual    if True, use CompressAI's ResidualBlockWithStride /
                  ResidualBlockUpsample instead of a plain conv+act.
      attention   if True, append one CompressAI AttentionBlock (Cheng2020-style
                  non-local self-attention) at the latent end.
      upsample_mode (synthesis transform only) "deconv" (learned transposed conv,
                  original) | "subpel" (sub-pixel / PixelShuffle conv, ESPCN-style)
                  | "nearest" (fixed nearest-neighbour resize + conv).
    """
    depth = max(2, int(depth))
    layers = []
    if not up:
        chans = [in_ch] + [N] * (depth - 1) + [out_ch]
        for i in range(depth):
            is_last = i == depth - 1
            layers.append(_NormActConv(
                chans[i], chans[i + 1], up=False,
                act="identity" if is_last else activation,
                norm="none" if is_last else norm,
                use_residual=(residual and not is_last),
                inverse_act=False,
            ))
        if attention:
            layers.append(AttentionBlock(out_ch))
    else:
        chans = [in_ch] + [N] * (depth - 1) + [out_ch]
        if attention:
            layers.append(AttentionBlock(in_ch))
        for i in range(depth):
            is_last = i == depth - 1
            layers.append(_NormActConv(
                chans[i], chans[i + 1], up=True,
                act="identity" if is_last else activation,
                norm="none" if is_last else norm,
                use_residual=(residual and not is_last),
                inverse_act=True,
                upsample_mode=upsample_mode,
            ))
    return nn.Sequential(*layers)


def build_hyper_a(N: int, M: int, depth: int = 2, in_ch: int | None = None) -> nn.Sequential:
    """Hyper-analysis h_a: M (or |y|) -> N, ONE stride-1 channel-change conv
    followed by `depth` stride-2 downsampling convs. depth=2 reproduces the
    original h_a exactly (M->N stride1, then 2x stride-2: total 4x downsample)."""
    depth = max(1, int(depth))
    in_ch = M if in_ch is None else in_ch
    layers = [conv(in_ch, N, stride=1, kernel_size=3)]
    for i in range(depth):
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv(N, N, stride=2))
    return nn.Sequential(*layers)


def build_hyper_s_scale(N: int, M: int, depth: int = 2) -> nn.Sequential:
    """Hyper-synthesis (scale-only): `depth` stride-2 upsampling deconvs then ONE
    stride-1 channel-change conv to M. depth=2 reproduces the original
    h_s_scale exactly (2x stride-2 upsample, then N->M stride1)."""
    depth = max(1, int(depth))
    layers = []
    for _ in range(depth):
        layers.append(deconv(N, N))
        layers.append(nn.ReLU(inplace=True))
    layers.append(conv(N, M, stride=1, kernel_size=3))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SmallContextModel(nn.Module):
    """Small autoregressive context model over the latent y, wrapping
    CompressAI's real MaskedConv2d, mirroring (a scaled-down) mbt2018 joint
    hyperprior+context design. Produces gaussian (scale, mean) params by fusing
    the hyper-network params with a masked-conv context prediction."""

    def __init__(self, N: int, M: int):
        super().__init__()
        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1, mask_type="A"
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 4, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 2, 1),
        )

    def forward(self, y_hat_for_ctx, hyper_params):
        ctx = self.context_prediction(y_hat_for_ctx)
        gaussian_params = self.entropy_parameters(
            torch.cat((hyper_params, ctx), dim=1)
        )
        scales, means = gaussian_params.chunk(2, 1)
        return scales, means
