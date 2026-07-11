"""Reusable flow-block constructors for the flow-* tasks.

Thin wrappers over `normflows` primitives used by the frozen task builders.
Flow constructors return individual `nf.flows.Flow` modules; the harness pairs
their ordered list with a base distribution and creates `nf.NormalizingFlow`.
"""
from __future__ import annotations

import normflows as nf


def diag_gaussian(dim: int, trainable: bool = False):
    """Standard diagonal-Gaussian base distribution (fixed, non-trainable)."""
    return nf.distributions.DiagGaussian(dim, trainable=trainable)


def gmm_base(dim: int, n_modes: int = 8, trainable: bool = True):
    """Learnable Gaussian-mixture base distribution.

    The component count and trainability are explicit constructor arguments.
    """
    return nf.distributions.GaussianMixture(n_modes, dim, trainable=trainable)


def affine_coupling_layer(dim: int, hidden: int = 64, n_hidden: int = 2):
    """One affine (RealNVP-style) coupling block.

    A small MLP maps one half of the dims to the scale+shift of the other half.
    `channel` split_mode partitions the input dims; alternate with a permutation.
    """
    in_half = dim // 2
    out_half = dim - in_half
    layers = [in_half] + [hidden] * n_hidden + [out_half * 2]
    param = nf.nets.MLP(layers, init_zeros=True)
    return nf.flows.AffineCouplingBlock(param, split_mode="channel", scale=True)


def spline_coupling_layer(dim: int, hidden: int = 64, num_blocks: int = 1,
                          num_bins: int = 8, tail_bound: float = 4.0,
                          reverse_mask: bool = False):
    """One rational-quadratic-spline coupling layer (Neural Spline Flow).

    Far more expressive per layer than affine: it fits sharp / multimodal
    densities (checkerboard, moons) with much lower NLL.
    """
    return nf.flows.CoupledRationalQuadraticSpline(
        dim, num_blocks, hidden, num_bins=num_bins, tail_bound=tail_bound,
        reverse_mask=reverse_mask,
    )


def maf_layer(dim: int, hidden: int = 64, num_blocks: int = 2):
    """One masked affine autoregressive (MAF) layer.

    Autoregressive conditioner: each dim's transform depends on all previous
    dims. More expressive than a single affine coupling, at higher per-step cost.
    """
    return nf.flows.MaskedAffineAutoregressive(
        dim, hidden, num_blocks=num_blocks, use_residual_blocks=True,
    )


def swap_permute(dim: int):
    """Deterministic swap permutation (RealNVP alternation)."""
    return nf.flows.Permute(dim, mode="swap")


def lu_permute(dim: int):
    """Learnable LU-linear invertible permutation (Glow-style mixing)."""
    return nf.flows.LULinearPermute(dim)


def masked_affine_layer(dim: int, mask, hidden: int = 64, n_hidden: int = 2):
    """RealNVP affine layer driven by an EXPLICIT binary mask (masking pattern).

    `mask` is a length-`dim` 0/1 tensor/list marking which dims are passed
    through unchanged (1) vs transformed conditioned on the passed-through dims
    (0).
    """
    import torch
    b = torch.as_tensor(mask, dtype=torch.float32)
    layers = [dim] + [hidden] * n_hidden + [dim]
    s = nf.nets.MLP(layers, init_zeros=True)
    t = nf.nets.MLP(layers, init_zeros=True)
    return nf.flows.MaskedAffineFlow(b, t, s)
