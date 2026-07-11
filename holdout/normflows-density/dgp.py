"""Held-out data-generating-process module for normflows-density (NOT agent-visible).

This module lives OUTSIDE every path that is bind-mounted into the agent's
container (the vendored package dir, the per-task scaffold). It is imported
only by this directory's ``generate_data.py``, a one-off HOST-SIDE script that
pre-generates FIXED train/test 2-D toy-density samples at repo-authoring time
and bakes ONLY the sampled arrays (never this source file) into
``vendor/normflows-density/_flow_data/``.

It holds the one secret that must never reach the agent's editable scope: the
EXACT closed-form sampling procedure -- including every hardcoded numeric
constant (radii, noise scales, grid spacing/cell size, means/covariances,
rotation rates) -- for each of the five 2-D toy targets (moons / checkerboard /
circles / pinwheel / 8gaussians). Knowing these exact numbers would let an
evaluated flow shortcut the benchmark: instead of learning a density from
samples, it could hand-code the target's analytic inverse map (or its exact
log-density) and score near-perfect held-out NLL WITHOUT doing any real
density-estimation learning.

The setting NAME (moons/checkerboard/circles/pinwheel/8gaussians) itself is
NOT secret -- it is legitimate, disclosed benchmark framing (like naming
"CIFAR-10" as the dataset for a vision task): these are standard, publicly
named toy-flow testbeds, and there are infinitely many valid
parameterizations of e.g. "checkerboard" (cell size, noise, extent). Knowing
the family name alone does not hand over the answer. Only the exact numeric
parameters (and the sampling code that encodes them) are held out here.

``_sample_two_moons`` / ``_sample_checkerboard`` / ``_sample_two_circles`` /
``_sample_pinwheel`` / ``_sample_eight_gaussians`` / ``TARGETS`` are
byte-identical to the originals that used to live in
``vendor/normflows-density/common.py`` before the oracle-leakage fix -- so the
sampled train/test arrays, and therefore every honest downstream result, are
reproduced exactly.
"""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Deterministic 2-D toy target densities — the secret DGP.
# ---------------------------------------------------------------------------
# Each sampler draws i.i.d. samples from a fixed 2-D density given a numpy
# Generator. Byte-identical to the pre-fix vendor/normflows-density/common.py.

def _sample_two_moons(n: int, rng: np.random.Generator) -> np.ndarray:
    """Two interleaving half-moons (the classic sklearn make_moons, noise=0.1)."""
    n0 = n // 2
    n1 = n - n0
    t0 = rng.uniform(0.0, math.pi, size=n0)
    x0 = np.stack([np.cos(t0), np.sin(t0)], axis=1)
    t1 = rng.uniform(0.0, math.pi, size=n1)
    x1 = np.stack([1.0 - np.cos(t1), 0.5 - np.sin(t1)], axis=1)
    x = np.concatenate([x0, x1], axis=0)
    x += rng.normal(scale=0.10, size=x.shape)
    # center + scale to roughly unit spread (helps a Gaussian base)
    x = (x - np.array([0.5, 0.25])) / 1.0
    return x.astype(np.float32)


def _sample_checkerboard(n: int, rng: np.random.Generator) -> np.ndarray:
    """Classic sharp checkerboard (the standard flow testbed).

    Mass lives only in alternating unit cells of a 4x4 grid over [-2,2]^2, with
    hard on/off boundaries. This sharp multimodal structure is exactly what an
    affine coupling cannot carve (it can only per-dim scale/shift), while a
    rational-quadratic spline coupling fits it — a large, reliable NLL gap.
    """
    # Fine 6-column checkerboard over [-3,3]^2: 18 filled unit cells with hard
    # on/off boundaries and 3 stacked modes of x2 per column. The many modes of
    # p(x2 | x1) are exactly what an affine coupling (Gaussian -> unimodal per
    # layer) cannot cheaply reproduce, while one rational-quadratic spline
    # coupling can — yielding a large, reliable affine-vs-spline NLL gap.
    n_cols = 6
    x1 = rng.uniform(-3.0, 3.0, size=n)
    col = np.clip(np.floor(x1 + 3.0).astype(int), 0, n_cols - 1)  # 0..5
    # 3 candidate rows per column; filled rows shift by (col % 2) -> checkerboard.
    row = 2 * rng.integers(0, 3, size=n) + (col % 2)             # {0,2,4} or {1,3,5}
    x2 = rng.uniform(0.0, 1.0, size=n) + row - 3.0              # into [-3,3]
    x = np.stack([x1, x2], axis=1)
    return x.astype(np.float32)


def _sample_two_circles(n: int, rng: np.random.Generator) -> np.ndarray:
    """Two concentric noisy circles (sklearn make_circles style)."""
    n0 = n // 2
    n1 = n - n0
    t0 = rng.uniform(0.0, 2 * math.pi, size=n0)
    r0 = 1.0
    x0 = np.stack([r0 * np.cos(t0), r0 * np.sin(t0)], axis=1)
    t1 = rng.uniform(0.0, 2 * math.pi, size=n1)
    r1 = 0.5
    x1 = np.stack([r1 * np.cos(t1), r1 * np.sin(t1)], axis=1)
    x = np.concatenate([x0, x1], axis=0)
    x += rng.normal(scale=0.06, size=x.shape)
    return x.astype(np.float32)


def _sample_pinwheel(n: int, rng: np.random.Generator) -> np.ndarray:
    """5-arm pinwheel: rotated, sheared Gaussian blobs (a standard flow testbed)."""
    n_blades = 5
    rate = 0.25
    per = n // n_blades
    feats = []
    for k in range(n_blades):
        m = per if k < n_blades - 1 else n - per * (n_blades - 1)
        base = rng.normal(size=(m, 2)) * np.array([0.30, 0.05]) + np.array([1.0, 0.0])
        ang = (2 * math.pi / n_blades) * k + rate * np.exp(base[:, 0])
        c, s = np.cos(ang), np.sin(ang)
        rot = np.stack([base[:, 0] * c - base[:, 1] * s,
                        base[:, 0] * s + base[:, 1] * c], axis=1)
        feats.append(rot)
    x = np.concatenate(feats, axis=0)
    return (x * 0.7).astype(np.float32)


def _sample_eight_gaussians(n: int, rng: np.random.Generator) -> np.ndarray:
    """Classic ring of 8 equally-weighted Gaussians (standard flow testbed).

    8 modes on a radius-1.5 circle, each an isotropic Gaussian (std 0.3). This
    well-separated multimodal target is exactly what a single affine coupling
    (Gaussian -> unimodal per layer) fits poorly, while a spline coupling or a
    multimodal base carves the 8 modes at much lower NLL. The moderate radius /
    std keep the affine baseline numerically stable (no exp-scale blow-up) so the
    affine<spline ordering is measured cleanly.
    """
    centers = np.array(
        [[1.5 * math.cos(2 * math.pi * k / 8), 1.5 * math.sin(2 * math.pi * k / 8)]
         for k in range(8)],
        dtype=np.float64,
    )
    idx = rng.integers(0, 8, size=n)
    x = centers[idx] + rng.normal(scale=0.30, size=(n, 2))
    return x.astype(np.float32)


TARGETS = {
    "checkerboard": _sample_checkerboard,
    "moons": _sample_two_moons,
    "circles": _sample_two_circles,
    "pinwheel": _sample_pinwheel,
    "8gaussians": _sample_eight_gaussians,
}


def make_dataset_arrays(target: str, n_train: int, n_test: int, seed: int) -> dict:
    """FIXED train/test samples for `target`, as plain numpy arrays.

    Byte-identical sampling procedure to the pre-fix
    ``common.py:make_dataset`` (same RNG streams: train uses ``seed``, test
    uses ``seed + 10_000``), so honest downstream results are unchanged.
    Returns a flat dict with train_x / test_x keys ready to ``np.savez``.
    """
    if target not in TARGETS:
        raise KeyError(f"unknown normflows-density target {target!r}; choose from {list(TARGETS)}")
    fn = TARGETS[target]
    rng_tr = np.random.default_rng(seed)
    rng_te = np.random.default_rng(seed + 10_000)
    x_tr = fn(n_train, rng_tr)
    x_te = fn(n_test, rng_te)
    return {"train_x": x_tr, "test_x": x_te}
