"""Pure math primitives for score normalization.

Two normalization functions:
- bounded_power: metrics with a theoretical bound (accuracy, loss, FID, ...)
- sigmoid_score: metrics without a theoretical bound (reward, throughput, ...)

Plus constraint penalty functions for hard requirements (e.g., cost <= 25).
"""

from __future__ import annotations

import math

from mlsbench.scoring._numeric import is_finite_real

GAMMA_MIN = 0.1
GAMMA_MAX = 10.0


# ---------------------------------------------------------------------------
# bounded_power
# ---------------------------------------------------------------------------

def bounded_power(x: float, floor: float, bound: float, gamma: float) -> float:
    """Normalize *x* into [0, 1] via a power curve between *floor* and *bound*.

    After direction unification (higher-is-better):

    * When ``bound`` is on the "better" side of ``floor`` (``bound > floor``) —
      the standard case used for metrics with a well-defined theoretical
      ceiling such as loss (``bound=0``) — the curve is::

          r = clip((x - floor) / (bound - floor), 0, 1)
          score = r ** gamma

      i.e. ``floor`` maps to score 0 and ``bound`` maps to score 1.

    * When ``bound`` is on the "worse" side of ``floor`` (``bound < floor``) —
      the pattern used for higher-is-better metrics where the spec anchors
      ``bound`` at a hard sanity floor such as random-guessing accuracy
      (``bound=25`` for arc_easy / hellaswag) — the score is inverted so that
      ``floor`` (best-baseline reference) still maps to 1 and ``bound``
      (random floor) maps to 0. Values worse than ``bound`` clip to 0
      instead of being silently inverted to 1.
    """
    if not all(is_finite_real(v) for v in (x, floor, bound, gamma)) or gamma <= 0.0:
        return 0.0
    if bound == floor:
        return 0.0
    if bound > floor:
        # Standard orientation: floor=worst, bound=best.
        r = (x - floor) / (bound - floor)
    else:
        # Inverted spec convention: bound is a hard sanity floor (worse side),
        # floor anchors the baseline reference.
        r = (x - bound) / (floor - bound)
    r = max(0.0, min(1.0, r))
    return r ** gamma


def solve_gamma(floor: float, bound: float, ref: float, ref_score: float) -> float:
    """Solve gamma such that ``bounded_power(ref, floor, bound, gamma) == ref_score``.

    The supported calibration range is [GAMMA_MIN, GAMMA_MAX]. Invalid,
    degenerate, or out-of-range calibration anchors raise instead of silently
    changing the score curve.
    """
    if not all(is_finite_real(v) for v in (floor, bound, ref, ref_score)):
        raise ValueError("bounded_power calibration values must be finite")
    if bound == floor:
        raise ValueError("bounded_power floor and bound are identical")
    # Mirror the inverted-spec convention used in ``bounded_power``.
    if bound > floor:
        r_ref = (ref - floor) / (bound - floor)
    else:
        r_ref = (ref - bound) / (floor - bound)
    # Degenerate: ref at floor or at bound
    if r_ref <= 0.0 or r_ref >= 1.0:
        raise ValueError(
            f"bounded_power reference is degenerate: r(ref)={r_ref:.4f} "
            f"(ref={ref}, floor={floor}, bound={bound})"
        )
    if not 0.0 < ref_score < 1.0:
        raise ValueError(f"bounded_power ref_score must be in (0, 1), got {ref_score}")
    gamma = math.log(ref_score) / math.log(r_ref)
    if not math.isfinite(gamma) or not GAMMA_MIN <= gamma <= GAMMA_MAX:
        raise ValueError(
            f"bounded_power gamma {gamma} is outside supported range "
            f"[{GAMMA_MIN}, {GAMMA_MAX}]"
        )
    return gamma


# ---------------------------------------------------------------------------
# sigmoid_score
# ---------------------------------------------------------------------------

def logistic_score(y: float, midpoint: float, scale: float) -> float:
    """Normalize *y* into (0, 1) with a pure logistic curve.

    ``midpoint`` maps to 0.5. Unlike ``sigmoid_score`` this has no hard floor,
    so values below the midpoint remain ordered instead of clipping to zero.
    """
    if not all(is_finite_real(v) for v in (y, midpoint, scale)) or scale <= 0:
        return 0.0
    z = (y - midpoint) / scale
    if z > 30:
        return 1.0
    if z < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_score(y: float, floor: float, scale: float) -> float:
    """Normalize *y* into [0, 1) via a shifted sigmoid.

    score(y) = 2 * sigma((y - floor) / scale) - 1   for y >= floor
    score(y) = 0                                      for y < floor

    Maps floor -> 0, approaches 1 as y -> +inf.
    """
    if not all(is_finite_real(v) for v in (y, floor, scale)) or scale <= 0:
        return 0.0
    if y <= floor:
        return 0.0
    z = (y - floor) / scale
    # Prevent overflow in exp
    if z > 30:
        return 1.0
    sig = 1.0 / (1.0 + math.exp(-z))
    return 2.0 * sig - 1.0


def solve_scale(floor: float, ref: float, ref_score: float) -> float:
    """Solve *scale* such that ``sigmoid_score(ref, floor, scale) == ref_score``.

    From: ref_score = 2 * sigma((ref - floor) / scale) - 1
    =>    sigma(z) = (ref_score + 1) / 2
    =>    z = logit((ref_score + 1) / 2)
    =>    scale = (ref - floor) / z
    """
    if not all(is_finite_real(v) for v in (floor, ref, ref_score)):
        raise ValueError("sigmoid calibration values must be finite")
    if ref <= floor:
        raise ValueError(
            f"sigmoid reference must be above floor, got ref={ref}, floor={floor}"
        )
    if not 0.0 < ref_score < 1.0:
        raise ValueError(f"sigmoid ref_score must be in (0, 1), got {ref_score}")
    p = (ref_score + 1.0) / 2.0
    # logit(p) = log(p / (1-p))
    z = math.log(p / (1.0 - p))
    if z <= 0:
        raise ValueError(f"sigmoid calibration produced invalid logit {z}")
    scale = (ref - floor) / z
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"sigmoid calibration produced invalid scale {scale}")
    return scale


# ---------------------------------------------------------------------------
# Constraint penalties
# ---------------------------------------------------------------------------

def penalty_upper(x: float, target: float, sharpness: float = 0.15) -> float:
    """Penalty for upper-bound constraint ``x <= target``.

    Returns 1.0 if satisfied, exponential decay otherwise.
    """
    if not all(is_finite_real(v) for v in (x, target, sharpness)) or sharpness <= 0.0:
        return 0.0
    if x <= target:
        return 1.0
    return math.exp(-sharpness * (x - target))


def penalty_lower(x: float, target: float, sharpness: float = 0.15) -> float:
    """Penalty for lower-bound constraint ``x >= target``.

    Returns 1.0 if satisfied, exponential decay otherwise.
    """
    if not all(is_finite_real(v) for v in (x, target, sharpness)) or sharpness <= 0.0:
        return 0.0
    if x >= target:
        return 1.0
    return math.exp(-sharpness * (target - x))


# ---------------------------------------------------------------------------
# Direction + transform helpers
# ---------------------------------------------------------------------------

def _identity(x: float) -> float:
    return x


def _strict_log(x: float) -> float:
    if x <= 0.0:
        raise ValueError(f"log transform requires x > 0, got {x}")
    return math.log(x)


def _strict_log1p(x: float) -> float:
    if x <= -1.0:
        raise ValueError(f"log1p transform requires x > -1, got {x}")
    return math.log1p(x)


TRANSFORMS = {
    "id": _identity,
    "log": _strict_log,
    "log1p": _strict_log1p,
}


def apply_direction_and_transform(
    x: float, direction: str, transform: str,
) -> float:
    """Unify raw metric to higher-is-better internal space.

    y = sign * transform(x)
    """
    if not is_finite_real(x):
        raise ValueError(f"metric value must be finite, got {x}")
    if not isinstance(transform, str):
        raise ValueError(f"Unknown transform: {transform!r}")
    tfn = TRANSFORMS.get(transform)
    if tfn is None:
        raise ValueError(f"Unknown transform: {transform!r}")
    val = tfn(x)
    if not math.isfinite(val):
        raise ValueError(
            f"transform {transform!r} returned a non-finite value for {x}"
        )
    if direction == "lower":
        val = -val
    elif direction != "higher":
        raise ValueError(f"Unknown direction: {direction!r}")
    return val
