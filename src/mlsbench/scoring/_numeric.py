"""Strict numeric predicates shared by scoring modules."""

from __future__ import annotations

import math
from numbers import Real


def is_finite_real(value: object) -> bool:
    """Return whether *value* is an actual finite real number.

    Numeric-looking strings and booleans are deliberately excluded. Accepting
    them during validation leaves downstream arithmetic with values of a
    different type than the validator promised.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False
