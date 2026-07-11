"""Variance-preserving residual addition for two equal-scale branches."""

import math


def residual_step(hidden, block_out):
    return (hidden + block_out) * math.sqrt(0.5)
