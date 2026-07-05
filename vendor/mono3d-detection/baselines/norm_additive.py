"""mono3d-depth-normalization WEAK baseline: RAW ADDITIVE residual (metres).

Apply the learned depth correction ADDITIVELY in raw metres on top of the geometry depth:
Z = f*H/h2d + residual_metres, where the residual is an unbounded scalar. Because the metric
depth spans 6-40 m, a single additive residual is badly scaled — a correction useful at near
(small absolute error) is far too small at far and vice-versa, and the unbounded additive term
can push Z negative / explode. Poorly-scaled correction -> higher depth error, especially at
range. Reference: monocular-depth heads parameterize the residual in LOG space precisely to be
scale-invariant across distance.
"""
import torch


def build_depth_norm():
    def apply(geom_Z, raw):
        return geom_Z + raw[:, 0]            # raw additive metres (badly scaled, can go <=0)

    return apply
