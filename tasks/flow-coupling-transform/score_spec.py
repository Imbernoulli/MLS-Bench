"""Score spec for flow-coupling-transform.

Held-out test NLL (lower is better) after the complete fixed protocol:
30,000 train samples, 30,000 verifier-only test samples, 20,000 optimizer
steps, seed 42, and one H20. The editable surface is parsed as a literal
AST and never executed. The strict parser requires unique design,
final-step, and metric proof for every configured target.

Fresh endpoints measured on the same immutable source and data checksums:
    checkerboard: affine=3.125589, spline=2.954646
    moons: affine=1.030755, spline=1.025927
    8gaussians: affine=2.386391, spline=2.373422
The spline endpoint maps to 0.5 and the native affine endpoint maps to 0.1
for every setting. Failed or partial anchor rows are not calibration input.
The source evidence file SHA-256 is
``73429c480ad6dc0e8f3fb147668e6195fb3d0fcc173079814f9868b8c18d41ef``.
"""
from mlsbench.scoring.dsl import *

term("nll_checkerboard",
    col("nll_checkerboard").lower().id()
    .sigmoid(ref=const(2.954646), scale=0.077799512058635861))
term("nll_moons",
    col("nll_moons").lower().id()
    .sigmoid(ref=const(1.025927), scale=0.00219731749307721))
term("nll_8gaussians",
    col("nll_8gaussians").lower().id()
    .sigmoid(ref=const(2.373422), scale=0.0059024462650617299))

setting("checkerboard", weighted_mean(("nll_checkerboard", 1.0)))
setting("moons", weighted_mean(("nll_moons", 1.0)))
setting("8gaussians", weighted_mean(("nll_8gaussians", 1.0)))

task(gmean("checkerboard", "moons", "8gaussians"))
