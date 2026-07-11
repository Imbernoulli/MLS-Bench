# Canonical Flow Recipe Family

Select a canonical normalizing-flow recipe for the two-dimensional pinwheel
density. This is a recipe-level comparison: affine uses RealNVP-style swap
mixing, while MAF and spline use learnable LU mixing.

Edit `normflows-density/solution/arch.py` and implement
`select_architecture() -> str`. Valid values are `affine`, `maf`, and `spline`.
Every recipe contains eight density transforms and eight between-transform
permutations. Base distribution, width, optimizer, data, seed, and budget are
fixed.

The single scored setting uses 30,000 training samples, 30,000 verifier-only
test samples, seed 42, and 20,000 Adam optimizer steps on one CUDA GPU. It
reports exact change-of-variables NLL; lower is preferred. A metric is accepted
only after the complete ordered protocol and terminal success proof validate.
