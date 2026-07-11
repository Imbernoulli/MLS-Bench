# Spline Bin Count

Choose the rational-quadratic spline resolution for a fixed coupling flow on the
two-dimensional checkerboard density.

Edit `normflows-density/solution/spline_bins.py` and implement
`select_spline_bins() -> int`. The value must be in `[2, 64]`. The verifier
builds eight spline couplings and eight LU permutations. Width, base
distribution, optimizer, data, seed, and budget are fixed.

The single scored setting uses 30,000 training samples, 30,000 verifier-only
test samples, seed 42, and 20,000 Adam optimizer steps on one CUDA GPU. It
reports exact NLL; lower is preferred. A metric is accepted only after the
complete ordered protocol and terminal success proof validate.
