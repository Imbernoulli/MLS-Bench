# Flow Conditioner Width

Choose the MLP width of each affine-coupling conditioner on the
two-dimensional checkerboard density.

Edit `normflows-density/solution/conditioner_width.py` and implement
`select_conditioner_width() -> int`. The value must be in `[2, 512]`. The
verifier builds eight affine couplings and eight LU permutations. Flow family,
depth, base distribution, optimizer, data, seed, and budget are fixed.

The single scored setting uses 30,000 training samples, 30,000 verifier-only
test samples, seed 42, and 20,000 Adam optimizer steps on one CUDA GPU. It
reports exact NLL; lower is preferred. A metric is accepted only after the
complete ordered protocol and terminal success proof validate.
