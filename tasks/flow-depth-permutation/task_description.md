# Affine Flow Depth With Fixed Mixing

Choose the number of affine coupling transforms on the two-dimensional moons
density. Only depth varies in this question; the permutation is deliberately
fixed so depth is not confounded with a second architectural change.

Edit `normflows-density/solution/depth.py` and implement `select_depth() -> int`.
The value must be in `[1, 32]`. For a selected depth `d`, the verifier builds
exactly `d` affine couplings and `d` swap permutations. Conditioner width, base
distribution, optimizer, data, seed, and budget are fixed.

The single scored setting uses 30,000 training samples, 30,000 verifier-only
test samples, seed 42, and 20,000 Adam optimizer steps on one CUDA GPU. It
reports exact NLL; lower is preferred. A metric is accepted only after the
complete ordered protocol and terminal success proof validate.
