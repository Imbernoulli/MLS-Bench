# Transform Family Under Fixed LU Mixing

Compare affine coupling, masked autoregressive, and spline-coupling transforms
on the two-dimensional 8-Gaussians density while holding the mixing construction
constant. Unlike the canonical-recipe task, every choice here uses the same LU
permutation after each transform, so this is a controlled transform-family
ablation rather than a second recipe comparison.

Edit `normflows-density/solution/autoregressive_coupling.py` and implement
`select_conditioner() -> str`. Valid values are `affine`, `maf`, and `spline`.
The verifier builds eight selected transforms and eight LU permutations. Base
distribution, width, optimizer, data, seed, and budget are fixed.

The single scored setting uses 30,000 training samples, 30,000 verifier-only
test samples, seed 42, and 20,000 Adam optimizer steps on one CUDA GPU. It
reports exact NLL; lower is preferred. A metric is accepted only after the
complete ordered protocol and terminal success proof validate.
