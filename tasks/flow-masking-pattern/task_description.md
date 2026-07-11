# RealNVP Masking Pattern

Choose the binary-mask sequence for a fixed masked affine flow on the
two-dimensional moons density.

Edit `normflows-density/solution/masking_pattern.py` and implement
`select_masks()`. Return exactly eight length-two lists containing one `0` and
one `1` each. The verifier builds eight masked affine transforms and no separate
permutation modules. Width, base distribution, optimizer, data, seed, and budget
are fixed.

The single scored setting uses 30,000 training samples, 30,000 verifier-only
test samples, seed 42, and 20,000 Adam optimizer steps on one CUDA GPU. It
reports exact NLL; lower is preferred. A metric is accepted only after the
complete ordered protocol and terminal success proof validate.
