# Flow Learning Rate

Choose the Adam learning rate for a fixed spline-coupling flow on the
two-dimensional moons density.

Edit `normflows-density/solution/learning_rate.py` and implement
`select_learning_rate() -> float`. The value must be finite and in `[1e-6, 1]`.
The verifier builds eight spline couplings and eight LU permutations. Batch size,
optimizer family, data, seed, and optimizer-step budget are fixed.

The single scored setting uses 30,000 training samples, 30,000 verifier-only
test samples, seed 42, batch size 512, and exactly 20,000 Adam optimizer steps on
one CUDA GPU. It reports exact NLL; lower is preferred. A metric is accepted
only after the complete ordered protocol and terminal success proof validate.
