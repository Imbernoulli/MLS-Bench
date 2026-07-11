# Flow Training Batch Size

Choose the batch size for a fixed spline-coupling flow on the two-dimensional
checkerboard density.

Edit `normflows-density/solution/batch_size.py` and implement
`select_batch_size() -> int`. The value must be in `[1, 8192]`. The verifier
builds eight spline couplings and eight LU permutations. Architecture, optimizer
family, learning rate, data, seed, and optimizer-step budget are fixed. Because
the comparison fixes optimizer steps rather than epochs, batch size also changes
the number of sampled training examples; the terminal proof reports that count.

The single scored setting uses 30,000 training samples, 30,000 verifier-only
test samples, seed 42, and exactly 20,000 Adam optimizer steps on one CUDA GPU.
It reports exact NLL; lower is preferred. A metric is accepted only after the
complete ordered protocol and terminal success proof validate.
