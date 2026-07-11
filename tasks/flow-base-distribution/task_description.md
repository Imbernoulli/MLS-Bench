# Flow Base Distribution

Select the base distribution beneath a deliberately minimal flow on the
two-dimensional 8-Gaussians density.

Edit `normflows-density/solution/base_dist.py` and implement
`select_base_distribution() -> str`. Valid values are `gaussian`,
`gaussian_trainable`, and `gmm`. The verifier fixes the learned transform to one
affine coupling followed by one swap permutation, isolating the base-distribution
choice.

The single scored setting uses 30,000 training samples, 30,000 verifier-only
test samples, seed 42, and 20,000 Adam optimizer steps on one CUDA GPU. It
reports exact change-of-variables NLL; lower is preferred. A metric is accepted
only after the complete ordered protocol and terminal success proof validate.
