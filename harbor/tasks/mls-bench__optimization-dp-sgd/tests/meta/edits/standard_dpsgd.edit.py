"""Standard DP-SGD baseline (Abadi et al., 2016).

This is the canonical DP-SGD algorithm: fixed per-sample gradient clipping
with constant Gaussian noise addition.

Reference:
  Abadi et al., "Deep Learning with Differential Privacy", CCS 2016.
  https://arxiv.org/abs/1607.00133

This matches the default template implementation — standard flat clipping
to max_grad_norm, noise calibrated as sigma * max_grad_norm / batch_size.
"""

_FILE = "opacus/custom_dpsgd.py"

_CONTENT = """\
class DPMechanism:
    \"\"\"Standard DP-SGD (Abadi et al., 2016).

    Fixed per-sample gradient clipping + constant Gaussian noise.
    \"\"\"

    def __init__(self, max_grad_norm, noise_multiplier, n_params,
                 dataset_size, batch_size, epochs, target_epsilon, target_delta):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.n_params = n_params
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

    def clip_and_noise(self, per_sample_grads, step, epoch):
        batch_size = per_sample_grads[0].shape[0]

        # Compute per-sample gradient norms (flat norm across all parameters)
        flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
        norms = flat.norm(2, dim=1)  # [B]

        # Clip per-sample gradients
        clip_factor = (self.max_grad_norm / norms.clamp(min=1e-8)).clamp(max=1.0)  # [B]

        noised_grads = []
        for g in per_sample_grads:
            shape = [batch_size] + [1] * (g.dim() - 1)
            clipped = g * clip_factor.reshape(shape)

            # Average over batch
            avg = clipped.mean(dim=0)

            # Add calibrated Gaussian noise
            noise = torch.randn_like(avg) * (
                self.noise_multiplier * self.max_grad_norm / batch_size
            )
            noised_grads.append(avg + noise)

        return noised_grads

    def get_effective_sigma(self, step, epoch):
        return self.noise_multiplier
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 152,
        "end_line": 233,
        "content": _CONTENT,
    },
]
