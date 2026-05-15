"""Adaptive Quantile Clipping baseline (Andrew et al., NeurIPS 2021).

Instead of using a fixed clipping threshold, this method dynamically adjusts
the clipping norm toward a target quantile (e.g., median) of the observed
per-sample gradient norm distribution. This benchmark implementation uses
the same geometric threshold update inside the existing DP-SGD harness.

The clipping threshold is updated each step via exponential moving average:
  C_t = C_{t-1} * exp(-lr_C * (frac_clipped - target_quantile))

where frac_clipped is the fraction of per-sample gradients exceeding C_{t-1}.

Reference:
  Andrew et al., "Differentially Private Learning with Adaptive Clipping",
  NeurIPS 2021.
  https://arxiv.org/abs/1905.03871
"""

_FILE = "opacus/custom_dpsgd.py"

_CONTENT = """\
class DPMechanism:
    \"\"\"Adaptive Quantile Clipping (Andrew et al., NeurIPS 2021).

    Dynamically adjusts clipping threshold to target quantile of gradient norms.
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

        # Adaptive clipping parameters for the Andrew et al. update rule.
        self.clip_norm = max_grad_norm  # Initial clipping threshold
        self.target_quantile = 0.5  # Target: median of gradient norms
        self.clip_lr = 0.2  # Learning rate for clipping threshold adaptation
        self.clip_min = 0.01  # Minimum clipping threshold
        self.clip_max = 100.0  # Maximum clipping threshold

    def clip_and_noise(self, per_sample_grads, step, epoch):
        batch_size = per_sample_grads[0].shape[0]

        # Compute per-sample gradient norms
        flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
        norms = flat.norm(2, dim=1)  # [B]

        # Compute fraction of samples exceeding current clip norm
        frac_above = (norms > self.clip_norm).float().mean().item()

        # Update clipping threshold using geometric update
        # If too many gradients are clipped, increase threshold; if too few, decrease
        self.clip_norm = self.clip_norm * math.exp(
            self.clip_lr * (frac_above - self.target_quantile)
        )
        self.clip_norm = max(self.clip_min, min(self.clip_max, self.clip_norm))

        # Clip per-sample gradients using adaptive threshold
        clip_factor = (self.clip_norm / norms.clamp(min=1e-8)).clamp(max=1.0)

        noised_grads = []
        for g in per_sample_grads:
            shape = [batch_size] + [1] * (g.dim() - 1)
            clipped = g * clip_factor.reshape(shape)

            # Average over batch
            avg = clipped.mean(dim=0)

            # Add noise calibrated to current clip norm
            noise = torch.randn_like(avg) * (
                self.noise_multiplier * self.clip_norm / batch_size
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
