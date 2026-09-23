"""Step-Decay Noise Schedule baseline (inspired by Global-Adapt-V2-S, 2025).

Uses a step-decay noise multiplier that decreases over training epochs,
combined with a step-decay clipping threshold. The key insight is that
gradient magnitudes tend to decrease as training progresses, so early
training can tolerate more noise (gradients are large/informative) while
later training benefits from less noise (gradients are small/refined).

The noise multiplier decays by a factor at each decay milestone:
  sigma_t = sigma_0 * decay_factor^(epoch // decay_interval)

The clipping threshold also decays to track the decreasing gradient norms:
  C_t = C_0 * clip_decay^(epoch // decay_interval)

This allocates more of the privacy budget to later epochs where it matters most.

Reference:
  DP-SGD-Global-Adapt-V2-S: "Triad improvements of privacy, accuracy and
  fairness via step decay noise multiplier and step decay upper clipping
  threshold", Electronic Commerce Research and Applications, 2025.
  https://arxiv.org/abs/2312.02400
"""

_FILE = "opacus/custom_dpsgd.py"

_CONTENT = """\
class DPMechanism:
    \"\"\"Step-Decay Noise Schedule (inspired by Global-Adapt-V2-S, 2025).

    Decays noise multiplier and clipping threshold over training epochs
    to allocate more privacy budget to later (more useful) training steps.

    Privacy accounting: sigma_0 is chosen so that the full schedule spends
    the same budget as the calibrated uniform sigma; the fixed harness
    composes the per-step sigma it actually applied.
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

        # Step-decay schedule parameters
        # Decay noise and clipping every decay_interval epochs
        self.decay_interval = max(1, epochs // 4)  # 4 decay stages
        self.noise_decay_factor = 0.8  # Reduce noise by 20% at each stage
        self.clip_decay_factor = 0.85  # Reduce clip norm by 15% at each stage

        # Pre-compute the per-epoch sigma schedule so we can do accurate
        # RDP accounting.  Steps per epoch = dataset_size // batch_size
        # (drop_last=True in DataLoader).
        self.steps_per_epoch = dataset_size // batch_size

        # Compute sigma_0: scale the calibrated (uniform) sigma up so that
        # the harmonic-mean-equivalent sigma across all steps equals the
        # calibrated value.  This keeps the total privacy spend equal to
        # the budget even though individual steps have different noise.
        total_steps = self.steps_per_epoch * epochs
        inv_sq_sum = 0.0
        for e in range(1, epochs + 1):
            stage = (e - 1) // self.decay_interval
            factor = self.noise_decay_factor ** stage
            # Each epoch contributes steps_per_epoch steps at sigma_0*factor
            # 1/sigma_t^2 = 1/(sigma_0*factor)^2 = 1/(sigma_0^2 * factor^2)
            inv_sq_sum += self.steps_per_epoch / (factor * factor)
        # sigma_eff = sqrt(total_steps / inv_sq_sum) * sigma_0
        # We want sigma_eff == noise_multiplier (the calibrated value), so:
        #   noise_multiplier = sigma_0 * sqrt(total_steps / inv_sq_sum)
        #   sigma_0 = noise_multiplier / sqrt(total_steps / inv_sq_sum)
        #           = noise_multiplier * sqrt(inv_sq_sum / total_steps)
        self.sigma_0 = noise_multiplier * (inv_sq_sum / total_steps) ** 0.5
        self.clip_0 = max_grad_norm

        # Current values
        self._current_sigma = self.sigma_0
        self._current_clip = self.clip_0

    def clip(self, per_sample_grads, step, epoch):
        batch_size = per_sample_grads[0].shape[0]

        # Update schedule based on epoch
        stage = (epoch - 1) // self.decay_interval
        self._current_sigma = self.sigma_0 * (self.noise_decay_factor ** stage)
        self._current_clip = self.clip_0 * (self.clip_decay_factor ** stage)

        # Compute per-sample gradient norms
        flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
        norms = flat.norm(2, dim=1)  # [B]

        # Clip per-sample gradients using current (decayed) threshold
        clip_factor = (self._current_clip / norms.clamp(min=1e-8)).clamp(max=1.0)

        # The harness adds noise calibrated to the current clip norm and sigma
        return clip_factor, self._current_clip

    def get_noise_multiplier(self, step, epoch):
        \"\"\"Current (decayed) noise multiplier; the harness accounts each
        step with the sigma it actually applied.\"\"\"
        return self._current_sigma
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
