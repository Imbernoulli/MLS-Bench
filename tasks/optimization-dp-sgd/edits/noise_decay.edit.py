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

    Privacy accounting: sigma_0 is calibrated so that the full decayed
    schedule spends the target budget under the harness's accountant
    (compute_epsilon_schedule); the fixed harness composes the per-step
    sigma it actually applied.
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

        # Per-epoch sigma schedule for the RDP accounting. Steps per epoch =
        # dataset_size // batch_size (drop_last=True in DataLoader).
        self.steps_per_epoch = dataset_size // batch_size
        q = batch_size / dataset_size

        def schedule(sigma_0):
            return [(sigma_0 * (self.noise_decay_factor ** ((e - 1) // self.decay_interval)),
                     self.steps_per_epoch) for e in range(1, epochs + 1)]

        # Calibrate sigma_0 by bisection: the smallest sigma_0 whose decayed
        # schedule stays within the budget. sigma_0 = noise_multiplier (the
        # calibrated uniform sigma) over-spends once sigma decays; at the upper
        # end every step's sigma is >= noise_multiplier.
        lo = noise_multiplier
        hi = noise_multiplier / self.noise_decay_factor ** ((epochs - 1) // self.decay_interval)
        while hi - lo > 1e-4 * hi:
            mid = (lo + hi) / 2
            eps, _ = compute_epsilon_schedule(schedule(mid), q, target_delta)
            if eps > target_epsilon:
                lo = mid
            else:
                hi = mid
        self.sigma_0 = hi
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
