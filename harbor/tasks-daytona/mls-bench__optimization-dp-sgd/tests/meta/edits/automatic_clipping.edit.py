"""Automatic Clipping (AUTO-S) baseline (Bu et al., NeurIPS 2023).

Replaces fixed-threshold clipping with per-sample gradient normalization.
Instead of clipping to a fixed norm C, each per-sample gradient is divided
by its own norm (plus a small constant gamma), making the clipping threshold
implicit and removing it as a hyperparameter entirely.

AUTO-S: g_clipped_i = g_i / (||g_i|| + gamma)

This has the same privacy guarantee as standard DP-SGD with C=1 because
the sensitivity of the normalized gradient is bounded by 1.

Reference:
  Bu et al., "Automatic Clipping: Differentially Private Deep Learning
  Made Easier and Stronger", NeurIPS 2023.
  https://arxiv.org/abs/2206.07136
"""

_FILE = "opacus/custom_dpsgd.py"

_CONTENT = """\
class DPMechanism:
    \"\"\"AUTO-S Automatic Clipping (Bu et al., NeurIPS 2023).

    Per-sample gradient normalization: g_i / (||g_i|| + gamma).
    Sensitivity bounded by 1, no clipping threshold to tune.
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
        # AUTO-S gamma for this benchmark harness. gamma=1.0 keeps the
        # existing learning-rate schedule stable.
        self.gamma = 1.0

    def clip(self, per_sample_grads, step, epoch):
        batch_size = per_sample_grads[0].shape[0]

        # Compute per-sample gradient norms
        flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
        norms = flat.norm(2, dim=1)  # [B]

        # AUTO-S normalization: scale each gradient by 1/(||g_i|| + gamma)
        # This bounds sensitivity to 1 (since ||g_i / (||g_i|| + gamma)|| <= 1)
        scale = 1.0 / (norms + self.gamma)  # [B]

        # Sensitivity bound C=1 for AUTO-S: the harness adds noise sigma * 1 / B
        return scale, 1.0

    def get_noise_multiplier(self, step, epoch):
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
