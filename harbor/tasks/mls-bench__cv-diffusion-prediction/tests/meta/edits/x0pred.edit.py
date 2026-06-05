"""X0-prediction baseline (SNR-scaled).

Model predicts sqrt(alpha_t) * x_0 instead of raw x_0. This naturally
balances the loss across timesteps: at high noise (small alpha), the
target is small so the loss doesn't dominate. Equivalent to applying
SNR-proportional weighting to the raw x0 MSE loss.

At inference, recover x_0 = model_output / sqrt(alpha_t).
"""

_FILE = "diffusers-main/custom_train.py"

_X0PRED = '''\
def compute_training_target(x_0, noise, timesteps, schedule):
    # SNR-scaled X0: predict sqrt(alpha) * x_0
    # This balances loss across timesteps — high noise has small target
    sqrt_alpha = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
    return sqrt_alpha * x_0


def predict_x0(model_output, x_t, timesteps, schedule):
    # Recover x_0 from SNR-scaled prediction:
    # model predicts sqrt(alpha) * x_0 => x_0 = output / sqrt(alpha)
    sqrt_alpha = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
    return model_output / sqrt_alpha.clamp(min=1e-8)
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 83,
        "end_line": 118,
        "content": _X0PRED,
    },
]
