"""Epsilon prediction baseline (standard DDPM).

Model predicts the noise epsilon. This is the original parameterization
from Ho et al. (2020) "Denoising Diffusion Probabilistic Models".
"""

_FILE = "diffusers-main/custom_train.py"

_EPSILON = '''\
def compute_training_target(x_0, noise, timesteps, schedule):
    # Epsilon prediction: model learns to predict the added noise
    return noise


def predict_x0(model_output, x_t, timesteps, schedule):
    # Recover x_0 from epsilon prediction:
    # x_t = sqrt(alpha) * x_0 + sqrt(1-alpha) * eps
    # => x_0 = (x_t - sqrt(1-alpha) * eps) / sqrt(alpha)
    sqrt_alpha = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)
    return (x_t - sqrt_one_minus_alpha * model_output) / sqrt_alpha.clamp(min=1e-8)
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 83,
        "end_line": 118,
        "content": _EPSILON,
    },
]
