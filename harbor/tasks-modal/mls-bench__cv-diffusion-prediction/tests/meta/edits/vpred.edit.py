"""V-prediction baseline.

Model predicts velocity v = sqrt(alpha) * eps - sqrt(1-alpha) * x_0.
From Salimans & Ho (2022) "Progressive Distillation for Fast Sampling".
Used in Stable Diffusion v2.
"""

_FILE = "diffusers-main/custom_train.py"

_VPRED = '''\
def compute_training_target(x_0, noise, timesteps, schedule):
    # V-prediction: v = sqrt(alpha) * noise - sqrt(1-alpha) * x_0
    sqrt_alpha = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)
    return sqrt_alpha * noise - sqrt_one_minus_alpha * x_0


def predict_x0(model_output, x_t, timesteps, schedule):
    # Recover x_0 from v-prediction:
    # v = sqrt(alpha) * eps - sqrt(1-alpha) * x_0
    # x_t = sqrt(alpha) * x_0 + sqrt(1-alpha) * eps
    # => x_0 = sqrt(alpha) * x_t - sqrt(1-alpha) * v
    sqrt_alpha = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)
    return sqrt_alpha * x_t - sqrt_one_minus_alpha * model_output
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 83,
        "end_line": 118,
        "content": _VPRED,
    },
]
