"""X0-prediction baseline.

Model directly predicts the clean image x_0.
Simple but can have unstable gradients at high noise levels.
"""

_FILE = "diffusers-main/custom_train.py"

_X0PRED = '''\
def compute_training_target(x_0, noise, timesteps, schedule):
    # X0-prediction: model directly predicts the clean image
    return x_0


def predict_x0(model_output, x_t, timesteps, schedule):
    # Model output IS x_0, no conversion needed
    return model_output
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
