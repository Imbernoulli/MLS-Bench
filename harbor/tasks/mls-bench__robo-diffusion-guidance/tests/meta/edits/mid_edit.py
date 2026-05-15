"""Mid-edit operations for the robo-diffusion-guidance task.

Applied to the CleanDiffuser workspace after pre_edit, before the agent starts.
Creates:
  - CleanDiffuser/pipelines/custom_guidance.py (the agent's editable algorithm file)
  - CleanDiffuser/configs/custom_guidance/mujoco/{mujoco.yaml, task/*.yaml}
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

_BASE_CONFIG = """defaults:
  - _self_
  - task: hopper-medium-v2

pipeline_name: custom_guidance
mode: train
seed: 42
device: cuda:0

# Environment
terminal_penalty: -100
discount: 0.997

# Diffuser
solver: ddpm
model_dim: 32
diffusion_steps: 20
sampling_steps: 20
predict_noise: False
action_loss_weight: 10.
ema_rate: 0.9999

# CFG (used by classifier-free guidance variants; ignored by CG default)
label_dropout: 0.25

# Training
diffusion_gradient_steps: 100000
classifier_gradient_steps: 100000
batch_size: 256
log_interval: 1000
save_interval: 50000

# Inference
ckpt: latest
num_envs: 10
num_episodes: 10
num_candidates: 64
temperature: 0.5
use_ema: True

# hydra
hydra:
  job:
    chdir: false
"""

_HOPPER_CONFIG = """env_name: "hopper-medium-v2"
dim_mult: [1, 2, 2, 2]
# Classifier-guidance weight (used by default/CG baseline)
w_cg: 0.3
# Classifier-free-guidance weight + target return (used by cfg/decision_diffuser baselines)
w_cfg: 4.4
target_return: 0.7
horizon: 32
"""

_WALKER2D_CONFIG = """env_name: "walker2d-medium-v2"
dim_mult: [1, 2, 2, 2]
w_cg: 0.007
w_cfg: 6.0
target_return: 0.75
horizon: 32
"""

_HALFCHEETAH_CONFIG = """env_name: "halfcheetah-medium-v2"
dim_mult: [1, 4, 2]
w_cg: 0.0001
w_cfg: 3.2
target_return: 1.1
horizon: 4
"""

OPS = [
    {
        "op": "create",
        "file": "CleanDiffuser/pipelines/custom_guidance.py",
        "content": _CUSTOM_PY,
    },
    {
        "op": "create",
        "file": "CleanDiffuser/configs/custom/mujoco/mujoco.yaml",
        "content": _BASE_CONFIG,
    },
    {
        "op": "create",
        "file": "CleanDiffuser/configs/custom/mujoco/task/hopper-medium-v2.yaml",
        "content": _HOPPER_CONFIG,
    },
    {
        "op": "create",
        "file": "CleanDiffuser/configs/custom/mujoco/task/walker2d-medium-v2.yaml",
        "content": _WALKER2D_CONFIG,
    },
    {
        "op": "create",
        "file": "CleanDiffuser/configs/custom/mujoco/task/halfcheetah-medium-v2.yaml",
        "content": _HALFCHEETAH_CONFIG,
    },
]
