"""Mid-edit operations for the robo-diffusion-policy task.

Applied after pre_edit, before the agent starts. Creates:
  - CleanDiffuser/pipelines/custom_policy.py (editable algorithm file)
  - CleanDiffuser/configs/custom/mujoco/{mujoco.yaml, task/*.yaml}
The base config mirrors dql/mujoco with per-env hyperparameters required by all
three baselines (weight_temperature for DQL/IDQL, eta for DQL).
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

_BASE_CONFIG = """defaults:
  - _self_
  - task: hopper-medium-v2

pipeline_name: custom_policy
mode: train
seed: 42
device: cuda:0

# Environment
normalize_reward: True
discount: 0.99

# Actor
solver: ddpm
diffusion_steps: 5
sampling_steps: 5
predict_noise: True
ema_rate: 0.995
actor_learning_rate: 0.0003

# Critic
hidden_dim: 256
critic_learning_rate: 0.0003

# IQL (used by idql baseline only; harmless as unused keys for DQL)
iql_tau: 0.7
actor_hidden_dim: 256
actor_n_blocks: 3
actor_dropout: 0.1
critic_hidden_dim: 256

# Training
gradient_steps: 1000000
batch_size: 256
ema_update_interval: 5
log_interval: 1000
save_interval: 100000

# Inference
ckpt: latest
num_envs: 50
num_episodes: 3
num_candidates: 50
temperature: 0.5
use_ema: True

# hydra
hydra:
  job:
    chdir: false
"""

_HOPPER_CONFIG = """env_name: "hopper-medium-v2"

weight_temperature: 100.
eta: 1.0
"""

_WALKER2D_CONFIG = """env_name: "walker2d-medium-v2"

weight_temperature: 300.
eta: 1.0
"""

_HALFCHEETAH_CONFIG = """env_name: "halfcheetah-medium-v2"

weight_temperature: 50.
eta: 1.0
"""

OPS = [
    {
        "op": "create",
        "file": "CleanDiffuser/pipelines/custom_policy.py",
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
