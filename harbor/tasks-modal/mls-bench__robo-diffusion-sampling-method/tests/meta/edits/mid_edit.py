"""Mid-edit operations for the robo-diffusion-sampling-method task.

Applied after pre_edit, before the agent starts. Creates:
  - CleanDiffuser/pipelines/custom_sampling_method.py (editable algorithm file)
  - CleanDiffuser/configs/custom/mujoco/{mujoco.yaml, task/*.yaml}

The base config defaults to the DDPM sampler with 100 steps (the template is a
DQL-shaped diffusion policy; switching samplers means changing `solver` and
`sampling_steps` in the YAML). This matches how cleandiffuser's own pipelines
select samplers in their per-algo YAMLs.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

_BASE_CONFIG = """defaults:
  - _self_
  - task: hopper-medium-v2

pipeline_name: custom_sampling_method
mode: train
seed: 42
device: cuda:0

# Environment
normalize_reward: True
discount: 0.99

# Actor
solver: ddpm
diffusion_steps: 100
sampling_steps: 100
predict_noise: True
ema_rate: 0.995
actor_learning_rate: 0.0003

# Critic
hidden_dim: 256
critic_learning_rate: 0.0003

# Training
gradient_steps: 100000
batch_size: 256
ema_update_interval: 5
log_interval: 1000
save_interval: 50000

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
        "file": "CleanDiffuser/pipelines/custom_sampling_method.py",
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
