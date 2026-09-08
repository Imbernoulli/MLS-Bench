"""Mid-edit operations for inverse-diffusion-algo.
Creates algo/custom.py from template and configs/algorithm/custom.yaml.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

_CUSTOM_YAML = """\
name: Custom
method:
  _target_: algo.custom.Custom
  diffusion_scheduler_config:
    num_steps: 1000
    schedule: 'vp'
    timestep: 'vp'
    scaling: 'vp'
  guidance_scale: 10.0
  sde: true
  num_optim_steps: 1000
  observation_weight: 1.0
  base_lambda: 0.25
  base_lr: 0.5
  num_mc_samples: 10
"""

OPS = [
    {
        "op": "create",
        "file": "InverseBench/algo/custom.py",
        "content": _CUSTOM_PY,
    },
    {
        "op": "create",
        "file": "InverseBench/configs/algorithm/custom.yaml",
        "content": _CUSTOM_YAML,
    },
    # Ensure checkpoints/cache directories exist in workspace so that
    # data_bind mount targets are valid even when fuse-overlayfs is
    # contended (under heavy parallelism). Without these, apptainer
    # falls back to overlay creation which can timeout and abort
    # container creation with "destination doesn't exist".
    {
        "op": "create",
        "file": "InverseBench/checkpoints/.gitkeep",
        "content": "",
    },
    {
        "op": "create",
        "file": "InverseBench/cache/.gitkeep",
        "content": "",
    },
]
