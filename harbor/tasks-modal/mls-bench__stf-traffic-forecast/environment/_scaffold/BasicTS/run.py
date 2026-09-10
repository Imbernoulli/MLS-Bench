"""Unified CLI runner for stf-traffic-forecast task.

Usage:  python run.py
Reads ENV and SEED from environment variables.
"""

import os
import sys

sys.path.insert(0, "/workspace")

from torch.optim.lr_scheduler import MultiStepLR

from custom_model import Custom, CustomConfig, CONFIG_OVERRIDES
from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.metrics import masked_mae
from basicts.runners.callback import GradientClipping

_DATA_ROOT = os.environ.get("DATA_ROOT", "/data")

DS_INFO = {
    "METR-LA":  {"num_features": 207},
    "PEMS-BAY": {"num_features": 325},
    "PEMS04":   {"num_features": 307},
}

env = os.environ.get("ENV", "METR-LA")
seed = int(os.environ.get("SEED", 42))
ds = DS_INFO[env]

model_config = CustomConfig(
    input_len=12,
    output_len=12,
    num_features=ds["num_features"],
)

# Per-method override: baselines/agents can set training hyperparameters via CONFIG_OVERRIDES
# in custom_model.py. Allowed keys: lr, weight_decay.
lr = float(os.environ.get("LR", CONFIG_OVERRIDES.get("lr", 2e-3)))
weight_decay = float(os.environ.get("WD", CONFIG_OVERRIDES.get("weight_decay", 1e-4)))

cfg = BasicTSForecastingConfig(
    model=Custom,
    model_config=model_config,
    dataset_name=env,
    dataset_params={
        "input_len": 12, "output_len": 12,
        "use_timestamps": True,
        "data_file_path": f"{_DATA_ROOT}/datasets/{env}",
    },
    gpus="0", seed=seed, num_epochs=100, batch_size=64,
    metrics=["MAE", "RMSE", "MAPE"], target_metric="MAE",
    loss=masked_mae, null_val=0.0,
    norm_each_channel=False, rescale=True,
    optimizer_params={"lr": lr, "weight_decay": weight_decay},
    lr_scheduler=MultiStepLR,
    lr_scheduler_params={"milestones": [1, 50, 80], "gamma": 0.5},
    callbacks=[GradientClipping(5.0)],
    ckpt_save_dir=f"checkpoints/Custom/{env}_s{seed}",
)

BasicTSLauncher.launch_training(cfg)
