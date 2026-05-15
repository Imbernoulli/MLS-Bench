"""Mid-edit operations for the robomimic-iql-vf task.

Applied to the robomimic workspace after pre_edit, before the agent starts.

1. Creates custom_iql_vf.py — the editable value function loss module.
2. Creates iql_gmm_config.json — IQL training config with GMM actor.
3. Patches robomimic/algo/iql.py to use custom_vf_loss in _compute_critic_loss.
"""

import json
from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── IQL config JSON ────────────────────────────────────────────────────
# Based on robomimic/exps/templates/iql.json with GMM actor
_IQL_GMM_CONFIG = {
    "algo_name": "iql",
    "experiment": {
        "name": "iql_custom_vf",
        "validate": True,
        "logging": {
            "terminal_output_to_txt": False,
            "log_tb": False,
            "log_wandb": False,
            "wandb_proj_name": "debug",
        },
        "save": {
            "enabled": True,
            "every_n_seconds": None,
            "every_n_epochs": 50,
            "epochs": [],
            "on_best_validation": False,
            "on_best_rollout_return": False,
            "on_best_rollout_success_rate": True,
        },
        "epoch_every_n_steps": 100,
        "validation_epoch_every_n_steps": 10,
        "env": None,
        "additional_envs": None,
        "render": False,
        "render_video": False,
        "keep_all_videos": False,
        "video_skip": 5,
        "rollout": {
            "enabled": True,
            "n": 50,
            "horizon": 400,
            "rate": 50,
            "warmstart": 0,
            "terminate_on_success": True,
        },
        "env_meta_update_dict": {},
        "ckpt_path": None,
    },
    "train": {
        "data": None,
        "output_dir": "/tmp/iql_trained_models",
        "normalize_weights_by_ds_size": False,
        "num_data_workers": 0,
        "hdf5_cache_mode": "all",
        "hdf5_use_swmr": True,
        "hdf5_load_next_obs": True,
        "hdf5_normalize_obs": False,
        "hdf5_filter_key": "train",
        "hdf5_validation_filter_key": "valid",
        "seq_length": 1,
        "pad_seq_length": True,
        "frame_stack": 1,
        "pad_frame_stack": True,
        "dataset_keys": ["actions", "rewards", "dones"],
        "action_keys": ["actions"],
        "action_config": {"actions": {"normalization": None}},
        "goal_mode": None,
        "cuda": True,
        "batch_size": 100,
        "num_epochs": 2000,
        "seed": 1,
        "max_grad_norm": None,
    },
    "algo": {
        "optim_params": {
            "critic": {
                "learning_rate": {
                    "initial": 0.0001,
                    "decay_factor": 0.0,
                    "epoch_schedule": [],
                },
                "regularization": {"L2": 0.0},
            },
            "vf": {
                "learning_rate": {
                    "initial": 0.0001,
                    "decay_factor": 0.0,
                    "epoch_schedule": [],
                },
                "regularization": {"L2": 0.0},
            },
            "actor": {
                "learning_rate": {
                    "initial": 0.0001,
                    "decay_factor": 0.0,
                    "epoch_schedule": [],
                },
                "regularization": {"L2": 0.0},
            },
        },
        "discount": 0.99,
        "target_tau": 0.01,
        "actor": {
            "net": {
                "type": "gmm",
                "common": {
                    "std_activation": "softplus",
                    "low_noise_eval": True,
                    "use_tanh": False,
                },
                "gaussian": {
                    "init_last_fc_weight": 0.001,
                    "init_std": 0.3,
                    "fixed_std": False,
                },
                "gmm": {
                    "num_modes": 5,
                    "min_std": 0.0001,
                },
            },
            "layer_dims": [300, 400],
            "max_gradient_norm": None,
        },
        "critic": {
            "ensemble": {"n": 2},
            "layer_dims": [300, 400],
            "use_huber": False,
            "max_gradient_norm": None,
        },
        "adv": {
            "clip_adv_value": None,
            "beta": 1.0,
            "use_final_clip": True,
        },
        "vf_quantile": 0.9,
    },
    "observation": {
        "modalities": {
            "obs": {
                "low_dim": [
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                    "object",
                ],
                "rgb": [],
                "depth": [],
                "scan": [],
            },
            "goal": {"low_dim": [], "rgb": [], "depth": [], "scan": []},
        },
        "encoder": {
            "low_dim": {
                "core_class": None,
                "core_kwargs": {},
                "obs_randomizer_class": None,
                "obs_randomizer_kwargs": {},
            },
            "rgb": {
                "core_class": "VisualCore",
                "core_kwargs": {},
                "obs_randomizer_class": None,
                "obs_randomizer_kwargs": {},
            },
            "depth": {
                "core_class": "VisualCore",
                "core_kwargs": {},
                "obs_randomizer_class": None,
                "obs_randomizer_kwargs": {},
            },
            "scan": {
                "core_class": "ScanCore",
                "core_kwargs": {},
                "obs_randomizer_class": None,
                "obs_randomizer_kwargs": {},
            },
        },
    },
    "meta": {"hp_base_config_file": None, "hp_keys": [], "hp_values": []},
}

_IQL_GMM_CONFIG_JSON = json.dumps(_IQL_GMM_CONFIG, indent=2)


# ── Patch for robomimic/algo/iql.py ────────────────────────────────────
# Replace lines 224-228 (the expectile regression VF loss) with a call
# to custom_vf_loss from custom_iql_vf.py.
_IQL_VF_PATCH = """\
        # V losses: custom VF loss (MLS-Bench)
        from custom_iql_vf import custom_vf_loss
        vf_loss = custom_vf_loss(vf_pred, q_pred, self.algo_config.vf_quantile)
"""


# ── Mid-edit operations ──────────────────────────────────────────────
OPS = [
    # 1. Create the editable custom VF loss module
    {
        "op": "create",
        "file": "robomimic/custom_iql_vf.py",
        "content": _CUSTOM_PY,
    },
    # 2. Create IQL config JSON (with GMM actor)
    {
        "op": "create",
        "file": "robomimic/iql_gmm_config.json",
        "content": _IQL_GMM_CONFIG_JSON,
    },
    # 3. Replace expectile regression with custom_vf_loss in iql.py
    #    (lines 224-228)
    {
        "op": "replace",
        "file": "robomimic/robomimic/algo/iql.py",
        "start_line": 224,
        "end_line": 228,
        "content": _IQL_VF_PATCH,
    },
]
