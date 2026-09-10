"""Mid-edit operations for the robomimic-bc-loss task.

Applied to the robomimic workspace after pre_edit, before the agent starts.

1. Creates custom_bc_loss.py — the editable loss module.
2. Creates bc_gmm_config.json — BC-GMM training config for robomimic.
3. Patches robomimic/algo/bc.py to import and use CustomBCLoss in BC_GMM.
"""

import json
from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── BC-GMM config JSON ─────────────────────────────────────────────────
# Based on robomimic/exps/templates/bc.json with gmm.enabled = true
_BC_GMM_CONFIG = {
    "algo_name": "bc",
    "experiment": {
        "name": "bc_gmm_custom_loss",
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
        "output_dir": "/tmp/bc_trained_models",
        "normalize_weights_by_ds_size": False,
        "num_data_workers": 0,
        "hdf5_cache_mode": "all",
        "hdf5_use_swmr": True,
        "hdf5_load_next_obs": False,
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
            "policy": {
                "optimizer_type": "adam",
                "learning_rate": {
                    "initial": 0.0001,
                    "decay_factor": 0.1,
                    "epoch_schedule": [],
                    "scheduler_type": "multistep",
                },
                "regularization": {"L2": 0.0},
            }
        },
        "loss": {"l2_weight": 1.0, "l1_weight": 0.0, "cos_weight": 0.0},
        "actor_layer_dims": [1024, 1024],
        "gaussian": {
            "enabled": False,
            "fixed_std": False,
            "init_std": 0.1,
            "min_std": 0.01,
            "std_activation": "softplus",
            "low_noise_eval": True,
        },
        "gmm": {
            "enabled": True,
            "num_modes": 5,
            "min_std": 0.0001,
            "std_activation": "softplus",
            "low_noise_eval": True,
        },
        "vae": {
            "enabled": False,
            "latent_dim": 14,
            "latent_clip": None,
            "kl_weight": 1.0,
            "decoder": {
                "is_conditioned": True,
                "reconstruction_sum_across_elements": False,
            },
            "prior": {
                "learn": False,
                "is_conditioned": False,
                "use_gmm": False,
                "gmm_num_modes": 10,
                "gmm_learn_weights": False,
                "use_categorical": False,
                "categorical_dim": 10,
                "categorical_gumbel_softmax_hard": False,
                "categorical_init_temp": 1.0,
                "categorical_temp_anneal_step": 0.001,
                "categorical_min_temp": 0.3,
            },
            "encoder_layer_dims": [300, 400],
            "decoder_layer_dims": [300, 400],
            "prior_layer_dims": [300, 400],
        },
        "rnn": {
            "enabled": False,
            "horizon": 10,
            "hidden_dim": 400,
            "rnn_type": "LSTM",
            "num_layers": 2,
            "open_loop": False,
            "kwargs": {"bidirectional": False},
        },
        "transformer": {
            "enabled": False,
            "context_length": 10,
            "embed_dim": 512,
            "num_layers": 6,
            "num_heads": 8,
            "emb_dropout": 0.1,
            "attn_dropout": 0.1,
            "block_output_dropout": 0.1,
            "sinusoidal_embedding": False,
            "activation": "gelu",
            "supervise_all_steps": False,
            "nn_parameter_for_timesteps": True,
            "pred_future_acs": False,
        },
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

_BC_GMM_CONFIG_JSON = json.dumps(_BC_GMM_CONFIG, indent=2)

# ── Patch for robomimic/algo/bc.py ─────────────────────────────────────
# Insert after BC_GMM._create_networks (line 370) to add _forward_training
# and _compute_losses overrides that use CustomBCLoss.
_BC_GMM_PATCH = '''
    def _forward_training(self, batch):
        """Override to store the GMM distribution for CustomBCLoss."""
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
        )
        assert len(dists.batch_shape) == 1
        # Store dist on self (not in predictions) to avoid TensorUtils.detach failure
        self._last_dists = dists
        predictions = OrderedDict(
            log_probs=dists.log_prob(batch["actions"]),
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """Override to use CustomBCLoss instead of standard NLL."""
        from custom_bc_loss import CustomBCLoss
        if not hasattr(self, "_custom_loss_fn"):
            self._custom_loss_fn = CustomBCLoss(action_dim=self.ac_dim).to(self.device)
        action_loss = self._custom_loss_fn(self._last_dists, batch["actions"])
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )
'''

# ── Mid-edit operations ──────────────────────────────────────────────
OPS = [
    # 1. Create the editable custom loss module
    {
        "op": "create",
        "file": "robomimic/custom_bc_loss.py",
        "content": _CUSTOM_PY,
    },
    # 2. Create BC-GMM config JSON
    {
        "op": "create",
        "file": "robomimic/bc_gmm_config.json",
        "content": _BC_GMM_CONFIG_JSON,
    },
    # 3. Patch BC_GMM in algo/bc.py to use CustomBCLoss
    #    Insert after line 370 (end of _create_networks method body)
    {
        "op": "insert",
        "file": "robomimic/robomimic/algo/bc.py",
        "after_line": 370,
        "content": _BC_GMM_PATCH,
    },
]
