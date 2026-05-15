"""Mid-edit operations for the robomimic-obs-encoder task.

Applied to the robomimic workspace after pre_edit, before the agent starts.

1. Creates custom_obs_encoder.py — the editable encoder module.
2. Creates bc_gmm_config.json — BC-GMM training config for robomimic.
3. Patches robomimic/algo/bc.py to replace BC_GMM with a version that uses
   CustomObsEncoder as the observation encoder.
"""

import json
from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── BC-GMM config JSON ─────────────────────────────────────────────────
# Same as bc-loss task config (BC with GMM enabled)
_BC_GMM_CONFIG = {
    "algo_name": "bc",
    "experiment": {
        "name": "bc_gmm_custom_encoder",
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
# Append to the END of bc.py (after line 897) to add a custom GMM policy
# class that uses CustomObsEncoder, and monkey-patch BC_GMM to use it.
_BC_GMM_CUSTOM_ENCODER_PATCH = '''

# ── MLS-Bench: Custom Observation Encoder integration ────────────────────
class _CustomEncoderGMMPolicy(nn.Module):
    """GMM policy using CustomObsEncoder for observation fusion."""

    def __init__(self, obs_shapes, ac_dim, mlp_layer_dims, num_modes,
                 min_std, std_activation, low_noise_eval):
        super().__init__()
        from custom_obs_encoder import CustomObsEncoder

        obs_dims = {k: v[0] for k, v in obs_shapes.items()}
        self.encoder = CustomObsEncoder(obs_dims)
        enc_dim = self.encoder.output_dim

        # MLP backbone
        layers = []
        d = enc_dim
        for h in mlp_layer_dims:
            layers.extend([nn.Linear(d, h), nn.ReLU()])
            d = h
        self.mlp = nn.Sequential(*layers)

        # GMM heads
        self.mean_head = nn.Linear(d, num_modes * ac_dim)
        self.scale_head = nn.Linear(d, num_modes * ac_dim)
        self.logits_head = nn.Linear(d, num_modes)

        self.num_modes = num_modes
        self.ac_dim = ac_dim
        self.min_std = min_std
        self.std_activation = std_activation
        self.low_noise_eval = low_noise_eval
        self._activations = {"softplus": F.softplus, "exp": torch.exp}

    def forward_train(self, obs_dict, goal_dict=None):
        """Return GMM distribution for training."""
        features = self.encoder(obs_dict)
        h = self.mlp(features)
        B = h.shape[0]
        means = torch.tanh(
            self.mean_head(h).reshape(B, self.num_modes, self.ac_dim))
        scales = self.scale_head(h).reshape(B, self.num_modes, self.ac_dim)
        logits = self.logits_head(h)

        if self.low_noise_eval and not self.training:
            scales = torch.ones_like(means) * 1e-4
        else:
            scales = self._activations[self.std_activation](scales) + self.min_std

        comp = D.Independent(D.Normal(means, scales), 1)
        mix = D.Categorical(logits=logits)
        return D.MixtureSameFamily(mix, comp)

    def forward(self, obs_dict=None, goal_dict=None, **kwargs):
        """Sample actions."""
        dist = self.forward_train(obs_dict, goal_dict)
        if self.low_noise_eval and not self.training:
            # Return approximate mode
            return dist.sample()
        return dist.sample()


# Monkey-patch BC_GMM to use custom encoder
_orig_BC_GMM_create_networks = BC_GMM._create_networks

def _patched_BC_GMM_create_networks(self):
    """Replacement _create_networks that uses CustomObsEncoder."""
    assert self.algo_config.gmm.enabled
    self.nets = nn.ModuleDict()
    self.nets["policy"] = _CustomEncoderGMMPolicy(
        obs_shapes=self.obs_shapes,
        ac_dim=self.ac_dim,
        mlp_layer_dims=self.algo_config.actor_layer_dims,
        num_modes=self.algo_config.gmm.num_modes,
        min_std=self.algo_config.gmm.min_std,
        std_activation=self.algo_config.gmm.std_activation,
        low_noise_eval=self.algo_config.gmm.low_noise_eval,
    )
    self.nets = self.nets.float().to(self.device)

BC_GMM._create_networks = _patched_BC_GMM_create_networks
'''


# ── Mid-edit operations ──────────────────────────────────────────────
OPS = [
    # 1. Create the editable custom encoder module
    {
        "op": "create",
        "file": "robomimic/custom_obs_encoder.py",
        "content": _CUSTOM_PY,
    },
    # 2. Create BC-GMM config JSON
    {
        "op": "create",
        "file": "robomimic/bc_gmm_config.json",
        "content": _BC_GMM_CONFIG_JSON,
    },
    # 3. Append custom encoder integration to algo/bc.py (after line 897)
    {
        "op": "insert",
        "file": "robomimic/robomimic/algo/bc.py",
        "after_line": 897,
        "content": _BC_GMM_CUSTOM_ENCODER_PATCH,
    },
]
