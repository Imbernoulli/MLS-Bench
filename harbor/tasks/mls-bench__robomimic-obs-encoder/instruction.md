# MLS-Bench: robomimic-obs-encoder

# Behavioral Cloning: Observation Encoder Design for Robot State Fusion

## Research Question
Design an improved observation encoder that fuses multiple robot state modalities for behavioral cloning. In robot manipulation, observations consist of heterogeneous components (end-effector pose, gripper state, object state) that may benefit from non-trivial fusion strategies beyond simple concatenation.

## Background
The training and evaluation pipeline follows **robomimic** (Mandlekar et al., CoRL 2021, arXiv:2108.03298). The default low-dimensional encoder concatenates per-modality vectors before feeding the result into the policy MLP. Concatenation discards modality-specific structure (rotation lives on SO(3), gripper is binary-like, object state may be high-dimensional and partly redundant), so a more structured encoder — per-modality projections, gating, attention, residual fusion — could help downstream BC.

## What You Can Modify
The `CustomObsEncoder` class in `custom_obs_encoder.py`. This module receives a dictionary of observation tensors and must return a fused feature vector.

Interface:
- **Input**: `obs_dict` — dictionary with keys:
  - `robot0_eef_pos`: [B, 3] end-effector position
  - `robot0_eef_quat`: [B, 4] end-effector quaternion orientation
  - `robot0_gripper_qpos`: [B, 2] gripper joint positions
  - `object`: [B, D_obj] object state
- **Output**: [B, output_dim] fused feature vector
- **Required attribute**: `self.output_dim` (int) — dimensionality of the output

You may add parameters to `__init__`, define helper methods, and add learnable layers.

## Evaluation
- **Metric**: `success_rate` — rollout success rate in the environment (higher is better)
- **Tasks**: Lift, Can, Square (robot manipulation with proficient human demonstrations)
- **Dataset**: ~200 proficient human demonstrations per task, low-dimensional observations
- **Policy**: GMM (Gaussian Mixture Model) with 5 mixture components, trained with NLL loss. A 2-layer MLP backbone (1024, 1024) with ReLU feeds into GMM heads (means, log-stds, mixture logits) on top of encoder output
- **Training**: 2000 epochs, Adam optimizer (lr = 1e-4), batch size 100
- **Rollout**: 50 episodes per task, horizon 400 steps, every 50 epochs


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/robomimic/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `robomimic/custom_obs_encoder.py`
- editable lines **19–46**




## Readable Context


### `robomimic/custom_obs_encoder.py`  [EDITABLE — lines 19–46 only]

```python
     1: """
     2: Custom Observation Encoder for multi-modal robot state fusion.
     3: 
     4: This module defines the observation encoder used by BC-GMM training
     5: in robomimic. The encoder receives a dictionary of observation tensors
     6: (end-effector pose, gripper state, object state) and returns a fused
     7: feature vector that is fed into the MLP backbone and GMM heads.
     8: 
     9: The custom encoder is imported and used by the patched BC_GMM network.
    10: """
    11: 
    12: import torch
    13: import torch.nn as nn
    14: import torch.nn.functional as F
    15: 
    16: 
    17: # ── Custom Observation Encoder ─────────────────────────────────────────────
    18: # EDITABLE REGION START
    19: class CustomObsEncoder(nn.Module):
    20:     """Custom observation encoder for multi-modal robot state fusion.
    21: 
    22:     Fuses multiple observation modalities (end-effector position,
    23:     orientation, gripper state, object state) into a single feature
    24:     vector. The default implementation concatenates all observations.
    25: 
    26:     Observation groups:
    27:         - robot0_eef_pos: [B, 3] end-effector position
    28:         - robot0_eef_quat: [B, 4] end-effector quaternion orientation
    29:         - robot0_gripper_qpos: [B, 2] gripper joint positions
    30:         - object: [B, D_obj] object state (position, orientation, etc.)
    31: 
    32:     Args:
    33:         obs_dims: dict mapping obs key names to their dimensions
    34: 
    35:     Returns:
    36:         [B, output_dim] fused feature vector
    37:     """
    38: 
    39:     def __init__(self, obs_dims):
    40:         super().__init__()
    41:         self.obs_dims = obs_dims
    42:         self.output_dim = sum(obs_dims.values())
    43: 
    44:     def forward(self, obs_dict):
    45:         parts = [obs_dict[k] for k in sorted(self.obs_dims.keys())]
    46:         return torch.cat(parts, dim=-1)
    47: # EDITABLE REGION END
```


## Adapter Warnings

Some reference context could not be rendered completely:

- `default` has no edit_ops entry


## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **tool_hang_ph** — wall-clock budget `8:00:00`, compute share `0.33`
- **can_ph** — wall-clock budget `8:00:00`, compute share `0.33`
- **square_ph** — wall-clock budget `8:00:00`, compute share `0.33`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `attention_fusion` baseline — editable region  [READ-ONLY — reference implementation]

In `robomimic/custom_obs_encoder.py`:

```python
Lines 19–51:
    16: 
    17: # ── Custom Observation Encoder ─────────────────────────────────────────────
    18: # EDITABLE REGION START
    19: class CustomObsEncoder(nn.Module):
    20:     """Attention-based cross-modality fusion encoder.
    21: 
    22:     Projects each modality to a shared embedding space, then applies
    23:     multi-head self-attention across modalities. The attended features
    24:     are concatenated for the final representation.
    25:     """
    26: 
    27:     def __init__(self, obs_dims, embed_dim=64, num_heads=2):
    28:         super().__init__()
    29:         self.obs_dims = obs_dims
    30:         self.embed_dim = embed_dim
    31:         self.projections = nn.ModuleDict()
    32:         for key in sorted(obs_dims.keys()):
    33:             d = obs_dims[key]
    34:             self.projections[key] = nn.Sequential(
    35:                 nn.Linear(d, embed_dim),
    36:                 nn.ReLU(),
    37:             )
    38:         self.attn = nn.MultiheadAttention(
    39:             embed_dim=embed_dim, num_heads=num_heads, batch_first=True
    40:         )
    41:         self.norm = nn.LayerNorm(embed_dim)
    42:         self.output_dim = embed_dim * len(obs_dims)
    43: 
    44:     def forward(self, obs_dict):
    45:         tokens = []
    46:         for key in sorted(self.obs_dims.keys()):
    47:             tokens.append(self.projections[key](obs_dict[key]))
    48:         tokens = torch.stack(tokens, dim=1)
    49:         attn_out, _ = self.attn(tokens, tokens, tokens)
    50:         tokens = self.norm(tokens + attn_out)
    51:         return tokens.reshape(tokens.shape[0], -1)
    52: # EDITABLE REGION END
```

### `gated_fusion` baseline — editable region  [READ-ONLY — reference implementation]

In `robomimic/custom_obs_encoder.py`:

```python
Lines 19–53:
    16: 
    17: # ── Custom Observation Encoder ─────────────────────────────────────────────
    18: # EDITABLE REGION START
    19: class CustomObsEncoder(nn.Module):
    20:     """Gated fusion encoder: sigmoid gates weight each modality.
    21: 
    22:     Each modality is processed by a small MLP, then a learned gate
    23:     (sigmoid) determines its contribution. The gated features are
    24:     concatenated for the final representation.
    25:     """
    26: 
    27:     def __init__(self, obs_dims, embed_dim=64):
    28:         super().__init__()
    29:         self.obs_dims = obs_dims
    30:         self.embed_dim = embed_dim
    31:         self.encoders = nn.ModuleDict()
    32:         self.gates = nn.ModuleDict()
    33:         for key in sorted(obs_dims.keys()):
    34:             d = obs_dims[key]
    35:             self.encoders[key] = nn.Sequential(
    36:                 nn.Linear(d, embed_dim),
    37:                 nn.ReLU(),
    38:                 nn.Linear(embed_dim, embed_dim),
    39:                 nn.ReLU(),
    40:             )
    41:             self.gates[key] = nn.Sequential(
    42:                 nn.Linear(d, embed_dim),
    43:                 nn.Sigmoid(),
    44:             )
    45:         self.output_dim = embed_dim * len(obs_dims)
    46: 
    47:     def forward(self, obs_dict):
    48:         parts = []
    49:         for key in sorted(self.obs_dims.keys()):
    50:             feat = self.encoders[key](obs_dict[key])
    51:             gate = self.gates[key](obs_dict[key])
    52:             parts.append(feat * gate)
    53:         return torch.cat(parts, dim=-1)
    54: # EDITABLE REGION END
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
