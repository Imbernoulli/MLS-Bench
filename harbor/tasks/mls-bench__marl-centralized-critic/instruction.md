# MLS-Bench: marl-centralized-critic

# Cooperative MARL: Centralized Critic Architecture for MAPPO

## Research Question
Improve cooperative multi-agent reinforcement learning by designing a better
**centralized critic architecture** for MAPPO (Multi-Agent PPO). You will
modify the `CustomCritic` class and may add custom imports inside the
editable region of `custom_critic.py`.

## Background
In cooperative MARL with partial observability, each agent only sees a
local observation but the team shares a common reward. Centralized-Training-
with-Decentralized-Execution (CTDE) methods train a centralized value
function during training (which can see the global state and possibly all
agents' information) and use it to reduce variance when computing
advantages for each agent's decentralized policy gradient update. The
architecture of this centralized critic — what it conditions on and how it
mixes per-agent features — directly determines the bias-variance tradeoff
and how well MAPPO scales to hard cooperation tasks.

The training uses EPyMARL's `ppo_learner` with the MAPPO default
hyperparameters from Yu et al. (2022) on cooperative SMAC maps via
**smaclite**, a pure-Python reimplementation of the StarCraft Multi-Agent
Challenge benchmark that does not require the StarCraft II binary. Each map
trains for roughly 5M environment steps. The actor architecture, learner,
optimizer, GAE settings, and environment interface are fixed.

## Interface
Your `CustomCritic` must:
- Inherit from `nn.Module`.
- Accept `(scheme, args)` in `__init__`, where:
  - `scheme["state"]["vshape"]` — global state dim
  - `scheme["obs"]["vshape"]` — per-agent observation dim
  - `args.n_agents`, `args.n_actions`, `args.hidden_dim`,
    `args.obs_last_action`, `args.obs_individual_obs`
- Set `self.output_type = "v"` in `__init__`.
- Implement `forward(self, batch, t=None)` where:
  - `batch["state"]` has shape `(B, T, state_dim)`
  - `batch["obs"]` has shape `(B, T, n_agents, obs_dim)`
  - `batch.batch_size`, `batch.max_seq_length`, `batch.device` are
    available
  - `t=None` means "whole sequence"; otherwise `t` is an integer time
    index
  - Returns `q` with shape `(B, T, n_agents, 1)`. The learner later does
    `.squeeze(3)`, so the trailing singleton is mandatory.

## Reference Implementations
The following baselines are provided as `*.edit.py` files for reference and
serve as design points spanning the literature:

- **IPPO critic** — per-agent MLP over `batch["obs"]` ⊕ agent-one-hot, no
  centralization. Floor baseline corresponding to the IPPO ablation in Yu
  et al., "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent
  Games" (arXiv:2103.01955, NeurIPS Datasets and Benchmarks 2022). See
  also `epymarl/src/modules/critics/ac.py`.
- **MAPPO critic** — shared MLP over `(batch["state"] ⊕ agent-one-hot)`.
  The standard MAPPO central V from the same paper. See also
  `epymarl/src/modules/critics/centralV.py`.
- **MAT-style attention critic** — projects per-agent features
  `(obs ⊕ broadcast state)` into tokens, then a single
  `TransformerEncoder` layer with self-attention across the agent axis
  produces a per-agent value. Adapted (critic-only form) from Wen et al.,
  "Multi-Agent Reinforcement Learning is a Sequence Modeling Problem"
  (arXiv:2205.14953, NeurIPS 2022); the MAPPO actor is kept unchanged.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/epymarl/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `epymarl/src/modules/critics/custom_critic.py`
- editable lines **7–8**
- editable lines **13–69**


Other files you may **read** for context (do not modify):
- `epymarl/src/modules/critics/centralV.py`
- `epymarl/src/modules/critics/ac.py`
- `epymarl/src/learners/ppo_learner.py`


## Readable Context


### `epymarl/src/modules/critics/custom_critic.py`  [EDITABLE — lines 7–8, lines 13–69 only]

```python
     1: import numpy as np
     2: import torch as th
     3: import torch.nn as nn
     4: import torch.nn.functional as F
     5: 
     6: 
     7: # ── Custom imports (editable) ────────────────────────────────────────────
     8: 
     9: 
    10: # ======================================================================
    11: # EDITABLE — Custom centralized critic for MAPPO
    12: # ======================================================================
    13: class CustomCritic(nn.Module):
    14:     """Centralized critic for MAPPO on SMAC (via smaclite).
    15: 
    16:     Plugged into epymarl's ppo_learner via ``critic_type: "custom_critic"``
    17:     in ``custom_mappo.yaml``. The learner calls ``critic(batch)`` without
    18:     the ``t`` argument and later does ``.squeeze(3)``, so the output MUST
    19:     have shape ``(batch, T, n_agents, 1)``.
    20: 
    21:     Args:
    22:         scheme: dict with keys
    23:             ``"state"["vshape"]`` (int) — global state dim
    24:             ``"obs"["vshape"]``   (int) — per-agent obs dim
    25:             ``"actions_onehot"["vshape"]`` (tuple) — action one-hot dim
    26:         args: Namespace with attributes
    27:             ``n_agents``, ``n_actions``, ``hidden_dim``,
    28:             ``obs_agent_id``, ``obs_last_action``, ``obs_individual_obs``
    29: 
    30:     Interface:
    31:         forward(batch, t=None) -> q
    32:             batch : components.episode_buffer.EpisodeBatch
    33:                 batch["state"] : (B, T, state_dim)
    34:                 batch["obs"]   : (B, T, n_agents, obs_dim)
    35:                 batch.batch_size, batch.max_seq_length, batch.device
    36:             t     : int or None  (None = whole sequence)
    37:             q     : (B, T, n_agents, 1)     ← REQUIRED shape
    38:         self.output_type = "v"              ← REQUIRED attribute
    39:     """
    40: 
    41:     def __init__(self, scheme, args):
    42:         super(CustomCritic, self).__init__()
    43:         self.args = args
    44:         self.n_agents = args.n_agents
    45:         self.n_actions = args.n_actions
    46:         self.output_type = "v"
    47: 
    48:         # Default: simple state + agent-id MLP (central V baseline).
    49:         self.state_dim = int(scheme["state"]["vshape"])
    50:         input_shape = self.state_dim + self.n_agents
    51:         self.fc1 = nn.Linear(input_shape, args.hidden_dim)
    52:         self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
    53:         self.fc3 = nn.Linear(args.hidden_dim, 1)
    54: 
    55:     def forward(self, batch, t=None):
    56:         bs = batch.batch_size
    57:         max_t = batch.max_seq_length if t is None else 1
    58:         ts = slice(None) if t is None else slice(t, t + 1)
    59: 
    60:         state = batch["state"][:, ts]                                    # (B, T, state_dim)
    61:         state = state.unsqueeze(2).expand(-1, -1, self.n_agents, -1)     # (B, T, n, state_dim)
    62:         agent_id = th.eye(self.n_agents, device=batch.device)
    63:         agent_id = agent_id.unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
    64:         inputs = th.cat([state, agent_id], dim=-1)                       # (B, T, n, state+n)
    65: 
    66:         x = F.relu(self.fc1(inputs))
    67:         x = F.relu(self.fc2(x))
    68:         q = self.fc3(x)                                                  # (B, T, n, 1)
    69:         return q
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `ippo_critic` baseline — editable region  [READ-ONLY — reference implementation]

In `epymarl/src/modules/critics/custom_critic.py`:

```python
Lines 7–8:
     4: import torch.nn.functional as F
     5: 
     6: 
     7: # ── Custom imports (editable) ────────────────────────────────────────────
     8: 
     9: 
    10: # ======================================================================
    11: # EDITABLE — Custom centralized critic for MAPPO

Lines 13–47:
    10: # ======================================================================
    11: # EDITABLE — Custom centralized critic for MAPPO
    12: # ======================================================================
    13: class CustomCritic(nn.Module):
    14:     """IPPO critic — per-agent MLP over local obs + agent one-hot.
    15: 
    16:     Matches epymarl's ACCritic. No centralization: each agent's value
    17:     depends only on its own observation. Serves as the "no centralization"
    18:     floor baseline from Yu et al. 2022 (arXiv 2103.01955).
    19:     """
    20: 
    21:     def __init__(self, scheme, args):
    22:         super(CustomCritic, self).__init__()
    23:         self.args = args
    24:         self.n_agents = args.n_agents
    25:         self.n_actions = args.n_actions
    26:         self.output_type = "v"
    27: 
    28:         obs_dim = int(scheme["obs"]["vshape"])
    29:         input_shape = obs_dim + self.n_agents   # obs + agent-one-hot
    30:         self.fc1 = nn.Linear(input_shape, args.hidden_dim)
    31:         self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
    32:         self.fc3 = nn.Linear(args.hidden_dim, 1)
    33: 
    34:     def forward(self, batch, t=None):
    35:         bs = batch.batch_size
    36:         max_t = batch.max_seq_length if t is None else 1
    37:         ts = slice(None) if t is None else slice(t, t + 1)
    38: 
    39:         obs = batch["obs"][:, ts]                                        # (B, T, n, obs_dim)
    40:         agent_id = th.eye(self.n_agents, device=batch.device)
    41:         agent_id = agent_id.unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
    42:         inputs = th.cat([obs, agent_id], dim=-1)                         # (B, T, n, obs+n)
    43: 
    44:         x = F.relu(self.fc1(inputs))
    45:         x = F.relu(self.fc2(x))
    46:         q = self.fc3(x)                                                  # (B, T, n, 1)
    47:         return q
```

### `mappo_critic` baseline — editable region  [READ-ONLY — reference implementation]

In `epymarl/src/modules/critics/custom_critic.py`:

```python
Lines 7–8:
     4: import torch.nn.functional as F
     5: 
     6: 
     7: # ── Custom imports (editable) ────────────────────────────────────────────
     8: 
     9: 
    10: # ======================================================================
    11: # EDITABLE — Custom centralized critic for MAPPO

Lines 13–49:
    10: # ======================================================================
    11: # EDITABLE — Custom centralized critic for MAPPO
    12: # ======================================================================
    13: class CustomCritic(nn.Module):
    14:     """MAPPO critic — shared MLP over (state + agent one-hot).
    15: 
    16:     Standard centralized V from Yu et al. 2022 (arXiv 2103.01955).
    17:     Matches epymarl's CentralVCritic. All agents share the same network;
    18:     the agent one-hot lets the shared network produce agent-specific
    19:     value estimates while still conditioning on the full global state.
    20:     """
    21: 
    22:     def __init__(self, scheme, args):
    23:         super(CustomCritic, self).__init__()
    24:         self.args = args
    25:         self.n_agents = args.n_agents
    26:         self.n_actions = args.n_actions
    27:         self.output_type = "v"
    28: 
    29:         state_dim = int(scheme["state"]["vshape"])
    30:         input_shape = state_dim + self.n_agents
    31:         self.fc1 = nn.Linear(input_shape, args.hidden_dim)
    32:         self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
    33:         self.fc3 = nn.Linear(args.hidden_dim, 1)
    34: 
    35:     def forward(self, batch, t=None):
    36:         bs = batch.batch_size
    37:         max_t = batch.max_seq_length if t is None else 1
    38:         ts = slice(None) if t is None else slice(t, t + 1)
    39: 
    40:         state = batch["state"][:, ts]                                    # (B, T, state_dim)
    41:         state = state.unsqueeze(2).expand(-1, -1, self.n_agents, -1)     # (B, T, n, state_dim)
    42:         agent_id = th.eye(self.n_agents, device=batch.device)
    43:         agent_id = agent_id.unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
    44:         inputs = th.cat([state, agent_id], dim=-1)                       # (B, T, n, state+n)
    45: 
    46:         x = F.relu(self.fc1(inputs))
    47:         x = F.relu(self.fc2(x))
    48:         q = self.fc3(x)                                                  # (B, T, n, 1)
    49:         return q
```

### `mat_critic` baseline — editable region  [READ-ONLY — reference implementation]

In `epymarl/src/modules/critics/custom_critic.py`:

```python
Lines 7–8:
     4: import torch.nn.functional as F
     5: 
     6: 
     7: # ── Custom imports (editable) ────────────────────────────────────────────
     8: 
     9: 
    10: # ======================================================================
    11: # EDITABLE — Custom centralized critic for MAPPO

Lines 13–70:
    10: # ======================================================================
    11: # EDITABLE — Custom centralized critic for MAPPO
    12: # ======================================================================
    13: class CustomCritic(nn.Module):
    14:     """MAT-style attention critic — self-attention over per-agent tokens.
    15: 
    16:     Adapted from Wen et al. 2022 MAT (arXiv 2205.14953), critic-only form.
    17:     Each agent's token encodes its local observation together with the
    18:     global state; a single TransformerEncoder layer mixes information
    19:     across agents via self-attention, then a per-token linear head
    20:     produces the scalar value.
    21:     """
    22: 
    23:     def __init__(self, scheme, args):
    24:         super(CustomCritic, self).__init__()
    25:         self.args = args
    26:         self.n_agents = args.n_agents
    27:         self.n_actions = args.n_actions
    28:         self.output_type = "v"
    29: 
    30:         obs_dim = int(scheme["obs"]["vshape"])
    31:         state_dim = int(scheme["state"]["vshape"])
    32:         self.d_model = args.hidden_dim
    33: 
    34:         # Per-agent token projection: [obs_i ⊕ state] → d_model
    35:         self.token_proj = nn.Linear(obs_dim + state_dim, self.d_model)
    36: 
    37:         # Single transformer encoder layer with self-attention across agents
    38:         enc_layer = nn.TransformerEncoderLayer(
    39:             d_model=self.d_model,
    40:             nhead=4,
    41:             dim_feedforward=4 * self.d_model,
    42:             dropout=0.0,
    43:             batch_first=True,
    44:             activation="gelu",
    45:         )
    46:         self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
    47: 
    48:         # Per-agent value head
    49:         self.v_head = nn.Linear(self.d_model, 1)
    50: 
    51:     def forward(self, batch, t=None):
    52:         bs = batch.batch_size
    53:         max_t = batch.max_seq_length if t is None else 1
    54:         ts = slice(None) if t is None else slice(t, t + 1)
    55: 
    56:         obs = batch["obs"][:, ts]                                        # (B, T, n, obs_dim)
    57:         state = batch["state"][:, ts]                                    # (B, T, state_dim)
    58:         state = state.unsqueeze(2).expand(-1, -1, self.n_agents, -1)     # (B, T, n, state_dim)
    59:         tokens = th.cat([obs, state], dim=-1)                            # (B, T, n, obs+state)
    60:         tokens = self.token_proj(tokens)                                 # (B, T, n, d_model)
    61: 
    62:         # Flatten (B, T) into a single batch dim for the transformer,
    63:         # then restore: TransformerEncoder expects (bs*, seq_len, d_model).
    64:         b, tt, n, d = tokens.shape
    65:         tokens = tokens.reshape(b * tt, n, d)
    66:         attn_out = self.encoder(tokens)                                  # (B*T, n, d_model)
    67:         attn_out = attn_out.reshape(b, tt, n, d)
    68: 
    69:         q = self.v_head(attn_out)                                       # (B, T, n, 1)
    70:         return q
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
