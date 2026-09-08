import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# ── Custom imports (editable) ────────────────────────────────────────────


# ======================================================================
# EDITABLE — Custom centralized critic for MAPPO
# ======================================================================
class CustomCritic(nn.Module):
    """Centralized critic for MAPPO on SMAC (via smaclite).

    Plugged into epymarl's ppo_learner via ``critic_type: "custom_critic"``
    in ``custom_mappo.yaml``. The learner calls ``critic(batch)`` without
    the ``t`` argument and later does ``.squeeze(3)``, so the output MUST
    have shape ``(batch, T, n_agents, 1)``.

    Args:
        scheme: dict with keys
            ``"state"["vshape"]`` (int) — global state dim
            ``"obs"["vshape"]``   (int) — per-agent obs dim
            ``"actions_onehot"["vshape"]`` (tuple) — action one-hot dim
        args: Namespace with attributes
            ``n_agents``, ``n_actions``, ``hidden_dim``,
            ``obs_agent_id``, ``obs_last_action``, ``obs_individual_obs``

    Interface:
        forward(batch, t=None) -> q
            batch : components.episode_buffer.EpisodeBatch
                batch["state"] : (B, T, state_dim)
                batch["obs"]   : (B, T, n_agents, obs_dim)
                batch.batch_size, batch.max_seq_length, batch.device
            t     : int or None  (None = whole sequence)
            q     : (B, T, n_agents, 1)     ← REQUIRED shape
        self.output_type = "v"              ← REQUIRED attribute
    """

    def __init__(self, scheme, args):
        super(CustomCritic, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.output_type = "v"

        # Default: simple state + agent-id MLP (central V baseline).
        self.state_dim = int(scheme["state"]["vshape"])
        input_shape = self.state_dim + self.n_agents
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        state = batch["state"][:, ts]                                    # (B, T, state_dim)
        state = state.unsqueeze(2).expand(-1, -1, self.n_agents, -1)     # (B, T, n, state_dim)
        agent_id = th.eye(self.n_agents, device=batch.device)
        agent_id = agent_id.unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
        inputs = th.cat([state, agent_id], dim=-1)                       # (B, T, n, state+n)

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)                                                  # (B, T, n, 1)
        return q
