"""MAPPO critic baseline — rigorous codebase edit ops.

Standard centralized V from Yu et al. 2022
"The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
(arXiv 2103.01955): a shared MLP over the global state concatenated
with an agent one-hot index (the "AS" / agent-specific form in the
paper's terminology). Matches ``CentralVCritic`` in
``epymarl/src/modules/critics/centralV.py``.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "epymarl/src/modules/critics/custom_critic.py"

# ── Replace CustomCritic class (lines 13-69) ──────────────────────────

_MAPPO_CLASS = """\
class CustomCritic(nn.Module):
    \"\"\"MAPPO critic — shared MLP over (state + agent one-hot).

    Standard centralized V from Yu et al. 2022 (arXiv 2103.01955).
    Matches epymarl's CentralVCritic. All agents share the same network;
    the agent one-hot lets the shared network produce agent-specific
    value estimates while still conditioning on the full global state.
    \"\"\"

    def __init__(self, scheme, args):
        super(CustomCritic, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.output_type = "v"

        state_dim = int(scheme["state"]["vshape"])
        input_shape = state_dim + self.n_agents
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
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 13,
        "end_line": 69,
        "content": _MAPPO_CLASS,
    },
]
