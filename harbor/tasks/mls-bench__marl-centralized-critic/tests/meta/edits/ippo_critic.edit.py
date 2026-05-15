"""IPPO critic baseline — rigorous codebase edit ops.

Per-agent decentralized critic: each agent has its own value estimate
computed from its local observation only (no global state, no peer
information). This is the floor baseline from Yu et al. 2022
"The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
(arXiv 2103.01955), and matches the ``ACCritic`` class in
``epymarl/src/modules/critics/ac.py``.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "epymarl/src/modules/critics/custom_critic.py"

# ── Replace CustomCritic class (lines 13-69) ──────────────────────────

_IPPO_CLASS = """\
class CustomCritic(nn.Module):
    \"\"\"IPPO critic — per-agent MLP over local obs + agent one-hot.

    Matches epymarl's ACCritic. No centralization: each agent's value
    depends only on its own observation. Serves as the \"no centralization\"
    floor baseline from Yu et al. 2022 (arXiv 2103.01955).
    \"\"\"

    def __init__(self, scheme, args):
        super(CustomCritic, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.output_type = "v"

        obs_dim = int(scheme["obs"]["vshape"])
        input_shape = obs_dim + self.n_agents   # obs + agent-one-hot
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        obs = batch["obs"][:, ts]                                        # (B, T, n, obs_dim)
        agent_id = th.eye(self.n_agents, device=batch.device)
        agent_id = agent_id.unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
        inputs = th.cat([obs, agent_id], dim=-1)                         # (B, T, n, obs+n)

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
        "content": _IPPO_CLASS,
    },
]
