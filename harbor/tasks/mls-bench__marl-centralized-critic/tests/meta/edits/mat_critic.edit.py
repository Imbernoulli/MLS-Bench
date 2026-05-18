"""Attention-based centralized critic baseline — rigorous codebase edit ops.

SOTA-representative for the "attention over agents" family of centralized
critics. Adapted from Multi-Agent Transformer (Wen et al. 2022,
arXiv 2205.14953) and MAAC (Iqbal & Sha 2019, ICML, arXiv 1810.02912),
both of which use per-agent tokens + self-attention to learn a
centralized value.

This is the **critic-only** form: we keep the standard MAPPO actor
(RNN policy over individual agents, independent at execution time) and
replace only the value function with a single TransformerEncoder layer
that mixes information across the agent axis. Full MAT additionally
replaces the actor with a transformer decoder that generates actions
auto-regressively across agents; we do NOT do that here because the
task's research question is specifically centralized critic architecture.

Per-agent token = Linear(obs_i ⊕ broadcast(state)) → d_model. A single
TransformerEncoder layer (nhead=4, dim_ff=4*d_model) lets each agent
attend to every other agent; a per-token linear head outputs the value.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "epymarl/src/modules/critics/custom_critic.py"

# ── Replace CustomCritic class (lines 13-69) ──────────────────────────

_MAT_CLASS = """\
class CustomCritic(nn.Module):
    \"\"\"MAT-style attention critic — self-attention over per-agent tokens.

    Adapted from Wen et al. 2022 MAT (arXiv 2205.14953), critic-only form.
    Each agent's token encodes its local observation together with the
    global state; a single TransformerEncoder layer mixes information
    across agents via self-attention, then a per-token linear head
    produces the scalar value.
    \"\"\"

    def __init__(self, scheme, args):
        super(CustomCritic, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.output_type = "v"

        obs_dim = int(scheme["obs"]["vshape"])
        state_dim = int(scheme["state"]["vshape"])
        self.d_model = args.hidden_dim

        # Per-agent token projection: [obs_i ⊕ state] → d_model
        self.token_proj = nn.Linear(obs_dim + state_dim, self.d_model)

        # Single transformer encoder layer with self-attention across agents
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=4 * self.d_model,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

        # Per-agent value head
        self.v_head = nn.Linear(self.d_model, 1)

    def forward(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        obs = batch["obs"][:, ts]                                        # (B, T, n, obs_dim)
        state = batch["state"][:, ts]                                    # (B, T, state_dim)
        state = state.unsqueeze(2).expand(-1, -1, self.n_agents, -1)     # (B, T, n, state_dim)
        tokens = th.cat([obs, state], dim=-1)                            # (B, T, n, obs+state)
        tokens = self.token_proj(tokens)                                 # (B, T, n, d_model)

        # Flatten (B, T) into a single batch dim for the transformer,
        # then restore: TransformerEncoder expects (bs*, seq_len, d_model).
        b, tt, n, d = tokens.shape
        tokens = tokens.reshape(b * tt, n, d)
        attn_out = self.encoder(tokens)                                  # (B*T, n, d_model)
        attn_out = attn_out.reshape(b, tt, n, d)

        q = self.v_head(attn_out)                                       # (B, T, n, 1)
        return q
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 13,
        "end_line": 69,
        "content": _MAT_CLASS,
    },
]
