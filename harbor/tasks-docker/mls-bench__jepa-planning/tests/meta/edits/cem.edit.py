"""CEM (Cross-Entropy Method) baseline -- rigorous codebase edit ops.

Replaces the CustomPlanner stub with a CEM implementation adapted from
CEMPlanner in eb_jepa/planning.py.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "eb_jepa/custom_planner.py"

# -- Replace the CustomPlanner class (lines 323-367) --

_CEM_CLASS = """\
class CustomPlanner(Planner):
    \"\"\"CEM (Cross-Entropy Method) planner for JEPA world models.\"\"\"

    def __init__(self, unroll, action_dim=2, plan_length=15,
                 num_samples=200, n_iters=20, **kwargs):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.plan_length = plan_length
        self.num_samples = num_samples
        self.n_iters = n_iters
        self.num_elites = max(10, num_samples // 10)
        self.var_scale = 1.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def plan(self, obs_init, steps_left=None, eval_mode=True,
             t0=False, plan_vis_path=None):
        from einops import rearrange

        plan_length = min(self.plan_length, steps_left) if steps_left else self.plan_length

        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.var_scale * torch.ones(plan_length, self.action_dim, device=self.device)
        actions = torch.empty(plan_length, self.num_samples, self.action_dim, device=self.device)

        losses = []
        elite_means = []
        elite_stds = []

        for _ in range(self.n_iters):
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length, self.num_samples, self.action_dim, device=self.device,
            )

            # Clip action norms
            max_norm = 2.45
            eps = 1e-6
            norms = actions.norm(dim=-1, keepdim=True)
            max_norms = torch.ones_like(norms) * max_norm
            min_norms = torch.ones_like(norms) * 0
            coeff = torch.min(torch.max(norms, min_norms), max_norms) / (norms + eps)
            actions = actions * coeff

            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init
            ).unsqueeze(1)
            losses.append(cost.min().item())

            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]

            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            mean = torch.mean(elite_actions, dim=1)
            std = torch.std(elite_actions, dim=1)

        return PlanningResult(
            actions=mean,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 323,
        "end_line": 367,
        "content": _CEM_CLASS,
    },
]
