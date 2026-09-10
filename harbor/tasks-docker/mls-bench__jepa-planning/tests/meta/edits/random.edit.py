"""Random Search baseline -- rigorous codebase edit ops.

Replaces the CustomPlanner stub with a random search implementation that
samples action sequences and returns the best one. Serves as a lower bound.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "eb_jepa/custom_planner.py"

# -- Replace the CustomPlanner class (lines 323-367) --

_RANDOM_CLASS = """\
class CustomPlanner(Planner):
    \"\"\"Random Search planner (lower bound baseline).

    Samples random action sequences and returns the one with lowest cost.
    No iterative refinement -- purely single-pass random sampling.
    \"\"\"

    def __init__(self, unroll, action_dim=2, plan_length=15,
                 num_samples=200, n_iters=20, **kwargs):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.plan_length = plan_length
        self.num_samples = num_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def plan(self, obs_init, steps_left=None, eval_mode=True,
             t0=False, plan_vis_path=None):
        from einops import rearrange

        plan_length = min(self.plan_length, steps_left) if steps_left else self.plan_length

        # Sample random actions
        actions = torch.randn(
            plan_length, self.num_samples, self.action_dim, device=self.device
        )

        # Clip action norms
        max_norm = 2.45
        norms = actions.norm(dim=-1, keepdim=True)
        actions = actions * (max_norm / norms.clamp(min=1e-6)).clamp(max=1.0)

        # Evaluate all samples and pick the best
        cost = self.cost_function(
            rearrange(actions, "t b a -> b a t"), obs_init
        )
        best_idx = cost.argmin()

        return PlanningResult(actions=actions[:, best_idx])
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 323,
        "end_line": 367,
        "content": _RANDOM_CLASS,
    },
]
