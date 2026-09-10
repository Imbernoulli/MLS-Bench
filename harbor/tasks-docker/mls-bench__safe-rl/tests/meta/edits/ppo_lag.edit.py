"""PPOLag (Lagrangian PPO) baseline -- rigorous codebase edit ops.

Inline Adam-based Lagrange multiplier: the multiplier is a torch Parameter
optimized by Adam with loss = -lambda * (Jc - cost_limit). When Jc exceeds
the limit the gradient pushes lambda up; advantage = (adv_r - lam*adv_c)/(1+lam).

Reference: omnisafe/omnisafe/algorithms/on_policy/naive_lagrange/ppo_lag.py

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "omnisafe/omnisafe/algorithms/on_policy/naive_lagrange/custom_lag.py"

_PPO_LAG_IMPORTS = """\

"""

_PPO_LAG_METHODS = """\
    def _init(self) -> None:
        super()._init()
        self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
        init_value = max(self._cfgs.lagrange_cfgs.lagrangian_multiplier_init, 0.0)
        self._lagrangian_multiplier = torch.nn.Parameter(
            torch.as_tensor(init_value), requires_grad=True,
        )
        self._lambda_optimizer = torch.optim.Adam(
            [self._lagrangian_multiplier],
            lr=self._cfgs.lagrange_cfgs.lambda_lr,
        )

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        # Lagrange multiplier update via Adam
        self._lambda_optimizer.zero_grad()
        lambda_loss = -self._lagrangian_multiplier * (Jc - self._cost_limit)
        lambda_loss.backward()
        self._lambda_optimizer.step()
        self._lagrangian_multiplier.data.clamp_(0.0)
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier.item()})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        \"\"\"PPOLag: penalize cost advantage using Lagrange multiplier.\"\"\"
        penalty = self._lagrangian_multiplier.item()
        return (adv_r - penalty * adv_c) / (1 + penalty)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 48,
        "end_line": 70,
        "content": _PPO_LAG_METHODS,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 20,
        "end_line": 20,
        "content": _PPO_LAG_IMPORTS,
    },
]
