"""Naive baseline (no constraint handling) -- rigorous codebase edit ops.

Pure PPO without any safety constraint mechanism. The Lagrange multiplier
stays at zero and cost advantage is completely ignored.
This serves as the lower bound for safe RL performance.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "omnisafe/omnisafe/algorithms/on_policy/naive_lagrange/custom_lag.py"

_NAIVE_IMPORTS = """\

"""

_NAIVE_METHODS = """\
    def _init(self) -> None:
        super()._init()
        self._lagrangian_multiplier: float = 0.0

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        # Naive: no multiplier update, stays at 0
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        \"\"\"Naive: ignore cost advantage entirely, optimize reward only.\"\"\"
        return adv_r
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 48,
        "end_line": 70,
        "content": _NAIVE_METHODS,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 20,
        "end_line": 20,
        "content": _NAIVE_IMPORTS,
    },
]
