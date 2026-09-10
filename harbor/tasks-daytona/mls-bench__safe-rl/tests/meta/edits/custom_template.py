"""Custom Lagrangian-based safe PPO for MLS-Bench.

EDITABLE section: imports + constraint handling methods.
FIXED sections: algorithm registration, learn() with metrics reporting.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO

# ===================================================================
# EDITABLE: Custom imports
# ===================================================================


# ===================================================================
# FIXED: Algorithm class definition
# ===================================================================
@registry.register
class CustomLag(PPO):
    """Custom Lagrangian-based safe RL algorithm.

    Extends PPO with constraint handling for safe reinforcement learning.
    The agent must design:
      1. _init: Initialize constraint handler state (call super()._init() first)
      2. _init_log: Register logging keys (call super()._init_log() first)
      3. _update: Update lagrangian multiplier, then call super()._update()
      4. _compute_adv_surrogate: Combine reward and cost advantages

    Available config:
        self._cfgs.lagrange_cfgs.cost_limit   (float, default 25.0)
        self._cfgs.lagrange_cfgs.lambda_lr    (float, default 0.035)

    Available logger:
        self._logger.get_stats('Metrics/EpCost')[0]  -- current mean episode cost
        self._logger.store({'key': value})             -- log a metric value
    """

    # ===============================================================
    # EDITABLE: Constraint handling mechanism
    # ===============================================================
    def _init(self) -> None:
        super()._init()
        self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
        self._lagrangian_multiplier: float = 0.0

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        # Default: no multiplier update -- agent should design this
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        """Combine reward and cost advantages.

        Default: only use reward advantage (ignores safety constraints entirely).
        Agent should incorporate self._lagrangian_multiplier to penalize cost.
        """
        return adv_r

    # ===============================================================
    # FIXED: Training loop with MLS-Bench metrics reporting
    # ===============================================================
    def learn(self) -> tuple[float, float, float]:
        """Training loop with TRAIN_METRICS and TEST_METRICS output."""
        start_time = time.time()
        self._logger.log('INFO: Start training')

        for epoch in range(self._cfgs.train_cfgs.epochs):
            epoch_time = time.time()

            rollout_time = time.time()
            self._env.rollout(
                steps_per_epoch=self._steps_per_epoch,
                agent=self._actor_critic,
                buffer=self._buf,
                logger=self._logger,
            )
            self._logger.store({'Time/Rollout': time.time() - rollout_time})

            update_time = time.time()
            self._update()
            self._logger.store({'Time/Update': time.time() - update_time})

            if self._cfgs.model_cfgs.exploration_noise_anneal:
                self._actor_critic.annealing(epoch)

            if self._cfgs.model_cfgs.actor.lr is not None:
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': (
                        0.0
                        if self._cfgs.model_cfgs.actor.lr is None
                        else self._actor_critic.actor_scheduler.get_last_lr()[0]
                    ),
                },
            )

            self._logger.dump_tabular()

            # -- MLS-Bench: TRAIN_METRICS --
            _ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
            _ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
            _ep_len = self._logger.get_stats('Metrics/EpLen')[0]
            print(
                f'TRAIN_METRICS epoch={epoch} '
                f'ep_ret={_ep_ret:.4f} ep_cost={_ep_cost:.4f} '
                f'ep_len={_ep_len:.1f}',
                flush=True,
            )

            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or (
                epoch + 1
            ) == self._cfgs.train_cfgs.epochs:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]

        # -- MLS-Bench: TEST_METRICS --
        print(
            f'TEST_METRICS ep_ret={ep_ret:.4f} ep_cost={ep_cost:.4f} '
            f'ep_len={ep_len:.1f}',
            flush=True,
        )

        self._logger.close()
        self._env.close()

        return ep_ret, ep_cost, ep_len
