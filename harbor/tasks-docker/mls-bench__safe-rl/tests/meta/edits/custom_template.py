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
    def _init_env(self) -> None:
        super()._init_env()
        self._mlsbench_env_steps = [0]
        self._count_env_steps()

    def _count_env_steps(self) -> None:
        """Make every step of the innermost env(s) under ``self._env`` add to the step count."""
        count = self._mlsbench_env_steps
        for env in (self._env, getattr(self._env, '_eval_env', None)):
            while env is not None and {'_env', 'env'} & vars(env).keys():
                env = vars(env).get('_env', vars(env).get('env'))
            if env is None or getattr(env.step, 'mlsbench_counted', False):
                continue

            def counted_step(*args, _step=env.step, _n=getattr(env, 'num_envs', 1), **kwargs):
                count[0] += _n
                return _step(*args, **kwargs)

            counted_step.mlsbench_counted = True
            env.step = counted_step

    def learn(self) -> tuple[float, float, float]:
        """Training loop with TRAIN_METRICS and TEST_METRICS output.

        The reported ep_ret / ep_cost / ep_len are the mean of the last 100
        episodes (OmniSafe's default ``logger_cfgs.window_lens``), taken from a
        copy that only the rollout below writes to: values stored into
        ``self._logger`` elsewhere do not reach them.

        Every step taken on the environment ``self._env`` wraps, from its
        creation on and by any code, is counted against the ``--total-steps``
        budget on the command line (not ``self._cfgs``, which editable code
        can change). A run that is over its pro-rata share of the budget at the
        end of any epoch is aborted.
        """
        import argparse
        import os
        from collections import deque

        from omnisafe.utils.distributed import dist_statistics_scalar

        start_time = time.time()
        self._logger.log('INFO: Start training')

        cli = argparse.ArgumentParser(add_help=False)
        cli.add_argument('--total-steps', type=int)
        budget = cli.parse_known_args()[0].total_steps
        if budget is None:  # not launched through train_safe_rl.py
            budget = int(self._cfgs.train_cfgs.total_steps)
        epochs = self._cfgs.train_cfgs.epochs
        env_steps = self._mlsbench_env_steps

        logger = self._logger
        episodes = {key: deque(maxlen=100) for key in ('Metrics/EpRet', 'Metrics/EpCost', 'Metrics/EpLen')}

        class RolloutLogger:
            """Forwards to the logger and keeps a copy of the episode statistics."""

            def __getattr__(self, name):
                return getattr(logger, name)

            def store(self, data=None, /, **kwargs):
                logger.store(data, **kwargs)
                if data is not None:
                    kwargs.update(data)
                for key in episodes.keys() & kwargs.keys():
                    val = kwargs[key]  # converted as Logger.store converts it
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    elif isinstance(val, np.ndarray):
                        val = val.mean()
                    episodes[key].append(val)

        def episode_mean(key: str) -> float:
            """What ``Logger.get_stats(key)[0]`` computes, over the copy."""
            vals = torch.tensor(list(episodes[key])).to(os.getenv('OMNISAFE_DEVICE', 'cpu'))
            return dist_statistics_scalar(vals)[0].item()

        for epoch in range(epochs):
            self._count_env_steps()  # also if editable code replaced self._env
            epoch_time = time.time()

            rollout_time = time.time()
            self._env.rollout(
                steps_per_epoch=self._steps_per_epoch,
                agent=self._actor_critic,
                buffer=self._buf,
                logger=RolloutLogger(),
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
            _ep_ret = episode_mean('Metrics/EpRet')
            _ep_cost = episode_mean('Metrics/EpCost')
            _ep_len = episode_mean('Metrics/EpLen')
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

            if env_steps[0] > budget * (epoch + 1) // epochs:
                raise RuntimeError(
                    f'MLS-Bench step budget exceeded: {env_steps[0]} environment steps after '
                    f'{epoch + 1} of {epochs} epochs, over the pro-rata share of '
                    f'--total-steps {budget}',
                )

        ep_ret = episode_mean('Metrics/EpRet')
        ep_cost = episode_mean('Metrics/EpCost')
        ep_len = episode_mean('Metrics/EpLen')

        # -- MLS-Bench: TEST_METRICS --
        print(
            f'TEST_METRICS ep_ret={ep_ret:.4f} ep_cost={ep_cost:.4f} '
            f'ep_len={ep_len:.1f}',
            flush=True,
        )

        self._logger.close()
        self._env.close()

        return ep_ret, ep_cost, ep_len
