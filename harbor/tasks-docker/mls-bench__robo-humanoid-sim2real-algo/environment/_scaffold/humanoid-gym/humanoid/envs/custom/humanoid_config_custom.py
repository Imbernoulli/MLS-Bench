# Custom environment configuration for robo-humanoid-sim2real-algo task.
# Algorithm is modified via actor_critic_custom.py / ppo_custom.py / rollout_storage_custom.py.
# The commands.ranges block below is editable, but the default values mirror the
# official XBot recipe from humanoid_config.py.

from humanoid.envs.custom.humanoid_config import XBotLCfg, XBotLCfgPPO


class XBotLCustomCfg(XBotLCfg):
    """Custom environment config - inherits XBotLCfg, overrides command ranges only."""

    class commands(XBotLCfg.commands):
        # READ-ONLY: official XBot training command distribution. Locked to keep
        # the train→eval comparison fair (eval samples vx∈[-0.5,1.0] and the
        # hidden high-speed env tests vx=1.5, deliberately widening beyond
        # training; agents proposing algorithmic improvements should not also
        # widen the training distribution to score higher on the hidden env).
        # heading_command=False so the policy's third command channel is the
        # raw ang_vel_yaw target, matching what the MuJoCo sim2sim eval feeds.
        # Upstream's heading_command=True samples a heading target and converts
        # to a corrective ang_vel internally, which produces a train/eval
        # contract mismatch.
        heading_command = False
        class ranges:
            lin_vel_x = [-0.3, 0.6]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.3, 0.3]
            heading = [-3.14, 3.14]


class XBotLCustomCfgPPO(XBotLCfgPPO):
    """Custom PPO runner config - uses the custom algorithm classes."""

    class algorithm(XBotLCfgPPO.algorithm):
        # EDITABLE: tune PPO hyperparameters per algorithm variant.
        # Defaults mirror XBotLCfgPPO.algorithm so existing baselines are unaffected.
        learning_rate = 1.0e-5
        entropy_coef = 0.001
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner(XBotLCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        experiment_name = 'XBot_ppo'
