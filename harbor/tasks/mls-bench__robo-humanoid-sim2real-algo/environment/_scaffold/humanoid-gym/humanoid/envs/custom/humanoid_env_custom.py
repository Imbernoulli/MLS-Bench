# Custom environment class for robo-humanoid-sim2real-algo task.
# XBotLCustomEnv inherits from XBotLFreeEnv without modification;
# the research contribution is in the algorithm (PPO/ActorCritic/RolloutStorage).

from humanoid.envs.custom.humanoid_env import XBotLFreeEnv


class XBotLCustomEnv(XBotLFreeEnv):
    """Custom environment - same as XBotLFreeEnv."""
    pass
