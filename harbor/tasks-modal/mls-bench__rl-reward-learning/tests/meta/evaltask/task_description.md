# Inverse RL: Reward Learning from Expert Demonstrations

## Research Question
Design and implement an inverse reinforcement learning (IRL) algorithm
that learns a reward function from expert demonstrations. Your code goes
in `custom_irl.py`, specifically the `RewardNetwork` and `IRLAlgorithm`
classes. Several reference implementations from the `imitation` library
are provided as read-only `*.edit.py` baselines.

## Background
Inverse reinforcement learning recovers a reward function that explains
observed expert behavior. The learned reward is then used to train a
policy via standard RL — PPO in this benchmark — and the resulting
policy is scored against the true environment reward. Key challenges
include:

- Designing reward network architectures that capture the structure of
  expert behavior.
- Balancing discriminator / reward training with policy improvement.
- Avoiding reward hacking, where the policy exploits artifacts in the
  learned reward.
- Ensuring the learned reward generalizes across the state distribution
  visited during policy training.

Reference baselines spanning the design space:
- **BC** — supervised behavior cloning on the expert state-action
  pairs. Does not learn a reward.
- **GAIL** — Ho and Ermon, "Generative Adversarial Imitation Learning"
  (arXiv:1606.03476, NeurIPS 2016). Adversarial discriminator between
  expert and policy occupancy; the policy is trained with the
  discriminator-derived reward.
- **AIRL** — Fu et al., "Learning Robust Rewards with Adversarial
  Inverse Reinforcement Learning" (arXiv:1710.11248, ICLR 2018).
  Adversarial reward learning with a state-only reward decomposition
  that yields a reward robust to dynamics changes.

## Evaluation
Trained and evaluated on Gymnasium MuJoCo locomotion environments
including HalfCheetah-v4, Hopper-v4 and Walker2d-v4 using
pre-generated expert demonstrations bundled with the benchmark. The
PPO policy is trained with the learned reward signal and evaluated
under the true environment reward. Metric: mean episodic return over
evaluation episodes (higher is better). Strong methods should learn
rewards that generalize across state distributions and task dynamics.
