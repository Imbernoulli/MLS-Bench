# Humanoid Robot Sim2Real: Algorithm Design

## Objective
Design novel reinforcement learning algorithm components for humanoid robot locomotion that achieve robust sim-to-real transfer. You implement custom components in the PPO (Proximal Policy Optimization) framework so that policies can follow diverse 3D velocity commands with natural, stable gaits.

## Research Question
**What algorithm implementations (network architecture, policy optimization, experience replay) lead to policies that can successfully execute diverse locomotion commands in sim2sim transfer (Isaac Gym → MuJoCo)?**

The key challenge: standard PPO implementations often struggle with diverse command following and sim-to-real transfer. Your modifications should improve:
- Policy robustness across varied commands (different speeds, directions, turning rates)
- Sample efficiency during training
- Generalization from simulation to simulation (Isaac Gym → MuJoCo)
- Natural, energy-efficient gaits
- Stable transitions between different commands

## Background
The humanoid locomotion task requires the robot to track 3D velocity commands:
- `vx`: forward / backward velocity (m/s)
- `vy`: lateral velocity (m/s)
- `dyaw`: yaw angular velocity (rad/s)

The training and evaluation infrastructure follows **Humanoid-Gym** (Gu, Wang, Chen, 2024, arXiv:2404.05695, https://github.com/roboterax/humanoid-gym), an Isaac Gym → MuJoCo RL framework verified on RobotEra's XBot-S/XBot-L. The standard PPO algorithm consists of three main components:
1. **Actor-Critic Network** (`actor_critic.py`): neural network architecture with separate actor (policy) and critic (value function) heads.
2. **PPO Optimizer** (`ppo.py`): policy optimization using a clipped surrogate objective, value function loss, and entropy regularization.
3. **Rollout Storage** (`rollout_storage.py`): experience buffer for collecting and processing trajectories.

**The problem**: standard implementations often struggle with diverse command distributions, have poor sample efficiency on complex locomotion tasks, fail to transfer between simulators (Isaac Gym → MuJoCo), produce unnatural or unstable gaits, and require extensive hyperparameter tuning.

## Task
Implement custom algorithm components in the editable sections of the following files:

**1. Actor-Critic Network: `actor_critic_custom.py`**

In `ActorCritic.__init__`:
- Design custom network architecture (layer sizes, activation functions)
- Add normalization layers (LayerNorm, BatchNorm, custom)
- Implement custom initialization schemes
- Add auxiliary heads or features

In `ActorCritic.act`:
- Modify action sampling strategy
- Add custom exploration mechanisms
- Implement action post-processing

In `ActorCritic.evaluate_actions`:
- Customize value function computation
- Modify action log probability calculation
- Add auxiliary losses or regularization

**2. PPO Optimizer: `ppo_custom.py`**

In `PPO.__init__`:
- Configure optimizer settings
- Set up learning rate schedules
- Initialize custom training components

In `PPO.update`:
- Modify policy loss computation (clipping strategy, advantage normalization)
- Customize value function loss (Huber loss, clipping, multi-step returns)
- Adjust entropy regularization
- Implement custom gradient clipping or normalization
- Add auxiliary losses (e.g. behavioral cloning, imitation)

**3. Rollout Storage: `rollout_storage_custom.py`**

In `RolloutStorage.__init__`:
- Design custom buffer structure
- Add additional tracking tensors

In `RolloutStorage.add_transitions`:
- Customize how experiences are stored
- Add data augmentation or preprocessing

In `RolloutStorage.compute_returns`:
- Modify advantage estimation (GAE parameters, normalization)
- Implement custom return computation (n-step, λ-returns)
- Add reward shaping or preprocessing

**4. Training command distribution: `humanoid_config_custom.py`**

The training command ranges are editable, but the default values mirror the official XBot recipe: `vx ∈ [-0.3, 0.6]`, `vy ∈ [-0.3, 0.3]`, `dyaw ∈ [-0.3, 0.3]`. Keep these defaults for paper-aligned comparisons; widen them only as an explicit algorithmic choice.

**5. PPO hyperparameters: `humanoid_config_custom.py`**

PPO algorithm hyperparameters (`learning_rate`, `entropy_coef`, `num_learning_epochs`, `gamma`, `lam`, `num_mini_batches`) are editable per baseline so different algorithm variants (e.g. adaptive-KL, layernorm) can use their own training recipe without editing the algorithm code. Architecture and infrastructure constants (`num_envs`, `max_iterations`, `num_steps_per_env`, network sizes) remain fixed for fair comparison.

**Fixed components**:
- Environment and reward functions
- Training: 4096 parallel environments, official XBot iteration budget
- Observation / action spaces

## Evaluation
Trained in Isaac Gym (4096 parallel envs, official XBot iteration budget), then evaluated on **100 diverse random commands in MuJoCo (sim2sim transfer)**:

**Test procedure**:
1. Sample 100 random commands from ranges:
   - `vx`: [-0.5, 1.0] m/s
   - `vy`: [-0.4, 0.4] m/s
   - `dyaw`: [-0.5, 0.5] rad/s
2. For each command, run a 10-second episode in MuJoCo.
3. **Success criteria** (per command):
   - Robot doesn't fall (base height > 0.3 m, |roll|, |pitch| < 0.5 rad)
   - Average velocity tracking error < 0.5 (linear-norm + |yaw| error)

**Metrics**:
- `success_rate`: fraction of commands meeting both criteria above (primary metric, higher is better)
- `avg_vel_error`: average velocity tracking error across all 100 commands (lower is better)
- `fall_rate`: fraction of commands where the robot fell during the episode (lower is better)

## Reference Implementation
**default**: standard PPO implementation from Humanoid-Gym
- 3-layer MLP with [512, 256, 128] hidden units
- Standard PPO loss with clipping (ε = 0.2)
- Actor [512, 256, 128], critic [768, 256, 128]
- GAE with λ = 0.9, γ = 0.994

## Hints
- **Network architecture**: deeper networks, normalization layers, or residual connections may improve learning.
- **Advantage normalization**: proper normalization can stabilize training.
- **Value function**: accurate value estimates improve policy learning.
- **Exploration**: entropy regularization or noise injection can help exploration.
- **Sample efficiency**: better advantage estimation or multi-step returns can improve efficiency.
- **Sim2sim transfer**: algorithms that learn robust features transfer better.
- Consider:
  - Adaptive learning rates or schedules
  - Custom loss weightings (policy vs value vs entropy)
  - Gradient clipping strategies
  - Observation / action normalization
  - Auxiliary tasks or losses
