# MLS-Bench: robo-humanoid-sim2real-algo

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


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/humanoid-gym/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `humanoid-gym/humanoid/algo/ppo/actor_critic_custom.py`
- editable lines **36–128**
- `humanoid-gym/humanoid/algo/ppo/ppo_custom.py`
- editable lines **39–185**
- `humanoid-gym/humanoid/algo/ppo/rollout_storage_custom.py`
- editable lines **34–182**
- `humanoid-gym/humanoid/envs/custom/humanoid_config_custom.py`
- editable lines **29–34**


Other files you may **read** for context (do not modify):
- `humanoid-gym/humanoid/envs/custom/humanoid_config.py`
- `humanoid-gym/humanoid/algo/ppo/actor_critic.py`
- `humanoid-gym/humanoid/algo/ppo/ppo.py`
- `humanoid-gym/humanoid/algo/ppo/rollout_storage.py`


## Readable Context


### `humanoid-gym/humanoid/algo/ppo/actor_critic_custom.py`  [EDITABLE — lines 36–128 only]

```python
     1: # SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
     2: # SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
     3: # SPDX-License-Identifier: BSD-3-Clause
     4: #
     5: # Redistribution and use in source and binary forms, with or without
     6: # modification, are permitted provided that the following conditions are met:
     7: #
     8: # 1. Redistributions of source code must retain the above copyright notice, this
     9: # list of conditions and the following disclaimer.
    10: #
    11: # 2. Redistributions in binary form must reproduce the above copyright notice,
    12: # this list of conditions and the following disclaimer in the documentation
    13: # and/or other materials provided with the distribution.
    14: #
    15: # 3. Neither the name of the copyright holder nor the names of its
    16: # contributors may be used to endorse or promote products derived from
    17: # this software without specific prior written permission.
    18: #
    19: # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    20: # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    21: # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    22: # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    23: # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    24: # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    25: # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    26: # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    27: # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    28: # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    29: #
    30: # Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
    31: 
    32: import torch
    33: import torch.nn as nn
    34: from torch.distributions import Normal
    35: 
    36: class ActorCritic(nn.Module):
    37:     def __init__(self,  num_actor_obs,
    38:                         num_critic_obs,
    39:                         num_actions,
    40:                         actor_hidden_dims=[256, 256, 256],
    41:                         critic_hidden_dims=[256, 256, 256],
    42:                         init_noise_std=1.0,
    43:                         activation = nn.ELU(),
    44:                         **kwargs):
    45:         if kwargs:
    46:             print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
    47:         super(ActorCritic, self).__init__()
    48: 
    49: 
    50:         mlp_input_dim_a = num_actor_obs
    51:         mlp_input_dim_c = num_critic_obs
    52:         # Policy
    53:         actor_layers = []
    54:         actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
    55:         actor_layers.append(activation)
    56:         for l in range(len(actor_hidden_dims)):
    57:             if l == len(actor_hidden_dims) - 1:
    58:                 actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
    59:             else:
    60:                 actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
    61:                 actor_layers.append(activation)
    62:         self.actor = nn.Sequential(*actor_layers)
    63: 
    64:         # Value function
    65:         critic_layers = []
    66:         critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
    67:         critic_layers.append(activation)
    68:         for l in range(len(critic_hidden_dims)):
    69:             if l == len(critic_hidden_dims) - 1:
    70:                 critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
    71:             else:
    72:                 critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
    73:                 critic_layers.append(activation)
    74:         self.critic = nn.Sequential(*critic_layers)
    75: 
    76:         print(f"Actor MLP: {self.actor}")
    77:         print(f"Critic MLP: {self.critic}")
    78: 
    79:         # Action noise
    80:         self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
    81:         self.distribution = None
    82:         # disable args validation for speedup
    83:         Normal.set_default_validate_args = False
    84: 
    85: 
    86:     @staticmethod
    87:     # not used at the moment
    88:     def init_weights(sequential, scales):
    89:         [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
    90:          enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
    91: 
    92: 
    93:     def reset(self, dones=None):
    94:         pass
    95: 
    96:     def forward(self):
    97:         raise NotImplementedError
    98: 
    99:     @property
   100:     def action_mean(self):
   101:         return self.distribution.mean
   102: 
   103:     @property
   104:     def action_std(self):
   105:         return self.distribution.stddev
   106: 
   107:     @property
   108:     def entropy(self):
   109:         return self.distribution.entropy().sum(dim=-1)
   110: 
   111:     def update_distribution(self, observations):
   112:         mean = self.actor(observations)
   113:         self.distribution = Normal(mean, mean*0. + self.std)
   114: 
   115:     def act(self, observations, **kwargs):
   116:         self.update_distribution(observations)
   117:         return self.distribution.sample()
   118: 
   119:     def get_actions_log_prob(self, actions):
   120:         return self.distribution.log_prob(actions).sum(dim=-1)
   121: 
   122:     def act_inference(self, observations):
   123:         actions_mean = self.actor(observations)
   124:         return actions_mean
   125: 
   126:     def evaluate(self, critic_observations, **kwargs):
   127:         value = self.critic(critic_observations)
   128:         return value
```

### `humanoid-gym/humanoid/algo/ppo/ppo_custom.py`  [EDITABLE — lines 39–185 only]

```python
     1: # SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
     2: # SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
     3: # SPDX-License-Identifier: BSD-3-Clause
     4: #
     5: # Redistribution and use in source and binary forms, with or without
     6: # modification, are permitted provided that the following conditions are met:
     7: #
     8: # 1. Redistributions of source code must retain the above copyright notice, this
     9: # list of conditions and the following disclaimer.
    10: #
    11: # 2. Redistributions in binary form must reproduce the above copyright notice,
    12: # this list of conditions and the following disclaimer in the documentation
    13: # and/or other materials provided with the distribution.
    14: #
    15: # 3. Neither the name of the copyright holder nor the names of its
    16: # contributors may be used to endorse or promote products derived from
    17: # this software without specific prior written permission.
    18: #
    19: # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    20: # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    21: # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    22: # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    23: # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    24: # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    25: # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    26: # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    27: # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    28: # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    29: #
    30: # Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
    31: 
    32: import torch
    33: import torch.nn as nn
    34: import torch.optim as optim
    35: 
    36: from .actor_critic_custom import ActorCritic
    37: from .rollout_storage_custom import RolloutStorage
    38: 
    39: class PPO:
    40:     actor_critic: ActorCritic
    41:     def __init__(self,
    42:                  actor_critic,
    43:                  num_learning_epochs=1,
    44:                  num_mini_batches=1,
    45:                  clip_param=0.2,
    46:                  gamma=0.998,
    47:                  lam=0.95,
    48:                  value_loss_coef=1.0,
    49:                  entropy_coef=0.0,
    50:                  learning_rate=1e-3,
    51:                  max_grad_norm=1.0,
    52:                  use_clipped_value_loss=True,
    53:                  schedule="fixed",
    54:                  desired_kl=0.01,
    55:                  device='cpu',
    56:                  ):
    57: 
    58:         self.device = device
    59: 
    60:         self.desired_kl = desired_kl
    61:         self.schedule = schedule
    62:         self.learning_rate = learning_rate
    63: 
    64:         # PPO components
    65:         self.actor_critic = actor_critic
    66:         self.actor_critic.to(self.device)
    67:         self.storage = None # initialized later
    68:         self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
    69:         self.transition = RolloutStorage.Transition()
    70: 
    71:         # PPO parameters
    72:         self.clip_param = clip_param
    73:         self.num_learning_epochs = num_learning_epochs
    74:         self.num_mini_batches = num_mini_batches
    75:         self.value_loss_coef = value_loss_coef
    76:         self.entropy_coef = entropy_coef
    77:         self.gamma = gamma
    78:         self.lam = lam
    79:         self.max_grad_norm = max_grad_norm
    80:         self.use_clipped_value_loss = use_clipped_value_loss
    81: 
    82:     def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
    83:         self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)
    84: 
    85:     def test_mode(self):
    86:         self.actor_critic.eval()
    87: 
    88:     def train_mode(self):
    89:         self.actor_critic.train()
    90: 
    91:     def act(self, obs, critic_obs):
    92:         # Compute the actions and values
    93:         self.transition.actions = self.actor_critic.act(obs).detach()
    94:         self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
    95:         self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
    96:         self.transition.action_mean = self.actor_critic.action_mean.detach()
    97:         self.transition.action_sigma = self.actor_critic.action_std.detach()
    98:         # need to record obs and critic_obs before env.step()
    99:         self.transition.observations = obs
   100:         self.transition.critic_observations = critic_obs
   101:         return self.transition.actions
   102: 
   103:     def process_env_step(self, rewards, dones, infos):
   104:         self.transition.rewards = rewards.clone()
   105:         self.transition.dones = dones
   106:         # Bootstrapping on time outs
   107:         if 'time_outs' in infos:
   108:             self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
   109: 
   110:         # Record the transition
   111:         self.storage.add_transitions(self.transition)
   112:         self.transition.clear()
   113:         self.actor_critic.reset(dones)
   114: 
   115:     def compute_returns(self, last_critic_obs):
   116:         last_values = self.actor_critic.evaluate(last_critic_obs).detach()
   117:         self.storage.compute_returns(last_values, self.gamma, self.lam)
   118: 
   119:     def update(self):
   120:         mean_value_loss = 0
   121:         mean_surrogate_loss = 0
   122: 
   123:         generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
   124:         for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch,             old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
   125: 
   126: 
   127:                 self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
   128:                 actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
   129:                 value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
   130:                 mu_batch = self.actor_critic.action_mean
   131:                 sigma_batch = self.actor_critic.action_std
   132:                 entropy_batch = self.actor_critic.entropy
   133: 
   134:                 # KL
   135:                 if self.desired_kl != None and self.schedule == 'adaptive':
   136:                     with torch.inference_mode():
   137:                         kl = torch.sum(
   138:                             torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
   139:                         kl_mean = torch.mean(kl)
   140: 
   141:                         if kl_mean > self.desired_kl * 2.0:
   142:                             self.learning_rate = max(1e-5, self.learning_rate / 1.5)
   143:                         elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
   144:                             self.learning_rate = min(1e-2, self.learning_rate * 1.5)
   145: 
   146:                         for param_group in self.optimizer.param_groups:
   147:                             param_group['lr'] = self.learning_rate
   148: 
   149: 
   150:                 # Surrogate loss
   151:                 ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
   152:                 surrogate = -torch.squeeze(advantages_batch) * ratio
   153:                 surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
   154:                                                                                 1.0 + self.clip_param)
   155:                 surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
   156: 
   157:                 # Value function loss
   158:                 if self.use_clipped_value_loss:
   159:                     value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
   160:                                                                                                     self.clip_param)
   161:                     value_losses = (value_batch - returns_batch).pow(2)
   162:                     value_losses_clipped = (value_clipped - returns_batch).pow(2)
   163:                     value_loss = torch.max(value_losses, value_losses_clipped).mean()
   164:                 else:
   165:                     value_loss = (returns_batch - value_batch).pow(2).mean()
   166: 
   167:                 loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
   168: 
   169:                 # Gradient step
   170:                 self.optimizer.zero_grad()
   171:                 loss.backward()
   172:                 nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
   173:                 self.optimizer.step()
   174: 
   175:                 mean_value_loss += value_loss.item()
   176:                 mean_surrogate_loss += surrogate_loss.item()
   177: 
   178:         num_updates = self.num_learning_epochs * self.num_mini_batches
   179:         mean_value_loss /= num_updates
   180:         mean_surrogate_loss /= num_updates
   181:         self.storage.clear()
   182: 
   183:         return mean_value_loss, mean_surrogate_loss
   184: 
```

### `humanoid-gym/humanoid/algo/ppo/rollout_storage_custom.py`  [EDITABLE — lines 34–182 only]

```python
     1: # SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
     2: # SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
     3: # SPDX-License-Identifier: BSD-3-Clause
     4: #
     5: # Redistribution and use in source and binary forms, with or without
     6: # modification, are permitted provided that the following conditions are met:
     7: #
     8: # 1. Redistributions of source code must retain the above copyright notice, this
     9: # list of conditions and the following disclaimer.
    10: #
    11: # 2. Redistributions in binary form must reproduce the above copyright notice,
    12: # this list of conditions and the following disclaimer in the documentation
    13: # and/or other materials provided with the distribution.
    14: #
    15: # 3. Neither the name of the copyright holder nor the names of its
    16: # contributors may be used to endorse or promote products derived from
    17: # this software without specific prior written permission.
    18: #
    19: # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    20: # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    21: # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    22: # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    23: # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    24: # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    25: # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    26: # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    27: # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    28: # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    29: #
    30: # Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
    31: 
    32: import torch
    33: 
    34: class RolloutStorage:
    35:     class Transition:
    36:         def __init__(self):
    37:             self.observations = None
    38:             self.critic_observations = None
    39:             self.actions = None
    40:             self.rewards = None
    41:             self.dones = None
    42:             self.values = None
    43:             self.actions_log_prob = None
    44:             self.action_mean = None
    45:             self.action_sigma = None
    46:             self.hidden_states = None
    47: 
    48:         def clear(self):
    49:             self.__init__()
    50: 
    51:     def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device='cpu'):
    52: 
    53:         self.device = device
    54: 
    55:         self.obs_shape = obs_shape
    56:         self.privileged_obs_shape = privileged_obs_shape
    57:         self.actions_shape = actions_shape
    58: 
    59:         # Core
    60:         self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
    61:         if privileged_obs_shape[0] is not None:
    62:             self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
    63:         else:
    64:             self.privileged_observations = None
    65:         self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
    66:         self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
    67:         self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
    68: 
    69:         # For PPO
    70:         self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
    71:         self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
    72:         self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
    73:         self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
    74:         self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
    75:         self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
    76: 
    77:         self.num_transitions_per_env = num_transitions_per_env
    78:         self.num_envs = num_envs
    79: 
    80:         # rnn
    81:         self.saved_hidden_states_a = None
    82:         self.saved_hidden_states_c = None
    83: 
    84:         self.step = 0
    85: 
    86:     def add_transitions(self, transition: Transition):
    87:         if self.step >= self.num_transitions_per_env:
    88:             raise AssertionError("Rollout buffer overflow")
    89:         self.observations[self.step].copy_(transition.observations)
    90:         if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(transition.critic_observations)
    91:         self.actions[self.step].copy_(transition.actions)
    92:         self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
    93:         self.dones[self.step].copy_(transition.dones.view(-1, 1))
    94:         self.values[self.step].copy_(transition.values)
    95:         self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
    96:         self.mu[self.step].copy_(transition.action_mean)
    97:         self.sigma[self.step].copy_(transition.action_sigma)
    98:         self._save_hidden_states(transition.hidden_states)
    99:         self.step += 1
   100: 
   101:     def _save_hidden_states(self, hidden_states):
   102:         if hidden_states is None or hidden_states==(None, None):
   103:             return
   104:         # make a tuple out of GRU hidden state to match the LSTM format
   105:         hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
   106:         hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
   107: 
   108:         # initialize if needed
   109:         if self.saved_hidden_states_a is None:
   110:             self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
   111:             self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
   112:         # copy the states
   113:         for i in range(len(hid_a)):
   114:             self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
   115:             self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])
   116: 
   117: 
   118:     def clear(self):
   119:         self.step = 0
   120: 
   121:     def compute_returns(self, last_values, gamma, lam):
   122:         advantage = 0
   123:         for step in reversed(range(self.num_transitions_per_env)):
   124:             if step == self.num_transitions_per_env - 1:
   125:                 next_values = last_values
   126:             else:
   127:                 next_values = self.values[step + 1]
   128:             next_is_not_terminal = 1.0 - self.dones[step].float()
   129:             delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
   130:             advantage = delta + next_is_not_terminal * gamma * lam * advantage
   131:             self.returns[step] = advantage + self.values[step]
   132: 
   133:         # Compute and normalize the advantages
   134:         self.advantages = self.returns - self.values
   135:         self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
   136: 
   137:     def get_statistics(self):
   138:         done = self.dones
   139:         done[-1] = 1
   140:         flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
   141:         done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
   142:         trajectory_lengths = (done_indices[1:] - done_indices[:-1])
   143:         return trajectory_lengths.float().mean(), self.rewards.mean()
   144: 
   145:     def mini_batch_generator(self, num_mini_batches, num_epochs=8):
   146:         batch_size = self.num_envs * self.num_transitions_per_env
   147:         mini_batch_size = batch_size // num_mini_batches
   148:         indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)
   149: 
   150:         observations = self.observations.flatten(0, 1)
   151:         if self.privileged_observations is not None:
   152:             critic_observations = self.privileged_observations.flatten(0, 1)
   153:         else:
   154:             critic_observations = observations
   155: 
   156:         actions = self.actions.flatten(0, 1)
   157:         values = self.values.flatten(0, 1)
   158:         returns = self.returns.flatten(0, 1)
   159:         old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
   160:         advantages = self.advantages.flatten(0, 1)
   161:         old_mu = self.mu.flatten(0, 1)
   162:         old_sigma = self.sigma.flatten(0, 1)
   163: 
   164:         for epoch in range(num_epochs):
   165:             for i in range(num_mini_batches):
   166: 
   167:                 start = i*mini_batch_size
   168:                 end = (i+1)*mini_batch_size
   169:                 batch_idx = indices[start:end]
   170: 
   171:                 obs_batch = observations[batch_idx]
   172:                 critic_observations_batch = critic_observations[batch_idx]
   173:                 actions_batch = actions[batch_idx]
   174:                 target_values_batch = values[batch_idx]
   175:                 returns_batch = returns[batch_idx]
   176:                 old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
   177:                 advantages_batch = advantages[batch_idx]
   178:                 old_mu_batch = old_mu[batch_idx]
   179:                 old_sigma_batch = old_sigma[batch_idx]
   180:                 yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,                        old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None
   181: 
```

### `humanoid-gym/humanoid/envs/custom/humanoid_config_custom.py`  [EDITABLE — lines 29–34 only]

```python
     1: # Custom environment configuration for robo-humanoid-sim2real-algo task.
     2: # Algorithm is modified via actor_critic_custom.py / ppo_custom.py / rollout_storage_custom.py.
     3: # The commands.ranges block below is editable, but the default values mirror the
     4: # official XBot recipe from humanoid_config.py.
     5: 
     6: from humanoid.envs.custom.humanoid_config import XBotLCfg, XBotLCfgPPO
     7: 
     8: 
     9: class XBotLCustomCfg(XBotLCfg):
    10:     """Custom environment config - inherits XBotLCfg, overrides command ranges only."""
    11: 
    12:     class commands(XBotLCfg.commands):
    13:         # READ-ONLY: official XBot training command distribution. Locked to keep
    14:         # the train→eval comparison fair (eval samples vx∈[-0.5,1.0] and the
    15:         # hidden high-speed env tests vx=1.5, deliberately widening beyond
    16:         # training; agents proposing algorithmic improvements should not also
    17:         # widen the training distribution to score higher on the hidden env).
    18:         # heading_command=False so the policy's third command channel is the
    19:         # raw ang_vel_yaw target, matching what the MuJoCo sim2sim eval feeds.
    20:         # Upstream's heading_command=True samples a heading target and converts
    21:         # to a corrective ang_vel internally, which produces a train/eval
    22:         # contract mismatch.
    23:         heading_command = False
    24:         class ranges:
    25:             lin_vel_x = [-0.3, 0.6]
    26:             lin_vel_y = [-0.3, 0.3]
    27:             ang_vel_yaw = [-0.3, 0.3]
    28:             heading = [-3.14, 3.14]
    29: 
    30: 
    31: class XBotLCustomCfgPPO(XBotLCfgPPO):
    32:     """Custom PPO runner config - uses the custom algorithm classes."""
    33: 
    34:     class algorithm(XBotLCfgPPO.algorithm):
    35:         # EDITABLE: tune PPO hyperparameters per algorithm variant.
    36:         # Defaults mirror XBotLCfgPPO.algorithm so existing baselines are unaffected.
    37:         learning_rate = 1.0e-5
    38:         entropy_coef = 0.001
    39:         num_learning_epochs = 2
    40:         gamma = 0.994
    41:         lam = 0.9
    42:         num_mini_batches = 4
    43: 
    44:     class runner(XBotLCfgPPO.runner):
    45:         policy_class_name = 'ActorCritic'
    46:         algorithm_class_name = 'PPO'
    47:         experiment_name = 'XBot_ppo'
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `ppo_adaptive_kl` baseline — editable region  [READ-ONLY — reference implementation]

In `humanoid-gym/humanoid/algo/ppo/actor_critic_custom.py`:

```python
Lines 36–128:
    33: import torch.nn as nn
    34: from torch.distributions import Normal
    35: 
    36: class ActorCritic(nn.Module):
    37:     def __init__(self,  num_actor_obs,
    38:                         num_critic_obs,
    39:                         num_actions,
    40:                         actor_hidden_dims=[256, 256, 256],
    41:                         critic_hidden_dims=[256, 256, 256],
    42:                         init_noise_std=1.0,
    43:                         activation = nn.ELU(),
    44:                         **kwargs):
    45:         if kwargs:
    46:             print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
    47:         super(ActorCritic, self).__init__()
    48: 
    49: 
    50:         mlp_input_dim_a = num_actor_obs
    51:         mlp_input_dim_c = num_critic_obs
    52:         # Policy
    53:         actor_layers = []
    54:         actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
    55:         actor_layers.append(activation)
    56:         for l in range(len(actor_hidden_dims)):
    57:             if l == len(actor_hidden_dims) - 1:
    58:                 actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
    59:             else:
    60:                 actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
    61:                 actor_layers.append(activation)
    62:         self.actor = nn.Sequential(*actor_layers)
    63: 
    64:         # Value function
    65:         critic_layers = []
    66:         critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
    67:         critic_layers.append(activation)
    68:         for l in range(len(critic_hidden_dims)):
    69:             if l == len(critic_hidden_dims) - 1:
    70:                 critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
    71:             else:
    72:                 critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
    73:                 critic_layers.append(activation)
    74:         self.critic = nn.Sequential(*critic_layers)
    75: 
    76:         print(f"Actor MLP: {self.actor}")
    77:         print(f"Critic MLP: {self.critic}")
    78: 
    79:         # Action noise
    80:         self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
    81:         self.distribution = None
    82:         # disable args validation for speedup
    83:         Normal.set_default_validate_args = False
    84: 
    85: 
    86:     @staticmethod
    87:     # not used at the moment
    88:     def init_weights(sequential, scales):
    89:         [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
    90:          enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
    91: 
    92: 
    93:     def reset(self, dones=None):
    94:         pass
    95: 
    96:     def forward(self):
    97:         raise NotImplementedError
    98: 
    99:     @property
   100:     def action_mean(self):
   101:         return self.distribution.mean
   102: 
   103:     @property
   104:     def action_std(self):
   105:         return self.distribution.stddev
   106: 
   107:     @property
   108:     def entropy(self):
   109:         return self.distribution.entropy().sum(dim=-1)
   110: 
   111:     def update_distribution(self, observations):
   112:         mean = self.actor(observations)
   113:         self.distribution = Normal(mean, mean*0.+self.std)
   114: 
   115:     def act(self, observations, **kwargs):
   116:         self.update_distribution(observations)
   117:         return self.distribution.sample()
   118: 
   119:     def get_actions_log_prob(self, actions):
   120:         return self.distribution.log_prob(actions).sum(dim=-1)
   121: 
   122:     def act_inference(self, observations):
   123:         actions_mean = self.actor(observations)
   124:         return actions_mean
   125: 
   126:     def evaluate(self, critic_observations, **kwargs):
   127:         value = self.critic(critic_observations)
   128:         return value
```

### `ppo_layernorm` baseline — editable region  [READ-ONLY — reference implementation]

In `humanoid-gym/humanoid/algo/ppo/actor_critic_custom.py`:

```python
Lines 36–132:
    33: import torch.nn as nn
    34: from torch.distributions import Normal
    35: 
    36: class ActorCritic(nn.Module):
    37:     def __init__(self,  num_actor_obs,
    38:                         num_critic_obs,
    39:                         num_actions,
    40:                         actor_hidden_dims=[512, 256, 128],
    41:                         critic_hidden_dims=[512, 256, 128],
    42:                         init_noise_std=1.0,
    43:                         activation = nn.ELU(),
    44:                         **kwargs):
    45:         if kwargs:
    46:             print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
    47:         super(ActorCritic, self).__init__()
    48: 
    49: 
    50:         mlp_input_dim_a = num_actor_obs
    51:         mlp_input_dim_c = num_critic_obs
    52:         # Policy with LayerNorm
    53:         actor_layers = []
    54:         actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
    55:         actor_layers.append(nn.LayerNorm(actor_hidden_dims[0]))
    56:         actor_layers.append(activation)
    57:         for l in range(len(actor_hidden_dims)):
    58:             if l == len(actor_hidden_dims) - 1:
    59:                 actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
    60:             else:
    61:                 actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
    62:                 actor_layers.append(nn.LayerNorm(actor_hidden_dims[l + 1]))
    63:                 actor_layers.append(activation)
    64:         self.actor = nn.Sequential(*actor_layers)
    65: 
    66:         # Value function with LayerNorm
    67:         critic_layers = []
    68:         critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
    69:         critic_layers.append(nn.LayerNorm(critic_hidden_dims[0]))
    70:         critic_layers.append(activation)
    71:         for l in range(len(critic_hidden_dims)):
    72:             if l == len(critic_hidden_dims) - 1:
    73:                 critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
    74:             else:
    75:                 critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
    76:                 critic_layers.append(nn.LayerNorm(critic_hidden_dims[l + 1]))
    77:                 critic_layers.append(activation)
    78:         self.critic = nn.Sequential(*critic_layers)
    79: 
    80:         print(f"Actor MLP with LayerNorm: {self.actor}")
    81:         print(f"Critic MLP with LayerNorm: {self.critic}")
    82: 
    83:         # Action noise
    84:         self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
    85:         self.distribution = None
    86:         # disable args validation for speedup
    87:         Normal.set_default_validate_args = False
    88: 
    89: 
    90:     @staticmethod
    91:     # not used at the moment
    92:     def init_weights(sequential, scales):
    93:         [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
    94:          enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
    95: 
    96: 
    97:     def reset(self, dones=None):
    98:         pass
    99: 
   100:     def forward(self):
   101:         raise NotImplementedError
   102: 
   103:     @property
   104:     def action_mean(self):
   105:         return self.distribution.mean
   106: 
   107:     @property
   108:     def action_std(self):
   109:         return self.distribution.stddev
   110: 
   111:     @property
   112:     def entropy(self):
   113:         return self.distribution.entropy().sum(dim=-1)
   114: 
   115:     def update_distribution(self, observations):
   116:         mean = self.actor(observations)
   117:         self.distribution = Normal(mean, mean*0.+self.std)
   118: 
   119:     def act(self, observations, **kwargs):
   120:         self.update_distribution(observations)
   121:         return self.distribution.sample()
   122: 
   123:     def get_actions_log_prob(self, actions):
   124:         return self.distribution.log_prob(actions).sum(dim=-1)
   125: 
   126:     def act_inference(self, observations):
   127:         actions_mean = self.actor(observations)
   128:         return actions_mean
   129: 
   130:     def evaluate(self, critic_observations, **kwargs):
   131:         value = self.critic(critic_observations)
   132:         return value
```

### `default` baseline — editable region  [READ-ONLY — reference implementation]

In `humanoid-gym/humanoid/algo/ppo/ppo_custom.py`:

```python
Lines 39–186:
    36: from .actor_critic_custom import ActorCritic
    37: from .rollout_storage_custom import RolloutStorage
    38: 
    39: class PPO:
    40:     actor_critic: ActorCritic
    41:     def __init__(self,
    42:                  actor_critic,
    43:                  num_learning_epochs=1,
    44:                  num_mini_batches=1,
    45:                  clip_param=0.2,
    46:                  gamma=0.998,
    47:                  lam=0.95,
    48:                  value_loss_coef=1.0,
    49:                  entropy_coef=0.0,
    50:                  learning_rate=1e-3,
    51:                  max_grad_norm=1.0,
    52:                  use_clipped_value_loss=True,
    53:                  schedule="fixed",
    54:                  desired_kl=0.01,
    55:                  device='cpu',
    56:                  ):
    57: 
    58:         self.device = device
    59: 
    60:         self.desired_kl = desired_kl
    61:         self.schedule = schedule
    62:         self.learning_rate = learning_rate
    63: 
    64:         # PPO components
    65:         self.actor_critic = actor_critic
    66:         self.actor_critic.to(self.device)
    67:         self.storage = None # initialized later
    68:         self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
    69:         self.transition = RolloutStorage.Transition()
    70: 
    71:         # PPO parameters
    72:         self.clip_param = clip_param
    73:         self.num_learning_epochs = num_learning_epochs
    74:         self.num_mini_batches = num_mini_batches
    75:         self.value_loss_coef = value_loss_coef
    76:         self.entropy_coef = entropy_coef
    77:         self.gamma = gamma
    78:         self.lam = lam
    79:         self.max_grad_norm = max_grad_norm
    80:         self.use_clipped_value_loss = use_clipped_value_loss
    81: 
    82:     def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
    83:         self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)
    84: 
    85:     def test_mode(self):
    86:         # Upstream calls .test() which doesn't exist on nn.Module; use .eval() to match
    87:         # PyTorch convention and the variants. Currently dead code (test_mode is never
    88:         # called in the runner), but kept correct so any future caller doesn't AttributeError.
    89:         self.actor_critic.eval()
    90: 
    91:     def train_mode(self):
    92:         self.actor_critic.train()
    93: 
    94:     def act(self, obs, critic_obs):
    95:         # Compute the actions and values
    96:         self.transition.actions = self.actor_critic.act(obs).detach()
    97:         self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
    98:         self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
    99:         self.transition.action_mean = self.actor_critic.action_mean.detach()
   100:         self.transition.action_sigma = self.actor_critic.action_std.detach()
   101:         # need to record obs and critic_obs before env.step()
   102:         self.transition.observations = obs
   103:         self.transition.critic_observations = critic_obs
   104:         return self.transition.actions
   105: 
   106:     def process_env_step(self, rewards, dones, infos):
   107:         self.transition.rewards = rewards.clone()
   108:         self.transition.dones = dones
   109:         # Bootstrapping on time outs
   110:         if 'time_outs' in infos:
   111:             self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
   112: 
   113:         # Record the transition
   114:         self.storage.add_transitions(self.transition)
   115:         self.transition.clear()
   116:         self.actor_critic.reset(dones)
   117: 
   118:     def compute_returns(self, last_critic_obs):
   119:         last_values= self.actor_critic.evaluate(last_critic_obs).detach()
   120:         self.storage.compute_returns(last_values, self.gamma, self.lam)
   121: 
   122:     def update(self):
   123:         mean_value_loss = 0
   124:         mean_surrogate_loss = 0
   125: 
   126:         generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
   127:         for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch,             old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
   128: 
   129: 
   130:                 self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
   131:                 actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
   132:                 value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
   133:                 mu_batch = self.actor_critic.action_mean
   134:                 sigma_batch = self.actor_critic.action_std
   135:                 entropy_batch = self.actor_critic.entropy
   136: 
   137:                 # KL
   138:                 if self.desired_kl != None and self.schedule == 'adaptive':
   139:                     with torch.inference_mode():
   140:                         kl = torch.sum(
   141:                             torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
   142:                         kl_mean = torch.mean(kl)
   143: 
   144:                         if kl_mean > self.desired_kl * 2.0:
   145:                             self.learning_rate = max(1e-5, self.learning_rate / 1.5)
   146:                         elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
   147:                             self.learning_rate = min(1e-2, self.learning_rate * 1.5)
   148: 
   149:                         for param_group in self.optimizer.param_groups:
   150:                             param_group['lr'] = self.learning_rate
   151: 
   152: 
   153:                 # Surrogate loss
   154:                 ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
   155:                 surrogate = -torch.squeeze(advantages_batch) * ratio
   156:                 surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
   157:                                                                                 1.0 + self.clip_param)
   158:                 surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
   159: 
   160:                 # Value function loss
   161:                 if self.use_clipped_value_loss:
   162:                     value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
   163:                                                                                                     self.clip_param)
   164:                     value_losses = (value_batch - returns_batch).pow(2)
   165:                     value_losses_clipped = (value_clipped - returns_batch).pow(2)
   166:                     value_loss = torch.max(value_losses, value_losses_clipped).mean()
   167:                 else:
   168:                     value_loss = (returns_batch - value_batch).pow(2).mean()
   169: 
   170:                 loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
   171: 
   172:                 # Gradient step
   173:                 self.optimizer.zero_grad()
   174:                 loss.backward()
   175:                 nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
   176:                 self.optimizer.step()
   177: 
   178:                 mean_value_loss += value_loss.item()
   179:                 mean_surrogate_loss += surrogate_loss.item()
   180: 
   181:         num_updates = self.num_learning_epochs * self.num_mini_batches
   182:         mean_value_loss /= num_updates
   183:         mean_surrogate_loss /= num_updates
   184:         self.storage.clear()
   185: 
   186:         return mean_value_loss, mean_surrogate_loss
```

### `ppo_adaptive_kl` baseline — editable region  [READ-ONLY — reference implementation]

In `humanoid-gym/humanoid/algo/ppo/ppo_custom.py`:

```python
Lines 39–184:
    36: from .actor_critic_custom import ActorCritic
    37: from .rollout_storage_custom import RolloutStorage
    38: 
    39: class PPO:
    40:     actor_critic: ActorCritic
    41:     def __init__(self,
    42:                  actor_critic,
    43:                  num_learning_epochs=1,
    44:                  num_mini_batches=1,
    45:                  clip_param=0.2,
    46:                  gamma=0.998,
    47:                  lam=0.95,
    48:                  value_loss_coef=1.0,
    49:                  entropy_coef=0.0,
    50:                  learning_rate=1e-3,
    51:                  max_grad_norm=1.0,
    52:                  use_clipped_value_loss=True,
    53:                  schedule="fixed",
    54:                  desired_kl=0.01,
    55:                  device='cpu',
    56:                  ):
    57: 
    58:         self.device = device
    59: 
    60:         self.desired_kl = desired_kl
    61:         self.schedule = schedule
    62:         self.learning_rate = learning_rate
    63: 
    64:         # PPO components
    65:         self.actor_critic = actor_critic
    66:         self.actor_critic.to(self.device)
    67:         self.storage = None # initialized later
    68:         self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
    69:         self.transition = RolloutStorage.Transition()
    70: 
    71:         # PPO parameters
    72:         self.clip_param = clip_param
    73:         self.num_learning_epochs = num_learning_epochs
    74:         self.num_mini_batches = num_mini_batches
    75:         self.value_loss_coef = value_loss_coef
    76:         self.entropy_coef = entropy_coef
    77:         self.gamma = gamma
    78:         self.lam = lam
    79:         self.max_grad_norm = max_grad_norm
    80:         self.use_clipped_value_loss = use_clipped_value_loss
    81: 
    82:     def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
    83:         self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)
    84: 
    85:     def test_mode(self):
    86:         self.actor_critic.eval()
    87: 
    88:     def train_mode(self):
    89:         self.actor_critic.train()
    90: 
    91:     def act(self, obs, critic_obs):
    92:         # Compute the actions and values
    93:         self.transition.actions = self.actor_critic.act(obs).detach()
    94:         self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
    95:         self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
    96:         self.transition.action_mean = self.actor_critic.action_mean.detach()
    97:         self.transition.action_sigma = self.actor_critic.action_std.detach()
    98:         # need to record obs and critic_obs before env.step()
    99:         self.transition.observations = obs
   100:         self.transition.critic_observations = critic_obs
   101:         return self.transition.actions
   102: 
   103:     def process_env_step(self, rewards, dones, infos):
   104:         self.transition.rewards = rewards.clone()
   105:         self.transition.dones = dones
   106:         # Bootstrapping on time outs
   107:         if 'time_outs' in infos:
   108:             self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
   109: 
   110:         # Record the transition
   111:         self.storage.add_transitions(self.transition)
   112:         self.transition.clear()
   113:         self.actor_critic.reset(dones)
   114: 
   115:     def compute_returns(self, last_critic_obs):
   116:         last_values = self.actor_critic.evaluate(last_critic_obs).detach()
   117:         self.storage.compute_returns(last_values, self.gamma, self.lam)
   118: 
   119:     def update(self):
   120:         mean_value_loss = 0
   121:         mean_surrogate_loss = 0
   122: 
   123:         generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
   124:         for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch,             old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
   125: 
   126: 
   127:                 self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
   128:                 actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
   129:                 value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
   130:                 mu_batch = self.actor_critic.action_mean
   131:                 sigma_batch = self.actor_critic.action_std
   132:                 entropy_batch = self.actor_critic.entropy
   133: 
   134:                 # KL divergence for adaptive learning rate (tighter thresholds: 1.5x instead of 2.0x)
   135:                 if self.desired_kl != None and self.schedule == 'adaptive':
   136:                     with torch.inference_mode():
   137:                         kl = torch.sum(
   138:                             torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
   139:                         kl_mean = torch.mean(kl)
   140: 
   141:                         # Adaptive learning rate with tighter KL thresholds for more responsive adaptation
   142:                         if kl_mean > self.desired_kl * 1.5:
   143:                             self.learning_rate = max(1e-5, self.learning_rate / 1.5)
   144:                         elif kl_mean < self.desired_kl / 1.5:
   145:                             self.learning_rate = min(1e-2, self.learning_rate * 1.5)
   146: 
   147:                         for param_group in self.optimizer.param_groups:
   148:                             param_group['lr'] = self.learning_rate
   149: 
   150:                 # Surrogate loss
   151:                 ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
   152:                 surrogate = -torch.squeeze(advantages_batch) * ratio
   153:                 surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
   154:                                                                                    1.0 + self.clip_param)
   155:                 surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
   156: 
   157:                 # Value function loss
   158:                 if self.use_clipped_value_loss:
   159:                     value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
   160:                                                                                                     self.clip_param)
   161:                     value_losses = (value_batch - returns_batch).pow(2)
   162:                     value_losses_clipped = (value_clipped - returns_batch).pow(2)
   163:                     value_loss = torch.max(value_losses, value_losses_clipped).mean()
   164:                 else:
   165:                     value_loss = (returns_batch - value_batch).pow(2).mean()
   166: 
   167:                 loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
   168: 
   169:                 # Gradient step
   170:                 self.optimizer.zero_grad()
   171:                 loss.backward()
   172:                 nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
   173:                 self.optimizer.step()
   174: 
   175:                 mean_value_loss += value_loss.item()
   176:                 mean_surrogate_loss += surrogate_loss.item()
   177: 
   178:         num_updates = self.num_learning_epochs * self.num_mini_batches
   179:         mean_value_loss /= num_updates
   180:         mean_surrogate_loss /= num_updates
   181:         self.storage.clear()
   182: 
   183:         return mean_value_loss, mean_surrogate_loss
   184: 
```

### `default` baseline — editable region  [READ-ONLY — reference implementation]

In `humanoid-gym/humanoid/algo/ppo/rollout_storage_custom.py`:

```python
Lines 34–180:
    31: 
    32: import torch
    33: 
    34: class RolloutStorage:
    35:     class Transition:
    36:         def __init__(self):
    37:             self.observations = None
    38:             self.critic_observations = None
    39:             self.actions = None
    40:             self.rewards = None
    41:             self.dones = None
    42:             self.values = None
    43:             self.actions_log_prob = None
    44:             self.action_mean = None
    45:             self.action_sigma = None
    46:             self.hidden_states = None
    47:         
    48:         def clear(self):
    49:             self.__init__()
    50: 
    51:     def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device='cpu'):
    52: 
    53:         self.device = device
    54: 
    55:         self.obs_shape = obs_shape
    56:         self.privileged_obs_shape = privileged_obs_shape
    57:         self.actions_shape = actions_shape
    58: 
    59:         # Core
    60:         self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
    61:         if privileged_obs_shape[0] is not None:
    62:             self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
    63:         else:
    64:             self.privileged_observations = None
    65:         self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
    66:         self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
    67:         self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
    68: 
    69:         # For PPO
    70:         self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
    71:         self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
    72:         self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
    73:         self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
    74:         self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
    75:         self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
    76: 
    77:         self.num_transitions_per_env = num_transitions_per_env
    78:         self.num_envs = num_envs
    79: 
    80:         # rnn
    81:         self.saved_hidden_states_a = None
    82:         self.saved_hidden_states_c = None
    83: 
    84:         self.step = 0
    85: 
    86:     def add_transitions(self, transition: Transition):
    87:         if self.step >= self.num_transitions_per_env:
    88:             raise AssertionError("Rollout buffer overflow")
    89:         self.observations[self.step].copy_(transition.observations)
    90:         if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(transition.critic_observations)
    91:         self.actions[self.step].copy_(transition.actions)
    92:         self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
    93:         self.dones[self.step].copy_(transition.dones.view(-1, 1))
    94:         self.values[self.step].copy_(transition.values)
    95:         self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
    96:         self.mu[self.step].copy_(transition.action_mean)
    97:         self.sigma[self.step].copy_(transition.action_sigma)
    98:         self._save_hidden_states(transition.hidden_states)
    99:         self.step += 1
   100: 
   101:     def _save_hidden_states(self, hidden_states):
   102:         if hidden_states is None or hidden_states==(None, None):
   103:             return
   104:         # make a tuple out of GRU hidden state sto match the LSTM format
   105:         hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
   106:         hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
   107: 
   108:         # initialize if needed 
   109:         if self.saved_hidden_states_a is None:
   110:             self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
   111:             self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
   112:         # copy the states
   113:         for i in range(len(hid_a)):
   114:             self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
   115:             self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])
   116: 
   117: 
   118:     def clear(self):
   119:         self.step = 0
   120: 
   121:     def compute_returns(self, last_values, gamma, lam):
   122:         advantage = 0
   123:         for step in reversed(range(self.num_transitions_per_env)):
   124:             if step == self.num_transitions_per_env - 1:
   125:                 next_values = last_values
   126:             else:
   127:                 next_values = self.values[step + 1]
   128:             next_is_not_terminal = 1.0 - self.dones[step].float()
   129:             delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
   130:             advantage = delta + next_is_not_terminal * gamma * lam * advantage
   131:             self.returns[step] = advantage + self.values[step]
   132: 
   133:         # Compute and normalize the advantages
   134:         self.advantages = self.returns - self.values
   135:         self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
   136: 
   137:     def get_statistics(self):
   138:         done = self.dones
   139:         done[-1] = 1
   140:         flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
   141:         done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
   142:         trajectory_lengths = (done_indices[1:] - done_indices[:-1])
   143:         return trajectory_lengths.float().mean(), self.rewards.mean()
   144: 
   145:     def mini_batch_generator(self, num_mini_batches, num_epochs=8):
   146:         batch_size = self.num_envs * self.num_transitions_per_env
   147:         mini_batch_size = batch_size // num_mini_batches
   148:         indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)
   149: 
   150:         observations = self.observations.flatten(0, 1)
   151:         if self.privileged_observations is not None:
   152:             critic_observations = self.privileged_observations.flatten(0, 1)
   153:         else:
   154:             critic_observations = observations
   155: 
   156:         actions = self.actions.flatten(0, 1)
   157:         values = self.values.flatten(0, 1)
   158:         returns = self.returns.flatten(0, 1)
   159:         old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
   160:         advantages = self.advantages.flatten(0, 1)
   161:         old_mu = self.mu.flatten(0, 1)
   162:         old_sigma = self.sigma.flatten(0, 1)
   163: 
   164:         for epoch in range(num_epochs):
   165:             for i in range(num_mini_batches):
   166: 
   167:                 start = i*mini_batch_size
   168:                 end = (i+1)*mini_batch_size
   169:                 batch_idx = indices[start:end]
   170: 
   171:                 obs_batch = observations[batch_idx]
   172:                 critic_observations_batch = critic_observations[batch_idx]
   173:                 actions_batch = actions[batch_idx]
   174:                 target_values_batch = values[batch_idx]
   175:                 returns_batch = returns[batch_idx]
   176:                 old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
   177:                 advantages_batch = advantages[batch_idx]
   178:                 old_mu_batch = old_mu[batch_idx]
   179:                 old_sigma_batch = old_sigma[batch_idx]
   180:                 yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,                        old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
