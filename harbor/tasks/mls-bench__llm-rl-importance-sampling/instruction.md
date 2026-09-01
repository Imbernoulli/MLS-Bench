# MLS-Bench: llm-rl-importance-sampling

# LLM Online RL: Importance-Sampling Granularity for Policy Optimization

## Research Question
Design a custom **importance-sampling (IS) strategy** for the clipped policy-gradient loss used in LLM online RL. The advantage estimator (GRPO), reward model, rollout setup, and KL configuration are fixed; the only variable is how the old-policy / current-policy log-probs are converted into ratios, clipped, and aggregated. The goal is improved math-reasoning accuracy and reduced gradient variance.

## Background
In PPO-style LLM RL, the per-token policy objective uses an importance ratio
```
r_{i,t} = exp(log_prob_new(y_{i,t}) − log_prob_old(y_{i,t}))
```
applied to per-token advantages. The granularity of this ratio (and of the clipping) is an open research axis:

- **Token-level IS** (vanilla PPO / GRPO; Schulman et al., PPO, 2017, arXiv:1707.06347; Shao et al., DeepSeekMath, 2024, arXiv:2402.03300). Each token has its own ratio, clipped independently. Variance can be very high for long LLM responses because per-token ratios are noisy and errors compound.
- **Sequence-level IS** — Zheng et al., "Group Sequence Policy Optimization", 2025, arXiv:2507.18071 (GSPO). Single scalar ratio per sequence `s_i = exp( mean_t (log_prob_new − log_prob_old) )`, broadcast to every token; reduces variance and stabilizes MoE RL.
- **Truncated / decoupled-clip IS** — Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale", 2025, arXiv:2503.14476. Decoupled (asymmetric) clip-low / clip-high, dynamic sampling, token-length-decoupled clipping; built on verl.
- **CISPO-style clipped IS with stop-grad** — MiniMax Team, "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention", 2025, arXiv:2506.13585. Clip the IS weight inside a stop-gradient so gradients flow through `log π` scaled by a bounded IS weight; no token's gradient is zeroed out.
- **Other variants**: dual-clip PPO, geometric-mean aggregation over groups, per-prompt normalised ratios, etc.

## What you can modify
The `compute_custom_policy_loss()` function in `verl/.../custom_policy_loss.py`. The read-only `core_algos.py` contains `compute_policy_loss_vanilla` (token-level PPO), `compute_policy_loss_gspo` (sequence-level), `compute_policy_loss_dppo_kl`, `compute_policy_loss_clip_cov`, etc., as references.

### Interface contract
```python
@register_policy_loss("custom")
def compute_custom_policy_loss(
    old_log_prob: torch.Tensor,      # (bs, response_length)
    log_prob: torch.Tensor,          # (bs, response_length)
    advantages: torch.Tensor,        # (bs, response_length)
    response_mask: torch.Tensor,     # (bs, response_length)
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
```
- `loss_agg_mode` — common values `"token-mean"`, `"seq-mean-token-mean"`.
- `config` — `ActorConfig` with `clip_ratio`, optional asymmetric `clip_ratio_low` / `clip_ratio_high` (fall back to `clip_ratio` if `None`), `clip_ratio_c` (dual-clip), and `global_batch_info` kwargs forwarded to `agg_loss`.
- Return `(pg_loss, metrics)` where `metrics` has at least `"actor/pg_clipfrac"` and `"actor/ppo_kl"` as Python floats.

Canonical aggregation pattern (from vanilla):
```python
pg_loss = agg_loss(
    loss_mat=pg_losses,
    loss_mask=response_mask,
    loss_agg_mode=loss_agg_mode,
    **config.global_batch_info,
)
```

Utilities: `verl_F.masked_mean`, `verl_F.masked_whiten`, `agg_loss`, `torch`. `assert config is not None` and read `config.clip_ratio` (do not hardcode ε). Clamp `log_prob − old_log_prob` to a safe range (e.g., `[-20, 20]`) before `exp` for numerical stability. If your strategy aggregates across the sequence (GSPO-like), use `loss_agg_mode="seq-mean-token-mean"` inside your `agg_loss` call. Apply `rollout_is_weights` multiplicatively on `pg_losses` if not `None` (see vanilla).

## Reference baselines
| Baseline | Granularity | Reference |
| --- | --- | --- |
| `token_level` | per-token ratio + per-token clip | PPO (Schulman et al., 2017) |
| `sequence_level` | sequence-mean log-ratio + sequence clip | GSPO, arXiv:2507.18071 |
| `first_k_tokens` | per-token ratio for first K=64 tokens, stop-grad after | DAPO-style truncated IS, arXiv:2503.14476 |

## Fixed Pipeline
- **Policy**: Qwen2.5-0.5B (full-parameter training), verl framework, GRPO advantage estimator.
- **Training set**: simpleRL-Zoo MATH level 3–5 (Qwen split).
- **Hyperparameters**: 100 PPO steps, 16 rollout samples per prompt, batch size 128, 1 H200 GPU.
- Advantage estimator, reward manager, model, rollout setup, optimizer, and evaluation are all fixed.

## Evaluation
Math-reasoning accuracy (`mean@1`) on **GSM8K**, **MATH-500**, and **AMC 23**; primary score is the mean across the three.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/verl/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `verl/verl/trainer/ppo/custom_policy_loss.py`
- editable lines **17–72**
- `verl/verl/trainer/main_ppo.py`
- editable lines **33–34**
- `verl/verl/workers/actor/dp_actor.py`
- editable lines **31–32**
- `verl/verl/workers/actor/megatron_actor.py`
- editable lines **41–42**




## Readable Context


### `verl/verl/trainer/ppo/custom_policy_loss.py`  [EDITABLE — lines 17–72 only]

```python
     1: # Copyright 2024 Bytedance Ltd. and/or its affiliates
     2: # Licensed under the Apache License, Version 2.0
     3: """Custom policy loss / importance-sampling strategy for verl PPO training."""
     4: 
     5: from typing import Any, Optional
     6: 
     7: import torch
     8: 
     9: import verl.utils.torch_functional as verl_F
    10: from verl.workers.config import ActorConfig
    11: from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
    12: 
    13: # =====================================================================
    14: # EDITABLE: Implement your custom importance-sampling policy loss below.
    15: # =====================================================================
    16: 
    17: 
    18: @register_policy_loss("custom")
    19: def compute_custom_policy_loss(
    20:     old_log_prob: torch.Tensor,
    21:     log_prob: torch.Tensor,
    22:     advantages: torch.Tensor,
    23:     response_mask: torch.Tensor,
    24:     loss_agg_mode: str = "token-mean",
    25:     config: Optional[ActorConfig] = None,
    26:     rollout_is_weights: torch.Tensor | None = None,
    27: ) -> tuple[torch.Tensor, dict[str, Any]]:
    28:     """Compute the clipped policy objective for LLM online RL.
    29: 
    30:     This function is called by the verl training loop.  The core design
    31:     axis is *importance-sampling granularity*: how the ratio
    32:         r = exp(log_prob - old_log_prob)
    33:     is formed and clipped (per-token, per-sequence, truncated to a
    34:     prefix, etc.).  See GSPO (Zheng et al., 2025, arXiv:2507.18071),
    35:     DAPO (arXiv:2503.14476), and CISPO (MiniMax M1, arXiv:2506.13585)
    36:     for references.
    37: 
    38:     Args:
    39:         old_log_prob: (bs, response_length)
    40:             Log-probabilities of each token under the old (rollout) policy.
    41:         log_prob: (bs, response_length)
    42:             Log-probabilities of each token under the current policy.
    43:         advantages: (bs, response_length)
    44:             Per-token advantage estimates.
    45:         response_mask: (bs, response_length)
    46:             Binary mask (1 = valid response token).
    47:         loss_agg_mode: Aggregation mode forwarded to ``agg_loss``.
    48:             Typical values: "token-mean", "seq-mean-token-mean".
    49:         config: ``ActorConfig`` with fields such as ``clip_ratio``,
    50:             ``clip_ratio_low``, ``clip_ratio_high``, and
    51:             ``global_batch_info`` (passed as kwargs to ``agg_loss``).
    52:             ``config.get("name", default)`` is supported for optional
    53:             fields like ``clip_ratio_c``.
    54:         rollout_is_weights: Optional per-token rollout-correction weights.
    55: 
    56:     Returns:
    57:         pg_loss: scalar policy-gradient loss tensor.
    58:         metrics: dict with at least ``actor/pg_clipfrac`` and
    59:             ``actor/ppo_kl`` as Python floats.
    60: 
    61:     Typical call to aggregate:
    62:         pg_loss = agg_loss(
    63:             loss_mat=pg_losses,
    64:             loss_mask=response_mask,
    65:             loss_agg_mode=loss_agg_mode,
    66:             **config.global_batch_info,
    67:         )
    68:     """
    69:     raise NotImplementedError(
    70:         "Implement your custom importance-sampling policy loss here. "
    71:         "See core_algos.py for reference (compute_policy_loss_vanilla / gspo)."
    72:     )
```

### `verl/verl/trainer/main_ppo.py`  [EDITABLE — lines 33–34 only]

```python
     1: # Copyright 2024 Bytedance Ltd. and/or its affiliates
     2: #
     3: # Licensed under the Apache License, Version 2.0 (the "License");
     4: # you may not use this file except in compliance with the License.
     5: # You may obtain a copy of the License at
     6: #
     7: #     http://www.apache.org/licenses/LICENSE-2.0
     8: #
     9: # Unless required by applicable law or agreed to in writing, software
    10: # distributed under the License is distributed on an "AS IS" BASIS,
    11: # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    12: # See the License for the specific language governing permissions and
    13: # limitations under the License.
    14: """
    15: Note that we don't combine the main with ray_trainer as ray_trainer is used by other mpain.
    16: """
    17: 
    18: import os
    19: import socket
    20: 
    21: import hydra
    22: import ray
    23: from omegaconf import OmegaConf
    24: 
    25: from verl.experimental.dataset.sampler import AbstractSampler
    26: from verl.experimental.reward_loop import migrate_legacy_reward_impl
    27: from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    28: from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    29: from verl.trainer.ppo.utils import need_critic, need_reference_policy
    30: from verl.utils.config import validate_config
    31: from verl.utils.device import auto_set_device, is_cuda_available
    32: from verl.utils.import_utils import load_extern_object
    33: import verl.trainer.ppo.custom_policy_loss  # noqa: F401  register custom policy loss
    34: 
    35: 
    36: @hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
    37: def main(config):
    38:     """Main entry point for PPO training with Hydra configuration management.
    39: 
    40:     Args:
    41:         config: Hydra configuration dictionary containing training parameters.
    42:     """
    43:     # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    44:     auto_set_device(config)
    45:     config = migrate_legacy_reward_impl(config)
    46:     run_ppo(config)
    47: 
    48: 
    49: # Define a function to run the PPO-like training process
    50: def run_ppo(config, task_runner_class=None) -> None:
    51:     """Initialize Ray cluster and run distributed PPO training process.
    52: 
    53:     Args:
    54:         config: Training configuration object containing all necessary parameters
    55:                 for distributed PPO training including Ray initialization settings,
    56:                 model paths, and training hyperparameters.
    57:         task_runner_class: For recipe to change TaskRunner.
    58:     """
    59:     # Check if Ray is not initialized
    60:     if not ray.is_initialized():
    61:         # Initialize Ray with a local cluster configuration
    62:         # Set environment variables in the runtime environment to control tokenizer parallelism,
    63:         # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
    64:         # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
    65:         default_runtime_env = get_ppo_ray_runtime_env()
    66:         ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
    67:         runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
    68: 
    69:         if config.transfer_queue.enable:
    70:             # Add runtime environment variables for transfer queue
    71:             runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
    72:             runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
    73:             runtime_env_kwargs["env_vars"] = runtime_env_vars
    74: 
    75:         runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
    76:         ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
    77:         print(f"ray init kwargs: {ray_init_kwargs}")
    78:         ray.init(**OmegaConf.to_container(ray_init_kwargs))
    79: 
    80:     if task_runner_class is None:
    81:         task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # please make sure main_task is not scheduled on head
    82: 
    83:     # Create a remote instance of the TaskRunner class, and
    84:     # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    85:     if (
    86:         is_cuda_available
    87:         and config.global_profiler.tool == "nsys"
    88:         and config.global_profiler.get("steps") is not None
    89:         and len(config.global_profiler.get("steps", [])) > 0
    90:     ):
    91:         from verl.utils.import_utils import is_nvtx_available
    92: 
    93:         assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
    94:         nsight_options = OmegaConf.to_container(
    95:             config.global_profiler.global_tool_config.nsys.controller_nsight_options
    96:         )
    97:         runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    98:     else:
    99:         runner = task_runner_class.remote()
   100:     ray.get(runner.run.remote(config))
   101: 
   102:     # [Optional] get the path of the timeline trace file from the configuration, default to None
   103:     # This file is used for performance analysis
   104:     timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
   105:     if timeline_json_file:
   106:         ray.timeline(filename=timeline_json_file)
   107: 
   108: 
   109: class TaskRunner:
   110:     """Ray remote class for executing distributed PPO training tasks.
   111: 
   112:     This class encapsulates the main training logic and runs as a Ray remote actor
   113:     to enable distributed execution across multiple nodes and GPUs.
   114: 
   115:     Attributes:
   116:         role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
   117:         mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
   118:     """
   119: 
   120:     def __init__(self):
   121:         self.role_worker_mapping = {}
   122:         self.mapping = {}
   123: 
   124:     def add_actor_rollout_worker(self, config):
   125:         """Add actor rollout worker based on the actor strategy."""
   126:         from verl.single_controller.ray import RayWorkerGroup
   127:         from verl.trainer.ppo.ray_trainer import Role
   128: 
   129:         use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
   130: 
   131:         # use new model engine implementation
   132:         if use_legacy_worker_impl == "disable":
   133:             from verl.workers.engine_workers import ActorRolloutRefWorker
   134: 
   135:             actor_rollout_cls = ActorRolloutRefWorker
   136:             ray_worker_group_cls = RayWorkerGroup
   137: 
   138:             lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
   139:             if lora_rank <= 0:
   140:                 lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
   141:             ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
   142:             # NOTE: In new model engine, ref policy and actor rollout are in same ActorRolloutRefWorker,
   143:             # while in legacy model engine, ref policy is in a separate ActorRolloutRefWorker.
   144:             if need_reference_policy(config) and not ref_in_actor:
   145:                 role = Role.ActorRolloutRef
   146:             else:
   147:                 role = Role.ActorRollout
   148:             self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
   149:             self.mapping[role] = "global_pool"
   150:             return actor_rollout_cls, ray_worker_group_cls
   151: 
   152:         # Note: sync mode validation is now handled in RolloutConfig.__post_init__
   153:         # Always use async worker since sync mode is deprecated and rejected
   154:         if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
   155:             from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
   156: 
   157:             actor_rollout_cls = AsyncActorRolloutRefWorker
   158:             ray_worker_group_cls = RayWorkerGroup
   159: 
   160:         elif config.actor_rollout_ref.actor.strategy == "megatron":
   161:             from verl.workers.megatron_workers import AsyncActorRolloutRefWorker
   162: 
   163:             actor_rollout_cls = AsyncActorRolloutRefWorker
   164:             ray_worker_group_cls = RayWorkerGroup
   165: 
   166:         elif (
   167:             config.actor_rollout_ref.actor.strategy == "veomni"
   168:             or config.actor_rollout_ref.actor.strategy == "torchtitan"
   169:         ):
   170:             raise NotImplementedError(
   171:                 f"{config.actor_rollout_ref.actor.strategy} does not support legacy worker implementation"
   172:             )
   173: 
   174:         else:
   175:             raise NotImplementedError
   176: 
   177:         self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
   178:         self.mapping[Role.ActorRollout] = "global_pool"
   179:         return actor_rollout_cls, ray_worker_group_cls
   180: 
   181:     def add_critic_worker(self, config):
   182:         """Add critic worker to role mapping."""
   183:         use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
   184:         if config.critic.strategy in {"fsdp", "fsdp2"}:
   185:             if use_legacy_worker_impl in ["auto", "enable"]:
   186:                 from verl.workers.fsdp_workers import CriticWorker
   187:             elif use_legacy_worker_impl == "disable":
   188:                 # we don't need to specialize critic worker. Just use TrainingWorker
   189:                 from verl.workers.engine_workers import TrainingWorker
   190: 
   191:                 CriticWorker = TrainingWorker
   192:                 print("Using new worker implementation")
   193:             else:
   194:                 raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")
   195: 
   196:         elif config.critic.strategy == "megatron":
   197:             # TODO: switch this to TrainingWorker as well
   198:             from verl.workers.megatron_workers import CriticWorker
   199: 
   200:         elif config.critic.strategy == "veomni" or config.critic.strategy == "torchtitan":
   201:             if use_legacy_worker_impl == "disable":
   202:                 from verl.workers.engine_workers import TrainingWorker
   203: 
   204:                 CriticWorker = TrainingWorker
   205:                 print(f"Using new worker implementation for {config.critic.strategy}")
   206:             else:
   207:                 raise ValueError(
   208:                     f"Invalid use_legacy_worker_impl for {config.critic.strategy}: {use_legacy_worker_impl}"
   209:                 )
   210: 
   211:         else:
   212:             raise NotImplementedError
   213: 
   214:         from verl.trainer.ppo.ray_trainer import Role
   215: 
   216:         self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
   217:         self.mapping[Role.Critic] = "global_pool"
   218: 
   219:     def init_resource_pool_mgr(self, config):
   220:         """Initialize resource pool manager."""
   221: 
   222:         global_pool_id = "global_pool"
   223:         resource_pool_spec = {
   224:             global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
   225:         }
   226: 
   227:         if config.reward.reward_model.enable_resource_pool:
   228:             if config.reward.reward_model.n_gpus_per_node <= 0:
   229:                 raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
   230:             if config.reward.reward_model.nnodes <= 0:
   231:                 raise ValueError("config.reward.reward_model.nnodes must be greater than 0")
   232: 
   233:             reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
   234:             resource_pool_spec["reward_pool"] = reward_pool
   235:         else:
   236:             config.reward.reward_model.nnodes = config.trainer.nnodes
   237:             config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
   238: 
   239:         from verl.trainer.ppo.ray_trainer import ResourcePoolManager
   240: 
   241:         resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
   242:         return resource_pool_manager
   243: 
   244:     def add_reward_model_resource_pool(self, config):
   245:         """Add reward model worker if enabled."""
   246:         from verl.trainer.ppo.ray_trainer import Role
   247: 
   248:         if config.reward.reward_model.enable:
   249:             # we do not use reward model workers, so we only register reward model in resource pool
   250:             # without continue to register reward model worker in role mapping
   251:             if config.reward.reward_model.enable_resource_pool:
   252:                 self.mapping[Role.RewardModel] = "reward_pool"
   253:             else:
   254:                 self.mapping[Role.RewardModel] = "global_pool"
   255: 
   256:     def add_ref_policy_worker(self, config, ref_policy_cls):
   257:         """Add reference policy worker if KL loss or KL reward is used."""
   258:         from verl.trainer.ppo.ray_trainer import Role
   259: 
   260:         # Ref policy has been fused into ActorRolloutRefWorker in new model engine,
   261:         # we don't need to add a separate ref policy worker group.
   262:         use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
   263:         if use_legacy_worker_impl == "disable":
   264:             return
   265: 
   266:         if need_reference_policy(config):
   267:             self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
   268:             self.mapping[Role.RefPolicy] = "global_pool"
   269: 
   270:     def run(self, config):
   271:         """Execute the main PPO training workflow.
   272: 
   273:         This method sets up the distributed training environment, initializes
   274:         workers, datasets, and reward functions, then starts the training process.
   275: 
   276:         Args:
   277:             config: Training configuration object containing all parameters needed
   278:                    for setting up and running the PPO training process.
   279:         """
   280:         # Print the initial configuration. `resolve=True` will evaluate symbolic values.
   281:         from pprint import pprint
   282: 
   283:         from omegaconf import OmegaConf
   284: 
   285:         from verl.utils.fs import copy_to_local
   286: 
   287:         print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
   288:         pprint(OmegaConf.to_container(config, resolve=True))
   289:         OmegaConf.resolve(config)
   290: 
   291:         actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
   292:         self.add_critic_worker(config)
   293: 
   294:         self.add_reward_model_resource_pool(config)
   295: 
   296:         # Add a reference policy worker if KL loss or KL reward is used.
   297:         self.add_ref_policy_worker(config, actor_rollout_cls)
   298: 
   299:         # validate config
   300:         validate_config(
   301:             config=config,
   302:             use_reference_policy=need_reference_policy(config),
   303:             use_critic=need_critic(config),
   304:         )
   305: 
   306:         # Download the checkpoint from HDFS to the local machine.
   307:         # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
   308:         local_path = copy_to_local(
   309:             config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
   310:         )
   311: 
   312:         # Instantiate the tokenizer and processor.
   313:         from verl.utils import hf_processor, hf_tokenizer
   314: 
   315:         trust_remote_code = config.data.get("trust_remote_code", False)
   316:         tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
   317:         # Used for multimodal LLM, could be None
   318:         processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
   319: 
   320:         resource_pool_manager = self.init_resource_pool_mgr(config)
   321: 
   322:         from verl.utils.dataset.rl_dataset import collate_fn
   323: 
   324:         # Create training and validation datasets.
   325:         train_dataset = create_rl_dataset(
   326:             config.data.train_files,
   327:             config.data,
   328:             tokenizer,
   329:             processor,
   330:             is_train=True,
   331:             max_samples=config.data.get("train_max_samples", -1),
   332:         )
   333:         val_dataset = create_rl_dataset(
   334:             config.data.val_files,
   335:             config.data,
   336:             tokenizer,
   337:             processor,
   338:             is_train=False,
   339:             max_samples=config.data.get("val_max_samples", -1),
   340:         )
   341:         train_sampler = create_rl_sampler(config.data, train_dataset)
   342: 
   343:         # Initialize the PPO trainer.
   344:         trainer = RayPPOTrainer(
   345:             config=config,
   346:             tokenizer=tokenizer,
   347:             processor=processor,
   348:             role_worker_mapping=self.role_worker_mapping,
   349:             resource_pool_manager=resource_pool_manager,
   350:             ray_worker_group_cls=ray_worker_group_cls,
   351:             train_dataset=train_dataset,
   352:             val_dataset=val_dataset,
   353:             collate_fn=collate_fn,
   354:             train_sampler=train_sampler,
   355:         )
   356:         # Initialize the workers of the trainer.
   357:         trainer.init_workers()
   358: 
   359:         # Start the training process.
   360:         trainer.fit()
   361: 
   362: 
   363: def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
   364:     """Create a dataset.
   365: 
   366:     Arguments:
   367:         data_paths: List of paths to data files.
   368:         data_config: The data config.
   369:         tokenizer (Tokenizer): The tokenizer.
   370:         processor (Processor): The processor.
   371: 
   372:     Returns:
   373:         dataset (Dataset): The dataset.
   374:     """
   375: 
   376:     from verl.utils.dataset.rl_dataset import get_dataset_class
   377: 
   378:     # Get the dataset class
   379:     dataset_cls = get_dataset_class(data_config)
   380: 
   381:     # Instantiate the dataset using the determined dataset class
   382:     dataset = dataset_cls(
   383:         data_files=data_paths,
   384:         tokenizer=tokenizer,
   385:         processor=processor,
   386:         config=data_config,
   387:         max_samples=max_samples,
   388:     )
   389: 
   390:     return dataset
   391: 
   392: 
   393: def create_rl_sampler(data_config, dataset):
   394:     """Create a sampler for the dataset.
   395: 
   396:     Arguments:
   397:         data_config: The data config.
   398:         dataset (Dataset): The dataset.
   399: 
   400:     Returns:
   401:         sampler (Sampler): The sampler.
   402:     """
   403:     import torch
   404:     from torch.utils.data import SequentialSampler
   405: 
   406:     # torch.utils.data.RandomSampler could not recover properly
   407:     from torchdata.stateful_dataloader.sampler import RandomSampler
   408: 
   409:     if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
   410:         curriculum_class = load_extern_object(
   411:             data_config.sampler.class_path,
   412:             data_config.sampler.class_name,
   413:         )
   414:         sampler = curriculum_class(
   415:             data_source=dataset,
   416:             data_config=data_config,
   417:         )
   418:         assert isinstance(sampler, AbstractSampler)
   419:         assert data_config.get("dataloader_num_workers", 8) == 0, (
   420:             "If using curriculum, num_workers must be 0 to prevent data caching. "
   421:             "If the dataloader caches data before the batch is done the "
   422:             "curriculum sampler won't have the opportunity to reorder it. "
   423:         )
   424: 
   425:     # Use a sampler to facilitate checkpoint resumption.
   426:     # If shuffling is enabled in the data configuration, create a random sampler.
   427:     elif data_config.shuffle:
   428:         train_dataloader_generator = torch.Generator()
   429:         seed = data_config.get("seed")
   430:         if seed is not None:
   431:             train_dataloader_generator.manual_seed(seed)
   432:         sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
   433:     else:
   434:         # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
   435:         sampler = SequentialSampler(data_source=dataset)
   436: 
   437:     return sampler
   438: 
   439: 
   440: if __name__ == "__main__":
   441:     main()
```

### `verl/verl/workers/actor/dp_actor.py`  [EDITABLE — lines 31–32 only]

```python
     1: # Copyright 2024 Bytedance Ltd. and/or its affiliates
     2: # Copyright 2023-2024 SGLang Team
     3: # Copyright 2025 ModelBest Inc. and/or its affiliates
     4: #
     5: # Licensed under the Apache License, Version 2.0 (the "License");
     6: # you may not use this file except in compliance with the License.
     7: # You may obtain a copy of the License at
     8: #
     9: #     http://www.apache.org/licenses/LICENSE-2.0
    10: #
    11: # Unless required by applicable law or agreed to in writing, software
    12: # distributed under the License is distributed on an "AS IS" BASIS,
    13: # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    14: # See the License for the specific language governing permissions and
    15: # limitations under the License.
    16: """
    17: Single Process Actor
    18: """
    19: 
    20: import logging
    21: import os
    22: 
    23: import torch
    24: from torch import nn
    25: from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    26: from torch.distributed.tensor import DTensor
    27: 
    28: import verl.utils.torch_functional as verl_F
    29: from verl import DataProto
    30: from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
    31: import verl.trainer.ppo.custom_policy_loss  # noqa: F401  register custom policy loss
    32: from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
    33: from verl.utils.device import get_device_id, get_device_name
    34: from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
    35: from verl.utils.profiler import GPUMemoryLogger
    36: from verl.utils.py_functional import append_to_dict
    37: from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
    38: from verl.utils.torch_dtypes import PrecisionType
    39: from verl.utils.torch_functional import logprobs_from_logits
    40: from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
    41: from verl.workers.actor import BasePPOActor
    42: from verl.workers.config import ActorConfig
    43: 
    44: __all__ = ["DataParallelPPOActor"]
    45: 
    46: logger = logging.getLogger(__file__)
    47: logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
    48: 
    49: 
    50: class DataParallelPPOActor(BasePPOActor):
    51:     """FSDP DataParallel PPO Actor or Ref worker
    52: 
    53:     Args:
    54:         config (ActorConfig): Actor config
    55:         actor_module (nn.Module): Actor or ref module
    56:         actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    57:     """
    58: 
    59:     def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
    60:         """When optimizer is None, it is Reference Policy"""
    61:         super().__init__(config)
    62:         self.actor_module = actor_module
    63:         self.actor_optimizer = actor_optimizer
    64:         role = "Ref" if actor_optimizer is None else "Actor"
    65: 
    66:         self.use_remove_padding = self.config.get("use_remove_padding", False)
    67:         if torch.distributed.get_rank() == 0:
    68:             print(f"{role} use_remove_padding={self.use_remove_padding}")
    69:         self.use_fused_kernels = self.config.get("use_fused_kernels", False)
    70:         if torch.distributed.get_rank() == 0:
    71:             print(f"{role} use_fused_kernels={self.use_fused_kernels}")
    72: 
    73:         self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
    74:         self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
    75: 
    76:         self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)
    77: 
    78:         self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
    79:         if torch.distributed.get_rank() == 0:
    80:             print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")
    81: 
    82:         if self.config.entropy_from_logits_with_chunking:
    83:             entropy_from_logits = verl_F.entropy_from_logits_with_chunking
    84:         else:
    85:             entropy_from_logits = verl_F.entropy_from_logits
    86: 
    87:         self.compute_entropy_from_logits = (
    88:             torch.compile(entropy_from_logits, dynamic=True)
    89:             if self.config.get("use_torch_compile", True)  # use torch compile by default
    90:             else entropy_from_logits
    91:         )
    92:         self.device_name = get_device_name()
    93:         self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
    94:         if self.param_dtype == torch.float16:
    95:             from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    96: 
    97:             self.scaler = ShardedGradScaler(growth_interval=400)
    98:         else:
    99:             self.scaler = None
   100: 
   101:         # Sum of squared probabilities computation (for optimal_token_baseline)
   102:         # Only initialize if calculate_sum_pi_squared config is enabled
   103:         if self.config.get("calculate_sum_pi_squared", False):
   104:             self.calculate_sum_pi_squared_from_logits = (
   105:                 torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
   106:                 if self.config.get("use_torch_compile", True)
   107:                 else verl_F.calculate_sum_pi_squared_from_logits
   108:             )
   109:             assert not (self.use_fused_kernels or self.use_prefix_grouper), (
   110:                 "calculate_sum_pi_squared is not supported with "
   111:                 f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
   112:             )
   113: 
   114:     def _forward_micro_batch(
   115:         self, micro_batch: dict[str, torch.Tensor], temperature: float, calculate_entropy: bool = False
   116:     ) -> dict[str, torch.Tensor]:
   117:         """
   118:         Returns:
   119:             dict[str, torch.Tensor]:
   120:                 log_probs: (bs, response_len)
   121:                 if calculate_entropy is True:
   122:                     entropys: (bs, response_len)
   123:                 if calculate_sum_pi_squared is False:
   124:                     sum_pi_squared: (bs, response_len)
   125:         """
   126:         calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
   127:         sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
   128:         # PrefixGrouper path for shared-prefix optimization
   129:         if self.use_prefix_grouper:
   130:             can_use_pg = (
   131:                 not self.use_remove_padding
   132:                 and not self.use_ulysses_sp
   133:                 and not self.use_fused_kernels
   134:                 and not self.use_dynamic_bsz
   135:             )
   136:             if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
   137:                 from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper
   138: 
   139:                 return forward_micro_batch_with_prefix_grouper(
   140:                     micro_batch=micro_batch,
   141:                     model=self.actor_module,
   142:                     temperature=temperature,
   143:                     calculate_entropy=calculate_entropy,
   144:                     device_name=self.device_name,
   145:                     param_dtype=self.param_dtype,
   146:                     use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
   147:                 )
   148: 
   149:         response_length = micro_batch["responses"].size(-1)
   150:         multi_modal_inputs = {}
   151:         if "multi_modal_inputs" in micro_batch.keys():
   152:             from verl.utils.model import extract_multi_modal_inputs
   153: 
   154:             multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])
   155: 
   156:         with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
   157:             input_ids = micro_batch["input_ids"]
   158:             batch_size, seqlen = input_ids.shape
   159:             attention_mask = micro_batch["attention_mask"]
   160:             position_ids = micro_batch["position_ids"]
   161:             entropy = None
   162:             if position_ids.dim() == 3:  # qwen2vl mrope
   163:                 position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)
   164: 
   165:             if self.use_remove_padding:
   166:                 input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
   167:                     input_ids.unsqueeze(-1), attention_mask
   168:                 )  # input_ids_rmpad (total_nnz, ...)
   169:                 input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
   170: 
   171:                 # unpad the position_ids to align the rotary
   172:                 if position_ids.dim() == 3:
   173:                     position_ids_rmpad = (
   174:                         index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
   175:                         .transpose(0, 1)
   176:                         .unsqueeze(1)
   177:                     )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
   178:                 else:
   179:                     position_ids_rmpad = index_first_axis(
   180:                         rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
   181:                     ).transpose(0, 1)
   182: 
   183:                 is_mask_all_zero = attention_mask.sum() == 0
   184:                 if is_mask_all_zero:
   185:                     input_ids_rmpad = torch.zeros(
   186:                         (1, self.ulysses_sequence_parallel_size),
   187:                         device=input_ids.device,
   188:                         dtype=input_ids.dtype,
   189:                     )
   190:                     if position_ids.dim() == 3:
   191:                         position_ids_rmpad = torch.zeros(
   192:                             (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
   193:                             device=position_ids.device,
   194:                             dtype=position_ids.dtype,
   195:                         )
   196:                     else:
   197:                         position_ids_rmpad = torch.zeros(
   198:                             (1, self.ulysses_sequence_parallel_size),
   199:                             device=position_ids.device,
   200:                             dtype=position_ids.dtype,
   201:                         )
   202: 
   203:                 if "image_bound" in multi_modal_inputs:
   204:                     from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo
   205: 
   206:                     multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
   207:                         input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
   208:                     )
   209: 
   210:                 # for compute the log_prob
   211:                 input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
   212: 
   213:                 # pad and slice the inputs if sp > 1
   214:                 if self.use_ulysses_sp:
   215:                     is_vlm_model = hasattr(
   216:                         getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
   217:                     )
   218:                     if is_vlm_model:
   219:                         # vlm model's inputs will be sliced after embedding
   220:                         input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
   221:                             input_ids_rmpad,
   222:                             position_ids_rmpad=position_ids_rmpad,
   223:                             sp_size=self.ulysses_sequence_parallel_size,
   224:                         )
   225:                     else:
   226:                         input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
   227:                             input_ids_rmpad,
   228:                             position_ids_rmpad=position_ids_rmpad,
   229:                             sp_size=self.ulysses_sequence_parallel_size,
   230:                         )
   231:                     input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
   232:                         input_ids_rmpad_rolled,
   233:                         position_ids_rmpad=None,
   234:                         sp_size=self.ulysses_sequence_parallel_size,
   235:                     )
   236: 
   237:                 input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
   238: 
   239:                 # only pass input_ids and position_ids to enable flash_attn_varlen
   240:                 extra_args = {}
   241:                 if self.use_fused_kernels:
   242:                     extra_args["temperature"] = temperature
   243:                     extra_args["return_dict"] = True
   244: 
   245:                 output = self.actor_module(
   246:                     input_ids=input_ids_rmpad,
   247:                     attention_mask=None,
   248:                     position_ids=position_ids_rmpad,
   249:                     **multi_modal_inputs,
   250:                     use_cache=False,
   251:                     **extra_args,
   252:                 )  # prevent model thinks we are generating
   253: 
   254:                 if self.use_fused_kernels:
   255:                     log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
   256:                     entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
   257: 
   258:                 else:
   259:                     logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
   260:                     logits_rmpad.div_(temperature)
   261: 
   262:                     # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
   263:                     inplace_backward = True
   264:                     if calculate_entropy:
   265:                         inplace_backward = False
   266:                     log_probs = logprobs_from_logits(
   267:                         logits=logits_rmpad,
   268:                         labels=input_ids_rmpad_rolled,
   269:                         inplace_backward=inplace_backward,
   270:                     )
   271: 
   272:                     # compute entropy
   273:                     if calculate_entropy:
   274:                         # ((total_nnz / sp) + pad)
   275:                         entropy_rmpad = (
   276:                             self.compute_entropy_from_logits(logits_rmpad)
   277:                             if not self.config.entropy_checkpointing
   278:                             else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
   279:                         )
   280: 
   281:                     # Compute sum_pi_squared if requested (for optimal_token_baseline)
   282:                     if calculate_sum_pi_squared:
   283:                         sum_pi_squared_rmpad = (
   284:                             self.calculate_sum_pi_squared_from_logits(logits_rmpad)
   285:                             if not sum_pi_squared_checkpointing
   286:                             else torch.utils.checkpoint.checkpoint(
   287:                                 self.calculate_sum_pi_squared_from_logits, logits_rmpad
   288:                             )
   289:                         )
   290: 
   291:                 # gather log_prob if sp > 1
   292:                 if self.use_ulysses_sp:
   293:                     # gather and unpad for the ulysses sp
   294:                     log_probs = gather_outputs_and_unpad(
   295:                         log_probs,
   296:                         gather_dim=0,
   297:                         unpad_dim=0,
   298:                         padding_size=pad_size,
   299:                     )
   300:                     if calculate_entropy:
   301:                         entropy_rmpad = gather_outputs_and_unpad(
   302:                             entropy_rmpad,
   303:                             gather_dim=0,
   304:                             unpad_dim=0,
   305:                             padding_size=pad_size,
   306:                         )
   307:                     if calculate_sum_pi_squared:
   308:                         sum_pi_squared_rmpad = gather_outputs_and_unpad(
   309:                             sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
   310:                         )
   311: 
   312:                 if is_mask_all_zero:
   313:                     log_probs = log_probs[:0]
   314:                     if calculate_entropy:
   315:                         entropy_rmpad = entropy_rmpad[:0]
   316: 
   317:                 # pad back to (bsz, seqlen)
   318:                 if calculate_entropy:
   319:                     full_entropy = pad_input(
   320:                         hidden_states=entropy_rmpad.unsqueeze(-1),
   321:                         indices=indices,
   322:                         batch=batch_size,
   323:                         seqlen=seqlen,
   324:                     )
   325:                 if calculate_sum_pi_squared:
   326:                     full_sum_pi_squared = pad_input(
   327:                         hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
   328:                         indices=indices,
   329:                         batch=batch_size,
   330:                         seqlen=seqlen,
   331:                     )
   332:                 full_log_probs = pad_input(
   333:                     hidden_states=log_probs.unsqueeze(-1),
   334:                     indices=indices,
   335:                     batch=batch_size,
   336:                     seqlen=seqlen,
   337:                 )
   338: 
   339:                 # only return response part:
   340:                 if calculate_entropy:
   341:                     entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
   342:                 if calculate_sum_pi_squared:
   343:                     # (bsz, response_length)
   344:                     sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
   345:                 log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
   346: 
   347:             else:  # not using rmpad and no ulysses sp
   348:                 extra_args = {}
   349:                 if self.use_fused_kernels:
   350:                     extra_args["temperature"] = temperature
   351:                     extra_args["return_dict"] = True
   352: 
   353:                 output = self.actor_module(
   354:                     input_ids=input_ids,
   355:                     attention_mask=attention_mask,
   356:                     position_ids=position_ids,
   357:                     **multi_modal_inputs,
   358:                     use_cache=False,
   359:                     **extra_args,
   360:                 )  # prevent model thinks we are generating
   361: 
   362:                 if self.use_fused_kernels:
   363:                     log_probs = output.log_probs[:, -response_length - 1 : -1]
   364:                     entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)
   365: 
   366:                 else:
   367:                     logits = output.logits
   368: 
   369:                     logits.div_(temperature)
   370:                     logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
   371:                     log_probs = logprobs_from_logits(logits, micro_batch["responses"])
   372:                     if calculate_entropy:
   373:                         if not self.config.entropy_checkpointing:
   374:                             entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
   375:                         else:
   376:                             entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
   377:                     # Compute sum_pi_squared if requested (for optimal_token_baseline)
   378:                     if calculate_sum_pi_squared:
   379:                         sum_pi_squared = (
   380:                             self.calculate_sum_pi_squared_from_logits(logits)
   381:                             if not sum_pi_squared_checkpointing
   382:                             else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
   383:                         )
   384: 
   385:             outputs = {"log_probs": log_probs}
   386:             if calculate_entropy:
   387:                 outputs["entropys"] = entropy
   388:             if calculate_sum_pi_squared:
   389:                 outputs["sum_pi_squared"] = sum_pi_squared
   390:             return outputs
   391: 
   392:     def _optimizer_step(self):
   393:         assert self.config.grad_clip is not None
   394:         if self.scaler is not None:
   395:             self.scaler.unscale_(self.actor_optimizer)
   396:         if isinstance(self.actor_module, FSDP):
   397:             grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
   398:         elif isinstance(self.actor_module, FSDPModule):
   399:             grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
   400:         else:
   401:             grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
   402: 
   403:         if isinstance(grad_norm, DTensor):
   404:             grad_norm = grad_norm.full_tensor()
   405: 
   406:         # if grad_norm is not finite, skip the update
   407:         if self.scaler is not None:
   408:             self.scaler.step(self.actor_optimizer)
   409:             self.scaler.update()
   410:         else:
   411:             if not torch.isfinite(grad_norm):
   412:                 print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
   413:                 self.actor_optimizer.zero_grad()
   414:             else:
   415:                 self.actor_optimizer.step()
   416: 
   417:         # Clear cached weight scales for QAT (weights changed)
   418:         if getattr(self.actor_module, "_qat_fuse_enabled", False):
   419:             from verl.utils.qat import invalidate_all_scales
   420: 
   421:             invalidate_all_scales(self.actor_module)
   422: 
   423:         return grad_norm
   424: 
   425:     @GPUMemoryLogger(role="dp actor", logger=logger)
   426:     def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
   427:         """Compute the log probability of the responses given input_ids, attention_mask and position_ids
   428: 
   429:         Args:
   430:             data (DataProto): a DataProto containing keys
   431: 
   432:                 ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
   433:                 concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.
   434: 
   435:                 ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.
   436: 
   437:                 ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.
   438: 
   439:                 ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.
   440: 
   441:         Returns:
   442:             dict[str, torch.Tensor]: a dict containing keys
   443:                 - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
   444:                 - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
   445:                 - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
   446:         """
   447:         calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
   448: 
   449:         # set to eval
   450:         self.actor_module.eval()
   451: 
   452:         micro_batch_size = data.meta_info["micro_batch_size"]
   453:         temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
   454:         use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
   455:         pad_token_id = data.meta_info.get("pad_token_id", 0)
   456:         has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
   457: 
   458:         select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
   459:         non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
   460:         if self.use_prefix_grouper:
   461:             select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
   462:             if "uid" in data.non_tensor_batch:
   463:                 non_tensor_select_keys.append("uid")
   464: 
   465:         data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
   466: 
   467:         if use_dynamic_bsz:
   468:             max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
   469:             # verl #2490: pass DP group (=WORLD; ulysses SP=1) so dynamic_bsz
   470:             # all_reduce(MAX)-pads every rank to the same micro-batch count and
   471:             # FSDP param all-gathers stay in lockstep (no NCCL deadlock).
   472:             _dp_group = torch.distributed.group.WORLD if torch.distributed.is_initialized() else None
   473:             micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len, dp_group=_dp_group)
   474:         else:
   475:             micro_batches = data.split(micro_batch_size)
   476: 
   477:         log_probs_lst = []
   478:         entropy_lst = []
   479:         sum_pi_squared_lst = []
   480:         for micro_batch in micro_batches:
   481:             micro_batch = micro_batch.to(get_device_id())
   482:             model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
   483:             with torch.no_grad():
   484:                 outputs = self._forward_micro_batch(
   485:                     model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
   486:                 )
   487:             log_probs_lst.append(outputs["log_probs"])
   488:             if calculate_entropy:
   489:                 entropy_lst.append(outputs["entropys"])
   490:             if calculate_sum_pi_squared:
   491:                 sum_pi_squared_lst.append(outputs["sum_pi_squared"])
   492: 
   493:         log_probs = torch.concat(log_probs_lst, dim=0)
   494:         if calculate_entropy:
   495:             entropys = torch.concat(entropy_lst, dim=0)
   496:         if calculate_sum_pi_squared:
   497:             sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)
   498: 
   499:         if use_dynamic_bsz:
   500:             log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

[truncated: showing at most 500 lines / 60000 bytes from verl/verl/workers/actor/dp_actor.py]
```

### `verl/verl/workers/actor/megatron_actor.py`  [EDITABLE — lines 41–42 only]

```python
     1: # Copyright 2024 Bytedance Ltd. and/or its affiliates
     2: #
     3: # Licensed under the Apache License, Version 2.0 (the "License");
     4: # you may not use this file except in compliance with the License.
     5: # You may obtain a copy of the License at
     6: #
     7: #     http://www.apache.org/licenses/LICENSE-2.0
     8: #
     9: # Unless required by applicable law or agreed to in writing, software
    10: # distributed under the License is distributed on an "AS IS" BASIS,
    11: # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    12: # See the License for the specific language governing permissions and
    13: # limitations under the License.
    14: """
    15: Megatron Actor.
    16: In megatron actor, the differences are:
    17: 1. We only make minibatch
    18: 
    19: Note that our model doesn't have to be `MegatronModule` because we don't share embedding in the last layer
    20: """
    21: 
    22: import itertools
    23: import logging
    24: import os
    25: from functools import partial
    26: from typing import Iterable
    27: 
    28: import torch
    29: import torch.distributed
    30: from megatron.core import parallel_state as mpu
    31: from megatron.core.distributed import finalize_model_grads
    32: 
    33: # from megatron.core.optimizer import DistributedOptimizer
    34: from megatron.core.optimizer import DistributedOptimizer
    35: from megatron.core.pipeline_parallel import get_forward_backward_func
    36: from omegaconf import OmegaConf
    37: from torch import nn
    38: 
    39: from verl import DataProto
    40: from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
    41: import verl.trainer.ppo.custom_policy_loss  # noqa: F401  register custom policy loss
    42: from verl.utils.device import get_device_id, get_torch_device
    43: from verl.utils.megatron.pipeline_parallel import make_batch_generator
    44: from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction
    45: from verl.utils.megatron.router_replay_utils import (
    46:     RouterReplayHelper,
    47:     merge_router_topk_indices,
    48:     pp_gather,
    49:     reorder_and_merge_vpp_layers,
    50:     set_router_replay_data,
    51: )
    52: from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits
    53: from verl.utils.megatron_utils import get_megatron_mtp_loss, get_model_config, unwrap_model
    54: from verl.utils.profiler import GPUMemoryLogger
    55: from verl.utils.py_functional import append_to_dict
    56: from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
    57: from verl.utils.torch_functional import broadcast_dict_tensor
    58: from verl.workers.actor import BasePPOActor
    59: from verl.workers.config import MtpConfig
    60: 
    61: __all__ = ["MegatronPPOActor"]
    62: 
    63: 
    64: logger = logging.getLogger(__file__)
    65: logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
    66: 
    67: 
    68: class MegatronPPOActor(BasePPOActor):
    69:     def __init__(
    70:         self,
    71:         config,
    72:         model_config,
    73:         hf_config,
    74:         tf_config,
    75:         actor_module: nn.ModuleList,
    76:         actor_optimizer: DistributedOptimizer,
    77:         mtp_config: MtpConfig = None,
    78:     ):
    79:         """MeagtronPPOActor class. This class implements the simple PPO logics when the model is built with Megatron.
    80: 
    81:         Args:
    82:             config (OmegaConf): the basic config that contains the hyper-parameters of PPO Actor. It must contain
    83: 
    84:                 ``ppo_micro_batch_size_per_gpu``: micro batch size when updating ppo.
    85: 
    86:                 ``ppo_mini_batch_size``: minibatch size when updating ppo using the batch data.
    87: 
    88:                 ``ppo_epochs``: number of epochs to update the actor using the batch data.
    89: 
    90:                 ``shuffle``: whether to shuffle the data after each ppo epoch.
    91: 
    92:                 ``clip_ratio``: clip ratio of the ppo algorithm. See https://arxiv.org/abs/1707.06347.
    93: 
    94:                 ``entropy_coeff``: entropy coefficient of the PPO loss. See https://arxiv.org/abs/1707.06347.
    95:             model_config (OmegaConf): model configuration. It must contains ``model_config.vocab_size`` and
    96:                 ``model_config.hidden_size``
    97:             hf_config (PretrainedConfig): huggingface config
    98:             tf_config (TransformerConfig): mcore transformer config
    99:             mtp_config (MtpConfig): mtp config, default None
   100:             actor_module (nn.ModuleList): actor module is a ModuleList that contains a list of nn.Module in this
   101:                 pp stage.
   102:                 each nn.Module in this rank holds a vpp module chunk. See https://arxiv.org/pdf/2104.04473.pdf for
   103:                 more details.
   104:                 The actor module has some constraints to follow in order to use the updating logics implemented here
   105: 
   106:                 1. It must implement unpad_input before any computation and pad_input after all the computation.
   107:                 Remove padding is an
   108:                 optimization that removes the padding tokens. See unpad_input and pad_input function in flash-attn
   109:                 (https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py).
   110: 
   111:                 2. Each pp stage must return the hidden state with the same shape [total_nnz, 1, hidden_size],
   112:                 where total_nnz is the number of valid tokens in this batch. If sequence parallel is enabled, the size
   113:                 of the hidden state is [total_nnz // tp, 1, hidden_size].
   114:             actor_optimizer (DistributedOptimizer): currently, we only support DistributedOptimizer in Megatron.
   115:                 It implements
   116:                 zero1 optimizer that shards the optimizer state across dp ranks.
   117: 
   118:         >>> from megatron.training import get_model
   119:         >>> from megatron.optimizer import get_megatron_optimizer
   120:         >>> actor_module = get_model(megatron_actor_model_provider, wrap_with_ddp=True)
   121:         >>> actor_module = nn.ModuleList(actor_module)
   122:         >>> actor_optimizer = get_megatron_optimizer(actor_module)
   123:         >>> actor = MegatronPPOActor(config=config,
   124:         >>>                          model_config=actor_model_config,
   125:         >>>                          hf_config=hf_config,
   126:         >>>                          tf_config=tf_config,
   127:         >>>                          actor_module=actor_module,
   128:         >>>                          actor_optimizer=actor_optimizer)
   129:         """
   130:         super().__init__(config)
   131:         self._validate_config(config)
   132:         self.model_config = model_config
   133:         self.hf_config = hf_config
   134:         self.tf_config = tf_config
   135:         self.mtp_config = mtp_config
   136:         self.actor_module = actor_module
   137:         self.actor_optimizer: DistributedOptimizer = actor_optimizer
   138: 
   139:         if self.mtp_config:
   140:             assert self.mtp_config.enable, "MTP requires mtp_config.enable to be True"
   141: 
   142:         self.use_fused_kernels = self.config.get("use_fused_kernels", False)
   143:         if getattr(self.mtp_config, "enable", False) and self.use_fused_kernels:
   144:             self.use_fused_kernels = False
   145:             logger.warning_once(
   146:                 "MTP is not compatible with fused kernels for now. Automatically disable use_fused_kernels."
   147:             )
   148:         if self.use_fused_kernels and not getattr(self.config, "overlap_moe_expert_parallel_comm", False):
   149:             # do not patch if overlap_moe_expert_parallel_comm is enabled
   150:             logger.warning_once(
   151:                 "Recommend to disable use_fused_kernels since the fused kernel's performance is broken for triton>=3.3"
   152:                 "Unless you are using a very old version of triton < 3.3"
   153:             )
   154:             from verl.models.mcore.model_forward_fused import patch_fused_forward
   155: 
   156:             for model in self.actor_module:
   157:                 patch_fused_forward(model)
   158:         else:
   159:             from verl.models.mcore.mtp_patch import patch_postprocess
   160: 
   161:             for model in self.actor_module:
   162:                 if self.mtp_config:
   163:                     from verl.models.mcore.mtp_patch import patch_mtp_layer_get_embeddings
   164: 
   165:                     patch_postprocess(model)
   166: 
   167:                     if self.mtp_config.detach_encoder:
   168:                         patch_mtp_layer_get_embeddings(model)
   169: 
   170:         self.optimizer_step_args = OmegaConf.create(
   171:             {
   172:                 "skip_grad": None,
   173:                 "overlap_dp_param_comm": False,
   174:                 "overlap_dp_grad_comm": False,
   175:                 "gradient_accumulation_steps": 1,
   176:                 "sequence_parallel": self.tf_config.sequence_parallel,
   177:                 "DDP_impl": "local",
   178:                 "layernorm_allreduce_bucket_threshold": 0,
   179:                 "reduce_grads_use_alltoall": False,
   180:             }
   181:         )
   182: 
   183:         self.router_replay = self.config.router_replay
   184:         self.enable_routing_replay = self.router_replay.mode != "disabled"
   185:         if self.enable_routing_replay:
   186:             self.mini_layer_topk_idx_list = []
   187: 
   188:         config = get_model_config(self.actor_module[0])
   189:         print(config)
   190:         config.finalize_model_grads_func = finalize_model_grads
   191: 
   192:     def _validate_config(self, config) -> None:
   193:         """Validate config options not implemented for Megatron backend"""
   194:         assert config.get("ulysses_sequence_parallel_size", 1) == 1
   195:         if config.get("shuffle", False):
   196:             assert config.data_loader_seed is not None, "If shuffle dataloader, seed must be manually set"
   197:         if config.megatron.tensor_model_parallel_size == 1:
   198:             print("[Warining] Because actor tp size == 1, set sp to False")
   199:             config.megatron.sequence_parallel = False
   200:         self.config = config
   201: 
   202:     @GPUMemoryLogger(role="megatron actor", logger=logger)
   203:     def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
   204:         """Compute the log probability of the responses given input_ids, attention_mask and position_ids
   205: 
   206:         Args:
   207:             data (DataProto): a DataProto containing keys
   208: 
   209:                 ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
   210:                 concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.
   211: 
   212:                 ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.
   213: 
   214:                 ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.
   215: 
   216:                 ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.
   217: 
   218:         Returns:
   219:             DataProto: torch.Tensor: the log_prob tensor
   220:         """
   221:         prev_modes = [m.training for m in self.actor_module]
   222:         for module in self.actor_module:
   223:             module.eval()
   224:         use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)
   225:         micro_batch_size = data.meta_info.get("micro_batch_size", None)
   226:         max_token_len = data.meta_info.get("max_token_len", None)
   227:         if use_dynamic_bsz:
   228:             assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
   229:             max_token_len = max_token_len * self.config.megatron.context_parallel_size
   230:         else:
   231:             assert micro_batch_size is not None, (
   232:                 "micro batch size is needed for forward compute when use_dynamic_bsz is False"
   233:             )
   234: 
   235:         def compute_logprobs_fn(output, data, use_dynamic_bsz=False, indices=None):
   236:             response = data["responses"]
   237:             response_length = response.size(1)
   238:             log_probs = output["log_probs"][:, -response_length - 1 : -1].contiguous()
   239:             return {"log_probs": log_probs}
   240: 
   241:         # We make recompute_old_log_prob by default here.
   242:         # TODO (zhangchi.usc1992): actually, this function should only return log_prob and this logic should be
   243:         # handled by user outside
   244:         recompute_old_log_prob = self.config.get("recompute_old_log_prob", True)
   245: 
   246:         entropys = torch.Tensor()
   247:         if recompute_old_log_prob:
   248:             select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
   249: 
   250:             if self.enable_routing_replay and self.config.router_replay.mode == "R3":
   251:                 assert "routed_experts" in data.batch.keys(), "routed_experts must be in data.batch.keys()"
   252:                 select_keys.append("routed_experts")
   253: 
   254:             batch = data.select(batch_keys=select_keys).batch
   255:             input_ids = batch["input_ids"]
   256:             batch_size = input_ids.size(0)
   257:             response = batch["responses"]
   258:             response_length = response.size(1)
   259:             with torch.no_grad():
   260:                 output = self.forward_backward_batch(
   261:                     data,
   262:                     forward_only=True,
   263:                     post_process_fn=compute_logprobs_fn,
   264:                     calculate_entropy=calculate_entropy,
   265:                     use_dynamic_bsz=use_dynamic_bsz,
   266:                     micro_batch_size=micro_batch_size,
   267:                     max_token_len=max_token_len,
   268:                 )
   269:                 if mpu.is_pipeline_last_stage(ignore_virtual=True):
   270:                     # only on last rank. It should be on every tp rank
   271:                     if calculate_entropy:
   272:                         log_probs = [o[0]["log_probs"] for o in output["output"]]  # (bs, seq_size)
   273:                     else:
   274:                         log_probs = [o["log_probs"] for o in output["output"]]  # (bs, seq_size)
   275:                     log_probs = torch.cat(log_probs, dim=0).to(torch.float32)
   276:                     if use_dynamic_bsz:
   277:                         indices = output["indices"]
   278:                         indices = list(itertools.chain.from_iterable(indices))
   279:                         assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
   280:                         revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
   281:                         log_probs = log_probs[revert_indices]
   282:                 else:
   283:                     log_probs = torch.empty(
   284:                         size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device
   285:                     )
   286:                 log_probs = log_probs.to(get_device_id())
   287:                 # broadcast across pp ranks
   288:                 torch.distributed.broadcast(
   289:                     tensor=log_probs,
   290:                     src=mpu.get_pipeline_model_parallel_last_rank(),
   291:                     group=mpu.get_pipeline_model_parallel_group(),
   292:                     async_op=False,
   293:                 )
   294:                 log_probs = log_probs.to("cpu")
   295:                 if calculate_entropy:
   296:                     # Note that o[0] is metrics, o[1] is entropy
   297:                     if mpu.is_pipeline_last_stage(ignore_virtual=True):
   298:                         entropys = torch.cat([o[1] for o in output["output"]], dim=0)
   299:                         entropys = entropys.to(torch.float32)
   300:                         if use_dynamic_bsz:
   301:                             indices = output["indices"]
   302:                             indices = list(itertools.chain.from_iterable(indices))
   303:                             assert len(indices) == entropys.size(0), f"{len(indices)} vs. {entropys.size()}"
   304:                             revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
   305:                             entropys = entropys[revert_indices]
   306:                     else:
   307:                         entropys = torch.empty(
   308:                             size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device
   309:                         )
   310:                     # broadcast across pp ranks
   311:                     entropys = entropys.to(get_device_id())
   312:                     torch.distributed.broadcast(
   313:                         tensor=entropys,
   314:                         src=mpu.get_pipeline_model_parallel_last_rank(),
   315:                         group=mpu.get_pipeline_model_parallel_group(),
   316:                         async_op=False,
   317:                     )
   318:                     entropys = entropys.to("cpu")
   319:                 layers_topk_idx = None
   320: 
   321:                 if RouterReplayHelper.is_r2_record_action(self.tf_config):
   322:                     # (bs, max_seq_len/response_len,local_layer_num,topk)
   323:                     layers_topk_idx = output["mini_layer_topk_idx_tensor"].to(torch.uint8)
   324:                     if use_dynamic_bsz:
   325:                         indices = output["indices"]
   326:                         indices = list(itertools.chain.from_iterable(indices))
   327:                         assert len(indices) == layers_topk_idx.size(0), f"{len(indices)} vs. {layers_topk_idx.size()}"
   328:                         revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
   329:                         layers_topk_idx = layers_topk_idx[revert_indices]
   330:                     layers_topk_idx = pp_gather(layers_topk_idx, self.tf_config)
   331:         # add empty cache after each compute
   332:         get_torch_device().empty_cache()
   333: 
   334:         for module, mode in zip(self.actor_module, prev_modes, strict=False):
   335:             module.train(mode)
   336:         return log_probs, entropys, layers_topk_idx
   337: 
   338:     def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
   339:         """Make minibatch iterator for updating the actor
   340: 
   341:         Args:
   342:             data (DataProto): a DataProto containing keys
   343: 
   344:                 ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64, where
   345:                 ``sequence_length = prompt_length + response_length``
   346: 
   347:                 ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64
   348: 
   349:                 ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64
   350: 
   351:                 ``responses``: tensor of shape [batch_size, response_length]. torch.int64. Note that
   352:                 responses = input_ids[:, -response_length:]
   353: 
   354:                 ``old_log_probs``: tensor of shape [batch_size, response_length]. torch.float32. The log probability
   355:                 of responses.
   356: 
   357:                 ``advantages``: tensor of shape [batch_size, response_length]. torch.float32. The advantages of
   358:                 responses.
   359:                 See PPO paper for details. https://arxiv.org/abs/1707.06347
   360: 
   361:         Returns:
   362: 
   363:         """
   364:         select_keys = [
   365:             "responses",
   366:             "input_ids",
   367:             "attention_mask",
   368:             "response_mask",
   369:             "position_ids",
   370:             "old_log_probs",
   371:             "advantages",
   372:         ]
   373:         if self.config.use_kl_loss:
   374:             select_keys.append("ref_log_prob")
   375:         # Include pre-computed IS weights if present in batch
   376:         # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
   377:         if "rollout_is_weights" in data.batch.keys():
   378:             select_keys.append("rollout_is_weights")
   379:         # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
   380:         if "rollout_log_probs" in data.batch.keys():
   381:             select_keys.append("rollout_log_probs")
   382:         self.has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
   383:         # router replay
   384:         if self.enable_routing_replay:
   385:             select_keys.append("routed_experts")
   386:         if self.has_multi_modal_inputs:
   387:             data = data.select(select_keys, ["multi_modal_inputs"])
   388:         else:
   389:             data = data.select(batch_keys=select_keys)
   390: 
   391:         return data.make_iterator(
   392:             mini_batch_size=self.config.ppo_mini_batch_size,
   393:             epochs=self.config.ppo_epochs,
   394:             seed=self.config.data_loader_seed,
   395:             dataloader_kwargs={"shuffle": self.config.shuffle},
   396:         )
   397: 
   398:     def forward_backward_batch(
   399:         self,
   400:         data: DataProto,
   401:         forward_only=False,
   402:         post_process_fn=None,
   403:         calculate_entropy=False,
   404:         use_dynamic_bsz=False,
   405:         micro_batch_size=None,
   406:         max_token_len=None,
   407:         mini_batch_size=None,
   408:     ):
   409:         """
   410:         We assume:
   411:         - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
   412:         - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
   413:         """
   414:         # broadcast from last pp rank to all other pp ranks
   415:         # TODO: actually, we just need to control the sampling order.
   416:         data.to(get_device_id())
   417:         data.batch = data.batch.contiguous()
   418:         mini_batch = data
   419:         broadcast_dict_tensor(
   420:             mini_batch.batch,
   421:             src=mpu.get_pipeline_model_parallel_last_rank(),
   422:             group=mpu.get_pipeline_model_parallel_group(),
   423:         )
   424:         mini_batch.to("cpu")
   425:         # split into micro-batches
   426:         mini_batch.batch["attention_mask"] = mini_batch.batch["attention_mask"].to(bool)
   427:         self.has_multi_modal_inputs = "multi_modal_inputs" in mini_batch.non_tensor_batch.keys()
   428:         if self.has_multi_modal_inputs:
   429:             mini_batch.batch["multi_modal_inputs"] = mini_batch.non_tensor_batch["multi_modal_inputs"]
   430:             mini_batch.batch["multi_modal_inputs_idx"] = torch.Tensor(
   431:                 list(range(len(mini_batch.non_tensor_batch["multi_modal_inputs"])))
   432:             ).to(torch.int64)
   433: 
   434:         if mini_batch.batch["position_ids"].dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
   435:             mini_batch.batch["position_ids"] = mini_batch.batch["position_ids"][
   436:                 :, 0
   437:             ]  # mcore patch recompute qwen2vl's pos ids during forward
   438: 
   439:         indices = None
   440:         temperature = data.meta_info["temperature"]
   441:         if use_dynamic_bsz:
   442:             assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
   443:             vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
   444:             if vpp_size is not None and vpp_size > 1:
   445:                 microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
   446:                 micro_batches, indices = rearrange_micro_batches(
   447:                     batch=mini_batch.batch,
   448:                     num_batches_divided_by=microbatch_group_size_per_vp_stage,
   449:                     max_token_len=max_token_len,
   450:                 )
   451:                 assert len(micro_batches) % self.tf_config.microbatch_group_size_per_vp_stage == 0, (
   452:                     f"micro_batches {micro_batches} must be divisible by microbatch_group_size_per_vp_stage "
   453:                     f"{microbatch_group_size_per_vp_stage} for megatron backend"
   454:                 )
   455:             else:
   456:                 micro_batches, indices = rearrange_micro_batches(batch=mini_batch.batch, max_token_len=max_token_len)
   457:             total_seqlen = max_token_len
   458:         else:
   459:             assert micro_batch_size is not None, (
   460:                 "micro_batch_size is needed to be passed in when not using dynamic batch size"
   461:             )
   462:             micro_batches = mini_batch.batch.split(micro_batch_size)
   463:             seq_len = micro_batches[0]["input_ids"].shape[1]
   464:             total_seqlen = micro_batch_size * seq_len
   465:         # compute input shapes for pp stages
   466:         n_micro_batch = len(micro_batches)
   467: 
   468:         forward_backward_func = get_forward_backward_func()
   469: 
   470:         def loss_func(output, data, meta_info):
   471:             # For memory efficiency
   472:             # We move calculation of entropy to compute_log_probs, forward_only == True
   473:             log_probs = None
   474:             entropy = None
   475:             if isinstance(output, dict):
   476:                 log_probs = output["log_probs"]
   477:                 if "entropy" in output:
   478:                     entropy = output["entropy"]
   479:             else:
   480:                 assert isinstance(output, torch.Tensor)
   481:                 log_probs = output
   482: 
   483:             device = log_probs.device
   484:             metrics = {}
   485:             if forward_only:
   486:                 if post_process_fn is None:
   487:                     pass
   488:                     # metrics["logits"] = output
   489:                 else:
   490:                     stats = post_process_fn(output, data)
   491:                     metrics.update(stats)
   492:                 if not calculate_entropy:
   493:                     return torch.tensor(1.0, device=device), metrics
   494: 
   495:             responses = data["responses"]
   496:             response_length = responses.size(1)
   497:             response_mask = data["response_mask"].to(bool)
   498:             loss_agg_mode = self.config.loss_agg_mode
   499:             # compute policy loss
   500:             log_prob = log_probs[:, -response_length - 1 : -1].contiguous()

[truncated: showing at most 500 lines / 60000 bytes from verl/verl/workers/actor/megatron_actor.py]
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **deepmath-3bench-h100** — wall-clock budget `6:00:00`, compute share `2`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `token_level` baseline — editable region  [READ-ONLY — reference implementation]

In `verl/verl/trainer/ppo/custom_policy_loss.py`:

```python
Lines 17–66:
    14: # EDITABLE: Implement your custom importance-sampling policy loss below.
    15: # =====================================================================
    16: 
    17: # =====================================================================
    18: 
    19: 
    20: @register_policy_loss("custom")
    21: def compute_custom_policy_loss(
    22:     old_log_prob: torch.Tensor,
    23:     log_prob: torch.Tensor,
    24:     advantages: torch.Tensor,
    25:     response_mask: torch.Tensor,
    26:     loss_agg_mode: str = "token-mean",
    27:     config: Optional[ActorConfig] = None,
    28:     rollout_is_weights: torch.Tensor | None = None,
    29: ) -> tuple[torch.Tensor, dict[str, Any]]:
    30:     """Token-level vanilla PPO: per-token ratio, per-token clip."""
    31:     assert config is not None
    32:     clip_ratio = config.clip_ratio
    33:     clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    34:     clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    35:     clip_ratio_c = config.get("clip_ratio_c", 3.0)
    36:     assert clip_ratio_c > 1.0
    37: 
    38:     negative_approx_kl = log_prob - old_log_prob
    39:     negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    40:     ratio = torch.exp(negative_approx_kl)
    41:     ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    42: 
    43:     pg_losses1 = -advantages * ratio
    44:     pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    45:     clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    46:     pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    47: 
    48:     pg_losses3 = -advantages * clip_ratio_c
    49:     clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    50:     pg_clipfrac_lower = verl_F.masked_mean(
    51:         torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    52:     )
    53:     pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    54: 
    55:     if rollout_is_weights is not None:
    56:         pg_losses = pg_losses * rollout_is_weights
    57: 
    58:     pg_loss = agg_loss(
    59:         loss_mat=pg_losses, loss_mask=response_mask,
    60:         loss_agg_mode=loss_agg_mode, **config.global_batch_info,
    61:     )
    62:     return pg_loss, {
    63:         "actor/pg_clipfrac": pg_clipfrac.detach().item(),
    64:         "actor/ppo_kl": ppo_kl.detach().item(),
    65:         "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    66:     }
```

### `sequence_level` baseline — editable region  [READ-ONLY — reference implementation]

In `verl/verl/trainer/ppo/custom_policy_loss.py`:

```python
Lines 17–62:
    14: # EDITABLE: Implement your custom importance-sampling policy loss below.
    15: # =====================================================================
    16: 
    17: # =====================================================================
    18: 
    19: 
    20: @register_policy_loss("custom")
    21: def compute_custom_policy_loss(
    22:     old_log_prob: torch.Tensor,
    23:     log_prob: torch.Tensor,
    24:     advantages: torch.Tensor,
    25:     response_mask: torch.Tensor,
    26:     loss_agg_mode: str = "token-mean",
    27:     config: Optional[ActorConfig] = None,
    28:     rollout_is_weights: torch.Tensor | None = None,
    29: ) -> tuple[torch.Tensor, dict[str, Any]]:
    30:     """Sequence-level IS (GSPO): one scalar ratio per sequence."""
    31:     assert config is not None
    32:     clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    33:     clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    34: 
    35:     negative_approx_kl = log_prob - old_log_prob
    36:     seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    37:     neg_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths
    38: 
    39:     # straight-through: keep per-token log_prob gradient, ratio value is per-sequence
    40:     log_seq_ratio = log_prob - log_prob.detach() + neg_kl_seq.detach().unsqueeze(-1)
    41:     log_seq_ratio = torch.clamp(log_seq_ratio, max=10.0)
    42:     seq_ratio = torch.exp(log_seq_ratio)
    43: 
    44:     pg_losses1 = -advantages * seq_ratio
    45:     pg_losses2 = -advantages * torch.clamp(seq_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    46:     pg_losses = torch.maximum(pg_losses1, pg_losses2)
    47: 
    48:     if rollout_is_weights is not None:
    49:         pg_losses = pg_losses * rollout_is_weights
    50: 
    51:     # GSPO aggregates at the sequence level (seq-mean-token-mean)
    52:     pg_loss = agg_loss(
    53:         loss_mat=pg_losses, loss_mask=response_mask,
    54:         loss_agg_mode="seq-mean-token-mean", **config.global_batch_info,
    55:     )
    56:     pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    57:     ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    58:     return pg_loss, {
    59:         "actor/pg_clipfrac": pg_clipfrac.detach().item(),
    60:         "actor/ppo_kl": ppo_kl.detach().item(),
    61:         "actor/pg_clipfrac_lower": 0.0,
    62:     }
```

### `first_k_tokens` baseline — editable region  [READ-ONLY — reference implementation]

In `verl/verl/trainer/ppo/custom_policy_loss.py`:

```python
Lines 17–65:
    14: # EDITABLE: Implement your custom importance-sampling policy loss below.
    15: # =====================================================================
    16: 
    17: # =====================================================================
    18: 
    19: 
    20: @register_policy_loss("custom")
    21: def compute_custom_policy_loss(
    22:     old_log_prob: torch.Tensor,
    23:     log_prob: torch.Tensor,
    24:     advantages: torch.Tensor,
    25:     response_mask: torch.Tensor,
    26:     loss_agg_mode: str = "token-mean",
    27:     config: Optional[ActorConfig] = None,
    28:     rollout_is_weights: torch.Tensor | None = None,
    29: ) -> tuple[torch.Tensor, dict[str, Any]]:
    30:     """First-K truncated IS: per-token ratio for t<K, detached for t>=K."""
    31:     assert config is not None
    32:     K = 64  # prefix length with live IS gradient
    33:     clip_ratio = config.clip_ratio
    34:     clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    35:     clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    36: 
    37:     negative_approx_kl = log_prob - old_log_prob
    38:     negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    39:     ratio = torch.exp(negative_approx_kl)
    40:     ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    41: 
    42:     # Build prefix mask: 1 for the first K response positions, 0 afterwards.
    43:     T = ratio.shape[-1]
    44:     positions = torch.arange(T, device=ratio.device).unsqueeze(0)  # (1, T)
    45:     prefix_mask = (positions < K).to(ratio.dtype)                  # (1, T)
    46:     # Detach ratio beyond prefix: ratio_eff = ratio*prefix + detach(ratio)*(1-prefix)
    47:     ratio_eff = ratio * prefix_mask + ratio.detach() * (1.0 - prefix_mask)
    48: 
    49:     pg_losses1 = -advantages * ratio_eff
    50:     pg_losses2 = -advantages * torch.clamp(ratio_eff, 1 - clip_ratio_low, 1 + clip_ratio_high)
    51:     pg_losses = torch.maximum(pg_losses1, pg_losses2)
    52: 
    53:     if rollout_is_weights is not None:
    54:         pg_losses = pg_losses * rollout_is_weights
    55: 
    56:     pg_loss = agg_loss(
    57:         loss_mat=pg_losses, loss_mask=response_mask,
    58:         loss_agg_mode=loss_agg_mode, **config.global_batch_info,
    59:     )
    60:     pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    61:     return pg_loss, {
    62:         "actor/pg_clipfrac": pg_clipfrac.detach().item(),
    63:         "actor/ppo_kl": ppo_kl.detach().item(),
    64:         "actor/pg_clipfrac_lower": 0.0,
    65:     }
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

## Time Budget

You have **5 hours** of wall-clock time before submission, covering
everything you do here: reading the code, editing it, and any trial runs
you launch.

Good luck.
