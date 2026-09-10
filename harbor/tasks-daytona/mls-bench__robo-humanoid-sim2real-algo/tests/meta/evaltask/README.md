# Robo-Humanoid Sim2Real Algorithm Task

This task evaluates an agent's ability to design and optimize reinforcement learning algorithms for humanoid robot locomotion.

## Task Description

The agent is given freedom to modify the PPO algorithm implementation in humanoidgym, including:
- **Network Architecture** (actor_critic_custom.py): Modify the actor and critic neural network architectures, activation functions, initialization strategies, etc.
- **Algorithm Implementation** (ppo_custom.py): Adjust the PPO algorithm including loss functions, optimization strategies, learning rate schedules, clipping parameters, etc.
- **Experience Buffer** (rollout_storage_custom.py): Modify how experiences are stored and sampled, including advantage computation, return calculation, and mini-batch generation.

## Evaluation

The task uses a sim2sim evaluation protocol:
- Training and evaluation on the XBotL humanoid robot
- Performance measured by locomotion success in simulation and sim-to-real transfer

## Files

### Editable Files
- `humanoid-gym/humanoid/algo/ppo/actor_critic_custom.py` - Neural network architecture
- `humanoid-gym/humanoid/algo/ppo/ppo_custom.py` - PPO algorithm implementation
- `humanoid-gym/humanoid/algo/ppo/rollout_storage_custom.py` - Experience buffer

### Reference Files (Read-only)
- `humanoid-gym/humanoid/algo/ppo/actor_critic.py` - Original actor-critic implementation
- `humanoid-gym/humanoid/algo/ppo/ppo.py` - Original PPO implementation
- `humanoid-gym/humanoid/algo/ppo/rollout_storage.py` - Original rollout storage implementation

## Runtime requirement: NVIDIA driver >= 570 on Hopper and newer

IsaacGym Preview 4 ships a `libPhysXGpu_64.so` whose newest cubin is `sm_80`,
plus a PTX payload targeting `sm_86`. On a Hopper GPU (`sm_90`) there is no
matching cubin, so the CUDA driver has to JIT-compile that PTX when the sim is
created. That JIT runs in the **user-space** driver, and 5xx user-space branches
older than 570 are known to segfault in it: the run dies with a bare
`Segmentation fault` inside `create_sim`, no python traceback, and scores 0
([issue #59](https://github.com/Imbernoulli/MLS-Bench/issues/59), whose gdb
backtrace lands inside `libnvidia-ptxjitcompiler` on driver 535.161.08).

Only the *user-space* driver has to move — the host's kernel driver can stay on
535/550. `scripts/gpu_env_preflight.sh`, sourced by every script in this task,
handles it:

- it logs one `GPU_ENV_PREFLIGHT` line with the GPU's compute capability, the
  kernel driver, the user-space driver actually loaded, and the device code
  present in `libPhysXGpu_64.so`;
- if the GPU is `sm_90`+ and the user-space driver is older than 570, it
  activates a CUDA forward-compatibility stack (`/opt/mlsbench/cuda-compat`,
  baked into the `humanoid-gym` image from NVIDIA's `cuda-compat-12-8`), after
  verifying that the new `libcuda` really is the one that loads and that
  `cuInit` succeeds — forward compat is a data-center-GPU feature, so it rolls
  back cleanly on hardware where it does not apply;
- if it cannot repair the stack, it prints what to do instead of letting the run
  segfault silently.

Overrides: `MLSBENCH_CUDA_COMPAT_DIR` (use your own compat directory),
`MLSBENCH_GPU_PREFLIGHT_MIN_DRIVER` (change the 570 threshold),
`MLSBENCH_SKIP_GPU_PREFLIGHT=1` (turn the whole thing off).

A separate failure with the same symptom is a **stripped** `libPhysXGpu_64.so`:
several community IsaacGym images prune the PTX payload, which leaves nothing
for any driver to JIT ([issue #47](https://github.com/Imbernoulli/MLS-Bench/issues/47)).
The preflight detects that too and says so explicitly.
Use the stock Preview 4 binary that
`vendor/data_scripts/humanoid-gym/prepare_isaacgym.py` downloads.

## Baseline

The default baseline uses the official PPO implementation from humanoid-gym with:
- 3-layer MLP with 256 hidden units and ELU activation
- Standard PPO with clipped surrogate objective
- GAE for advantage estimation
- Standard experience buffer with mini-batch sampling
