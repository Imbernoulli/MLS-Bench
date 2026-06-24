"""Pre-edit operations for humanoid-gym package.

Registers the custom environment XBotLCustomEnv with custom config in the task registry.
Also stubs wandb — the container's pydantic is too old for wandb's ConfigDict import,
and we don't need online logging in the MLS-Bench evaluation loop.
"""

_WANDB_STUB = '''\
# wandb stubbed out — container's pydantic is too old for wandb\'s ConfigDict import,
# and we don\'t need online logging here. This no-op wandb satisfies `import wandb`
# and `wandb.init(...)` without running any real init.
class _WandbStub:
    def init(self, *a, **kw): return None
    def log(self, *a, **kw): return None
    def finish(self, *a, **kw): return None
    def __getattr__(self, name): return self
    def __call__(self, *a, **kw): return self
import sys as _sys
_sys.modules[\'wandb\'] = _WandbStub()

# Tensorboard stubbed out — TB appends ~1MB/iter for 10k iters and the
# shared filesystem quota becomes a real problem on long PPO runs.
# MLS-Bench parses TRAIN_METRICS text lines, not TB scalars, so replace
# SummaryWriter with a no-op so add_scalar/add_histogram/etc. don\'t hit disk.
class _TBStub:
    def __init__(self, *a, **kw):
        # Real SummaryWriter creates log_dir as a side effect; downstream
        # code (rsl_rl save() in particular) expects the dir to exist.
        import os as _os
        log_dir = a[0] if a else kw.get(\'log_dir\')
        if log_dir:
            _os.makedirs(log_dir, exist_ok=True)
    def add_scalar(self, *a, **kw): pass
    def add_scalars(self, *a, **kw): pass
    def add_histogram(self, *a, **kw): pass
    def add_text(self, *a, **kw): pass
    def add_image(self, *a, **kw): pass
    def flush(self): pass
    def close(self): pass
    def __getattr__(self, name):
        return lambda *a, **kw: None
import torch.utils.tensorboard as _tb
_tb.SummaryWriter = _TBStub

# --- TF32 matmul parity (sm_90 torch upgrade) ---------------------------------
# The recorded anchor (sim2sim success 0.73) was trained under torch 1.12, whose
# default is torch.backends.cuda.matmul.allow_tf32 == True: every fp32 nn.Linear
# matmul in the actor/critic MLPs and PPO loss ran in TF32 (~10-bit mantissa).
# torch >=2.x flips this default to False (full fp32, "highest" precision). With
# identical CPU RNG and identical IsaacGym/obs eager numerics, this is the SOLE
# diverging framework default between the two images, and it deterministically
# shifts the optimization trajectory: the IsaacGym training reward still saturates
# (~182) but the resulting policy generalizes worse in the MuJoCo sim2sim eval
# (0.73 -> 0.20, same on seeds 42 and 123). Restore the torch-1.12 behavior so a
# newer (sm_90-capable) torch reproduces the anchor.
import torch as _torch
try:
    _torch.backends.cuda.matmul.allow_tf32 = True
    _torch.backends.cudnn.allow_tf32 = True
    if hasattr(_torch, "set_float32_matmul_precision"):
        # "high" == TF32 for fp32 matmuls (matches torch 1.12 default behavior).
        _torch.set_float32_matmul_precision("high")
except Exception:
    pass
'''

OPS = [
    {
        "op": "insert",
        "file": "humanoid-gym/humanoid/envs/__init__.py",
        "after_line": 36,
        "content": "from .custom.humanoid_config_custom import XBotLCustomCfg, XBotLCustomCfgPPO\n",
    },
    {
        "op": "insert",
        "file": "humanoid-gym/humanoid/envs/__init__.py",
        "after_line": 37,
        "content": "from .custom.humanoid_env_custom import XBotLCustomEnv\n",
    },
    {
        "op": "insert",
        "file": "humanoid-gym/humanoid/envs/__init__.py",
        "after_line": 42,
        "content": "\ntask_registry.register( \"humanoid_custom\", XBotLCustomEnv, XBotLCustomCfg(), XBotLCustomCfgPPO() )\n",
    },
    {
        "op": "insert",
        "file": "humanoid-gym/humanoid/algo/ppo/on_policy_runner.py",
        "after_line": 276,
        "content": """
        # Output parseable training metrics for MLS-Bench
        if len(locs["rewbuffer"]) > 0:
            print(f"TRAIN_METRICS iter={locs['it']} mean_reward={statistics.mean(locs['rewbuffer']):.4f} mean_value_loss={locs['mean_value_loss']:.4f} mean_surrogate_loss={locs['mean_surrogate_loss']:.4f}", flush=True)
""",
    },
    {
        "op": "replace",
        "file": "humanoid-gym/humanoid/algo/ppo/__init__.py",
        "start_line": 33,
        "end_line": 36,
        "content": """from .ppo_custom import PPO
from .on_policy_runner import OnPolicyRunner
from .actor_critic_custom import ActorCritic
from .rollout_storage_custom import RolloutStorage
""",
    },
    {
        "op": "replace",
        "file": "humanoid-gym/humanoid/algo/ppo/on_policy_runner.py",
        "start_line": 39,
        "end_line": 40,
        "content": """from .ppo_custom import PPO
from .actor_critic_custom import ActorCritic
""",
    },
    {
        "op": "insert",
        "file": "humanoid-gym/humanoid/algo/ppo/on_policy_runner.py",
        "after_line": 31,
        "content": _WANDB_STUB,
    },
    # Replace JIT-based policy export with plain torch.save. Reason: we set
    # PYTORCH_JIT=0 to bypass nvrtc 11.6 sm_90 crashes on H200 during training,
    # which also breaks `torch.jit.script` (used in the original export). Saving
    # the actor module via plain pickle and loading via torch.load works fine
    # with JIT disabled.
    {
        "op": "replace",
        "file": "humanoid-gym/humanoid/utils/helpers.py",
        "start_line": 248,
        "end_line": 254,
        "content": '''def export_policy_as_jit(actor_critic, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy_1.pt")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    model.eval()
    # Plain pickle save instead of torch.jit.script — compatible with PYTORCH_JIT=0
    torch.save(model, path)
''',
    },
    # Force graphics_device_id=-1 when --headless to skip Isaac Gym's NvfDeviceCreate
    # graphics-init path. On host driver 595/CUDA 13.2, libnvf.plugin.so dereferences
    # a NULL Vulkan function pointer inside NvfDeviceCreate (verified via gdb on
    # SLURM job 7377041). Setting graphics_device_id=-1 makes Isaac Gym skip the
    # entire GymGraphicsNvf init in carb::gym::GymCreateSim, which lets headless
    # training proceed normally. We always run --headless in MLS-Bench so this is
    # effectively a no-op for any case where rendering would have worked.
    {
        "op": "replace",
        "file": "humanoid-gym/humanoid/envs/base/base_task.py",
        "start_line": 60,
        "end_line": 60,
        "content": "        self.graphics_device_id = -1 if self.headless else self.sim_device_id\n",
    },
]
