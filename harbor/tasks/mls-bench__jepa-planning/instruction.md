# MLS-Bench: jepa-planning

# JEPA World Model Planning: Algorithm Design

## Objective
Design a planning algorithm that exploits a learned JEPA (Joint Embedding Predictive Architecture) world model for goal-conditioned navigation, where the agent must navigate around walls and through doorways to reach a randomly sampled goal location.

## Research Question
Can you design a planning algorithm that outperforms standard derivative-free methods such as CEM and MPPI by better exploiting the structure of a learned JEPA world model?

## Background
JEPA world models predict future latent encodings rather than future observations, so planning happens entirely in representation space. Recent work studies offline learning of latent dynamics models with JEPA-style objectives and uses derivative-free planners on top of them — see Sobal et al. 2025, "Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models" (arXiv:2502.14819). Two standard derivative-free planners used for this kind of latent world model are:
- **CEM (Cross-Entropy Method)**: iteratively refits a Gaussian over action sequences using the top-`k` elites under the model rollout cost.
- **MPPI (Model Predictive Path Integral)**: importance-weighted update over sampled action sequences (Williams et al., arXiv:1509.01149).

The JEPA world model checkpoint is fixed and provided by the evaluation environment; the task is to improve planning, not to retrain the model.

## What You Can Modify
You implement the `CustomPlanner` class in `custom_planner.py`. The class extends the `Planner` abstract base class and must implement the `plan()` method.

## Interface

### CustomPlanner Constructor
```python
def __init__(self, unroll, action_dim=2, plan_length=15,
             num_samples=200, n_iters=20, **kwargs):
```
- `unroll`: function that forward-simulates through the world model
- `action_dim`: action space dimensionality (2 for x/y movement)
- `plan_length`: maximum planning horizon
- `num_samples`: number of action samples (adjustable)
- `n_iters`: number of optimization iterations (adjustable)

### plan() Method
```python
def plan(self, obs_init, steps_left=None, eval_mode=True,
         t0=False, plan_vis_path=None) -> PlanningResult:
```
- `obs_init`: initial observation encoding `[1, C, 1, H, W]`
- `steps_left`: remaining steps in the episode
- Returns: `PlanningResult(actions=Tensor[T, A], ...)`

### Available Methods (Inherited)
- `self.unroll(obs_init, actions)`: forward-simulate actions through the world model.
  - `obs_init`: `[1, C, 1, H, W]` initial observation encoding
  - `actions`: `[B, A, T]` batch of action sequences
  - Returns: `[B, D, T+1, H, W]` predicted state encodings
- `self.objective(encodings)`: compute cost for predicted state encodings.
  - `encodings`: `[B, D, T, H, W]`
  - Returns: `[B]` cost per sample (lower is better)
- `self.cost_function(actions, obs_init)`: convenience method that calls `unroll` then `objective`.
  - Returns: `[B]` cost per sample

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/eb_jepa/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `eb_jepa/custom_planner.py`
- editable lines **323–367**


Other files you may **read** for context (do not modify):
- `eb_jepa/planning.py`
- `eb_jepa/jepa.py`


## Readable Context


### `eb_jepa/custom_planner.py`  [EDITABLE — lines 323–367 only]

```python
     1: """
     2: Self-contained script for JEPA planning evaluation.
     3: 
     4: Part 1: Load a released AC Video JEPA checkpoint (or retrain on demand).
     5: Part 2: Define CustomPlanner (EDITABLE REGION).
     6: Part 3: Run planning evaluation and report metrics.
     7: """
     8: 
     9: import os
    10: import sys
    11: # Prevent eb_jepa/logging.py from shadowing stdlib logging:
    12: # remove any sys.path entry that contains a `logging.py` file. This is
    13: # path-agnostic and safe when the script lives at the repo root (pwd-script
    14: # dir is /workspace/eb_jepa, which itself has no logging.py — only the
    15: # eb_jepa subpackage does).
    16: sys.path = [p for p in sys.path if not os.path.isfile(os.path.join(p, "logging.py"))]
    17: import random
    18: import time
    19: from pathlib import Path
    20: 
    21: import numpy as np
    22: import torch
    23: import torch.nn as nn
    24: from torch.amp import GradScaler, autocast
    25: from torch.optim import AdamW
    26: from tqdm import tqdm
    27: 
    28: from eb_jepa.architectures import (
    29:     ImpalaEncoder,
    30:     InverseDynamicsModel,
    31:     RNNPredictor,
    32: )
    33: from eb_jepa.datasets.two_rooms.env import DotWall
    34: from eb_jepa.datasets.utils import init_data
    35: from eb_jepa.jepa import JEPA, JEPAProbe
    36: from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
    37: from eb_jepa.planning import (
    38:     Planner,
    39:     PlanningResult,
    40:     ReprTargetDistMPCObjective,
    41: )
    42: from eb_jepa.schedulers import CosineWithWarmup
    43: from eb_jepa.state_decoder import MLPXYHead
    44: 
    45: # ============================================================================
    46: # PART 1: Model Training / Checkpoint Loading
    47: # ============================================================================
    48: 
    49: # Training hyperparameters (from train.yaml).
    50: # NOTE: mlsbench sets the ENV env var to the cmd_label (e.g. "horizon-30"),
    51: # which is NOT the dataset name. The only supported dataset for this task
    52: # is two_rooms, so hardcode it.
    53: TASK_ENV = "two-rooms"
    54: ENV_NAME = "two_rooms"
    55: SEED = int(os.environ.get("SEED", "42"))
    56: PLAN_LENGTH = int(os.environ.get("PLAN_LENGTH", "90"))
    57: OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/eb_jepa/outputs"))
    58: CKPT_DIR = OUTPUT_DIR / "checkpoints"
    59: CKPT_PATH = CKPT_DIR / "ac_jepa.pth.tar"
    60: PRETRAINED_CKPT_PATH = Path(
    61:     os.environ.get(
    62:         "JEPA_PRETRAINED_CKPT",
    63:         "/data/eb_jepa/checkpoints/ac_jepa_e11.pth.tar",
    64:     )
    65: )
    66: FORCE_TRAIN = os.environ.get("JEPA_FORCE_TRAIN", "0") == "1"
    67: EPOCHS = 12
    68: BATCH_SIZE = 384
    69: LR = 0.001
    70: WEIGHT_DECAY = 1e-5
    71: GRAD_CLIP = 2.0
    72: DOBS = 2
    73: HENC = 32
    74: DSTC = 32
    75: NSTEPS = 8
    76: COV_COEFF = 8
    77: STD_COEFF = 16
    78: SIM_COEFF_T = 12
    79: IDM_COEFF = 1
    80: 
    81: 
    82: def build_data_cfg(batch_size):
    83:     """Create the shared dataset config for the selected environment."""
    84:     return {
    85:         "env_name": ENV_NAME,
    86:         "batch_size": batch_size,
    87:         "num_workers": 0,
    88:         "pin_mem": False,
    89:         "persistent_workers": False,
    90:     }
    91: 
    92: 
    93: def resolve_checkpoint_path():
    94:     """Choose the released checkpoint by default, with local fallback."""
    95:     if not FORCE_TRAIN and PRETRAINED_CKPT_PATH.exists():
    96:         return PRETRAINED_CKPT_PATH
    97:     if CKPT_PATH.exists():
    98:         return CKPT_PATH
    99:     return None
   100: 
   101: 
   102: def setup_seed(seed):
   103:     """Set random seeds for reproducibility."""
   104:     os.environ["PYTHONHASHSEED"] = str(seed)
   105:     random.seed(seed)
   106:     np.random.seed(seed)
   107:     torch.manual_seed(seed)
   108:     if torch.cuda.is_available():
   109:         torch.cuda.manual_seed(seed)
   110:         torch.cuda.manual_seed_all(seed)
   111:     torch.backends.cudnn.benchmark = False
   112: 
   113: 
   114: def seed_env(env, seed):
   115:     for obj in (env, getattr(env, "action_space", None), getattr(env, "observation_space", None)):
   116:         if hasattr(obj, "seed"):
   117:             obj.seed(seed)
   118: 
   119: 
   120: def reset_env(env, seed):
   121:     seed_env(env, seed)
   122:     try:
   123:         return env.reset(seed=seed)
   124:     except TypeError:
   125:         return env.reset()
   126: 
   127: 
   128: def build_model(device, loader):
   129:     """Build the JEPA model with all components."""
   130:     img_size = loader.dataset.config.img_size
   131: 
   132:     # Encoder
   133:     encoder = ImpalaEncoder(
   134:         width=1,
   135:         stack_sizes=(16, HENC, DSTC),
   136:         num_blocks=2,
   137:         dropout_rate=None,
   138:         layer_norm=False,
   139:         input_channels=DOBS,
   140:         final_ln=True,
   141:         mlp_output_dim=512,
   142:         input_shape=(DOBS, img_size, img_size),
   143:     )
   144:     test_input = torch.rand((1, DOBS, 1, img_size, img_size))
   145:     test_output = encoder(test_input)
   146:     _, f, _, h, w = test_output.shape
   147: 
   148:     # Predictor
   149:     predictor = RNNPredictor(
   150:         hidden_size=encoder.mlp_output_dim, final_ln=encoder.final_ln
   151:     )
   152: 
   153:     # Action encoder (identity)
   154:     aencoder = nn.Identity()
   155: 
   156:     # IDM
   157:     idm = InverseDynamicsModel(
   158:         state_dim=h * w * f,
   159:         hidden_dim=256,
   160:         action_dim=2,
   161:     ).to(device)
   162: 
   163:     # Regularizer
   164:     regularizer = VC_IDM_Sim_Regularizer(
   165:         cov_coeff=COV_COEFF,
   166:         std_coeff=STD_COEFF,
   167:         sim_coeff_t=SIM_COEFF_T,
   168:         idm_coeff=IDM_COEFF,
   169:         idm=idm,
   170:         first_t_only=False,
   171:         projector=None,
   172:         spatial_as_samples=False,
   173:         idm_after_proj=False,
   174:         sim_t_after_proj=False,
   175:     )
   176: 
   177:     # Loss and JEPA
   178:     ploss = SquareLossSeq()
   179:     jepa = JEPA(encoder, aencoder, predictor, regularizer, ploss).to(device)
   180: 
   181:     # Position prober
   182:     xy_head = MLPXYHead(
   183:         input_shape=test_output.shape[1],
   184:         normalizer=loader.dataset.normalizer,
   185:     ).to(device)
   186:     xy_prober = JEPAProbe(jepa=jepa, head=xy_head, hcost=nn.MSELoss())
   187: 
   188:     return jepa, xy_head, xy_prober, test_output.shape
   189: 
   190: 
   191: def train_model(device):
   192:     """Train the AC Video JEPA model and save checkpoint."""
   193:     print("Training AC Video JEPA model...")
   194: 
   195:     loader, val_loader, data_config = init_data(
   196:         env_name=ENV_NAME,
   197:         cfg_data=build_data_cfg(BATCH_SIZE),
   198:     )
   199: 
   200:     jepa, xy_head, xy_prober, _ = build_model(device, loader)
   201: 
   202:     steps_per_epoch = data_config.size // data_config.batch_size
   203:     total_steps = EPOCHS * steps_per_epoch
   204: 
   205:     jepa_optimizer = AdamW(jepa.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
   206:     jepa_scheduler = CosineWithWarmup(jepa_optimizer, total_steps, warmup_ratio=0.1)
   207: 
   208:     probe_optimizer = AdamW(xy_head.parameters(), lr=1e-3, weight_decay=1e-5)
   209:     probe_scheduler = CosineWithWarmup(probe_optimizer, total_steps, warmup_ratio=0.1)
   210: 
   211:     dtype = torch.bfloat16
   212:     use_amp = True
   213:     scaler = GradScaler(device.type, enabled=use_amp)
   214: 
   215:     for epoch in range(EPOCHS):
   216:         epoch_start = time.time()
   217:         pbar = tqdm(
   218:             enumerate(loader),
   219:             total=len(loader),
   220:             desc=f"Epoch {epoch}/{EPOCHS - 1}",
   221:         )
   222:         for idx, (x, a, loc, _, _) in pbar:
   223:             x = x.to(device)
   224:             a = a.to(device)
   225:             loc = loc.to(device)
   226:             total_loss = torch.tensor(0.0, device=device)
   227: 
   228:             # JEPA loss
   229:             jepa_optimizer.zero_grad()
   230:             with autocast(device.type, enabled=use_amp, dtype=dtype):
   231:                 _, (jepa_loss, regl, regl_unweight, regldict, pl) = jepa.unroll(
   232:                     x, a,
   233:                     nsteps=NSTEPS,
   234:                     unroll_mode="autoregressive",
   235:                     ctxt_window_time=1,
   236:                     compute_loss=True,
   237:                     return_all_steps=False,
   238:                 )
   239:                 total_loss += jepa_loss
   240: 
   241:             scaler.scale(jepa_loss).backward()
   242:             scaler.unscale_(jepa_optimizer)
   243:             torch.nn.utils.clip_grad_norm_(jepa.encoder.parameters(), GRAD_CLIP)
   244:             torch.nn.utils.clip_grad_norm_(jepa.predictor.parameters(), GRAD_CLIP)
   245:             scaler.step(jepa_optimizer)
   246:             scaler.update()
   247:             jepa_scheduler.step()
   248: 
   249:             # Probe loss
   250:             probe_optimizer.zero_grad()
   251:             with autocast(device.type, enabled=use_amp, dtype=dtype):
   252:                 xy_loss = xy_prober(
   253:                     observations=x[:, :, :1],
   254:                     targets=loc[:, :, :1],
   255:                 )
   256:                 xy_loss = loader.dataset.normalizer.unnormalize_mse(xy_loss)
   257:                 total_loss += xy_loss
   258: 
   259:             scaler.scale(xy_loss).backward()
   260:             scaler.step(probe_optimizer)
   261:             scaler.update()
   262:             probe_scheduler.step()
   263: 
   264:             pbar.set_postfix({
   265:                 "loss": f"{total_loss.item():.4f}",
   266:                 "reg": f"{regl.item():.4f}",
   267:                 "pred": f"{pl.item():.4f}",
   268:             })
   269: 
   270:         epoch_time = time.time() - epoch_start
   271:         print(
   272:             f"TRAIN_METRICS: epoch={epoch}, loss={total_loss.item():.4f}, "
   273:             f"reg={regl.item():.4f}, pred={pl.item():.4f}, "
   274:             f"probe={xy_loss.item():.4f}, time={epoch_time:.1f}s",
   275:             flush=True,
   276:         )
   277: 
   278:     # Save checkpoint
   279:     os.makedirs(CKPT_DIR, exist_ok=True)
   280:     ckpt = {
   281:         "epoch": EPOCHS - 1,
   282:         "model_state_dict": jepa.state_dict(),
   283:         "xy_head_state_dict": xy_head.state_dict(),
   284:     }
   285:     torch.save(ckpt, CKPT_PATH)
   286:     print(f"Checkpoint saved to {CKPT_PATH}", flush=True)
   287: 
   288:     return jepa, xy_head, xy_prober, loader
   289: 
   290: 
   291: def load_model(device):
   292:     """Load the JEPA model from checkpoint."""
   293:     loader, val_loader, data_config = init_data(
   294:         env_name=ENV_NAME,
   295:         cfg_data=build_data_cfg(64),
   296:     )
   297: 
   298:     jepa, xy_head, xy_prober, _ = build_model(device, loader)
   299: 
   300:     checkpoint_path = resolve_checkpoint_path()
   301:     if checkpoint_path is None:
   302:         raise FileNotFoundError(
   303:             f"No checkpoint found at {PRETRAINED_CKPT_PATH} or {CKPT_PATH}"
   304:         )
   305:     print(f"Loading checkpoint from {checkpoint_path}...", flush=True)
   306:     checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
   307:     state_dict = checkpoint.get("model_state_dict", {})
   308:     state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
   309:     jepa.load_state_dict(state_dict)
   310: 
   311:     if "xy_head_state_dict" in checkpoint:
   312:         xy_head.load_state_dict(checkpoint["xy_head_state_dict"])
   313: 
   314:     print("Model loaded successfully.", flush=True)
   315:     return jepa, xy_head, xy_prober, loader
   316: 
   317: 
   318: # ============================================================================
   319: # PART 2: Custom Planner (EDITABLE REGION)
   320: # ============================================================================
   321: 
   322: # EDITABLE REGION START
   323: class CustomPlanner(Planner):
   324:     """Custom planning algorithm for JEPA world models.
   325: 
   326:     Uses the learned JEPA world model to search for optimal action
   327:     sequences that reach a specified goal state.
   328: 
   329:     Available methods (inherited from Planner):
   330:         self.unroll(obs_init, actions): Forward simulate actions through world model
   331:             obs_init: [1, C, 1, H, W] initial observation encoding
   332:             actions: [B, A, T] action sequences to evaluate
   333:             Returns: [B, D, T+1, H, W] predicted state encodings
   334: 
   335:         self.objective(encodings): Compute cost for predicted encodings
   336:             encodings: [B, D, T, H, W]
   337:             Returns: [B] cost per sample (lower is better)
   338: 
   339:         self.cost_function(actions, obs_init): Convenience method
   340:             Calls unroll then objective
   341:             Returns: [B] cost per sample
   342: 
   343:     plan() must return PlanningResult(actions=Tensor[T, A], ...)
   344:     """
   345: 
   346:     def __init__(self, unroll, action_dim=2, plan_length=15,
   347:                  num_samples=200, n_iters=20, **kwargs):
   348:         super().__init__(unroll)
   349:         self.action_dim = action_dim
   350:         self.plan_length = plan_length
   351:         self.num_samples = num_samples
   352:         self.n_iters = n_iters
   353:         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   354: 
   355:     @torch.no_grad()
   356:     def plan(self, obs_init, steps_left=None, eval_mode=True,
   357:              t0=False, plan_vis_path=None):
   358:         plan_length = min(self.plan_length, steps_left) if steps_left else self.plan_length
   359:         # TODO: Implement your planning algorithm here.
   360:         # You have access to:
   361:         #   self.unroll(obs_init, actions) - forward simulate through world model
   362:         #   self.objective(encodings)      - compute cost (lower = better)
   363:         #   self.cost_function(actions, obs_init) - convenience: unroll + objective
   364:         #
   365:         # Return PlanningResult with action sequence of shape [plan_length, action_dim]
   366:         actions = torch.zeros(plan_length, self.action_dim, device=self.device)
   367:         return PlanningResult(actions=actions)
   368: # EDITABLE REGION END
   369: 
   370: 
   371: # ============================================================================
   372: # PART 3: Planning Evaluation
   373: # ============================================================================
   374: 
   375: def create_env(data_config):
   376:     """Create the Two Rooms evaluation environment."""
   377:     if ENV_NAME != "two_rooms":
   378:         raise ValueError(f"Unsupported ENV='{TASK_ENV}' for jepa-planning")
   379:     return DotWall(
   380:         config=data_config,
   381:         n_allowed_steps=200,
   382:         level="normal",
   383:     )
   384: 
   385: 
   386: def run_planning_eval(jepa, xy_prober, loader, device, num_episodes=20):
   387:     """Run planning evaluation with the CustomPlanner."""
   388:     jepa.eval()
   389: 
   390:     data_config = loader.dataset.config
   391:     env = create_env(data_config)
   392:     reset_env(env, SEED)
   393: 
   394:     # Create a lightweight GCAgent-like wrapper that uses CustomPlanner
   395:     normalizer = env.normalizer
   396: 
   397:     # Create the planner
   398:     planner = CustomPlanner(
   399:         unroll=None,  # Will be set via agent wrapper
   400:         action_dim=2,
   401:         plan_length=PLAN_LENGTH,
   402:         num_samples=200,
   403:         n_iters=20,
   404:     )
   405: 
   406:     # We need to wire up the unroll function properly
   407:     # The planner needs access to model unroll through the GCAgent's unroll method
   408:     class PlanningAgent:
   409:         def __init__(self, model, planner, normalizer, env, prober):
   410:             self.model = model
   411:             self.planner = planner
   412:             self.normalizer = normalizer
   413:             self.env = env
   414:             self.device = next(model.parameters()).device
   415:             self.loc_prober = prober
   416:             self.goal_state = None
   417:             self.goal_position = None
   418:             self.goal_state_enc = None
   419:             self.objective = None
   420:             self.num_act_stepped = 1
   421: 
   422:             # Wire planner's unroll to agent's unroll
   423:             self.planner.unroll = self.unroll
   424: 
   425:         def unroll(self, obs_init, actions, repeat_batch=True):
   426:             batch_size = actions.shape[0]
   427:             nsteps = actions.shape[2]
   428:             if repeat_batch:
   429:                 obs_init_rep = obs_init.repeat(batch_size, 1, 1, 1, 1)
   430:             else:
   431:                 obs_init_rep = obs_init
   432:             predicted_states, _ = self.model.unroll(
   433:                 obs_init_rep, actions,
   434:                 nsteps=nsteps,
   435:                 unroll_mode="autoregressive",
   436:                 ctxt_window_time=1,
   437:                 compute_loss=False,
   438:                 return_all_steps=False,
   439:             )
   440:             return predicted_states
   441: 
   442:         def set_goal(self, goal_state, goal_position=None):
   443:             self.goal_position = goal_position
   444:             self.goal_state = goal_state
   445:             self.goal_state_enc = self.model.encode(
   446:                 self.normalizer.normalize_state(goal_state.to(self.device))
   447:                 .unsqueeze(0)
   448:                 .unsqueeze(2)
   449:             )
   450:             self.objective = ReprTargetDistMPCObjective(
   451:                 target_enc=self.goal_state_enc,
   452:                 sum_all_diffs=True,
   453:             )
   454:             self.planner.set_objective(self.objective)
   455: 
   456:         def act(self, obs_tensor, steps_left=None, t0=False):
   457:             planning_result = self.planner.plan(
   458:                 obs_tensor,
   459:                 steps_left=steps_left,
   460:                 eval_mode=True,
   461:                 t0=t0,
   462:             )
   463:             return planning_result.actions[:self.num_act_stepped]
   464: 
   465:     agent = PlanningAgent(jepa, planner, normalizer, env, xy_prober)
   466: 
   467:     successes = []
   468:     distances = []
   469:     steps_to_success = []
   470: 
   471:     for ep in range(num_episodes):
   472:         obs, info = reset_env(env, SEED + ep)
   473:         obs, reward, done, truncated, info = env.step(
   474:             np.zeros(env.action_space.shape[0])
   475:         )
   476:         goal_img = info["target_obs"]
   477: 
   478:         agent.set_goal(
   479:             goal_img.detach().clone().to(dtype=torch.float32),
   480:             info["target_position"],
   481:         )
   482: 
   483:         steps_left = env.n_allowed_steps
   484:         total_steps = env.n_allowed_steps
   485:         t0 = True
   486:         success = False
   487:         state_dist = float("inf")
   488:         first_success_step = None
   489: 
   490:         while steps_left > 0:
   491:             obs_tensor = (
   492:                 normalizer.normalize_state(
   493:                     obs.detach().clone().to(dtype=torch.float32, device=device)
   494:                 )
   495:                 .unsqueeze(0)
   496:                 .unsqueeze(2)
   497:             )
   498:             with torch.no_grad():
   499:                 action = agent.act(
   500:                     obs_tensor, steps_left=steps_left, t0=t0

[truncated: showing at most 500 lines / 60000 bytes from eb_jepa/custom_planner.py]
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `random` baseline — editable region  [READ-ONLY — reference implementation]

In `eb_jepa/custom_planner.py`:

```python
Lines 323–361:
   320: # ============================================================================
   321: 
   322: # EDITABLE REGION START
   323: class CustomPlanner(Planner):
   324:     """Random Search planner (lower bound baseline).
   325: 
   326:     Samples random action sequences and returns the one with lowest cost.
   327:     No iterative refinement -- purely single-pass random sampling.
   328:     """
   329: 
   330:     def __init__(self, unroll, action_dim=2, plan_length=15,
   331:                  num_samples=200, n_iters=20, **kwargs):
   332:         super().__init__(unroll)
   333:         self.action_dim = action_dim
   334:         self.plan_length = plan_length
   335:         self.num_samples = num_samples
   336:         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   337: 
   338:     @torch.no_grad()
   339:     def plan(self, obs_init, steps_left=None, eval_mode=True,
   340:              t0=False, plan_vis_path=None):
   341:         from einops import rearrange
   342: 
   343:         plan_length = min(self.plan_length, steps_left) if steps_left else self.plan_length
   344: 
   345:         # Sample random actions
   346:         actions = torch.randn(
   347:             plan_length, self.num_samples, self.action_dim, device=self.device
   348:         )
   349: 
   350:         # Clip action norms
   351:         max_norm = 2.45
   352:         norms = actions.norm(dim=-1, keepdim=True)
   353:         actions = actions * (max_norm / norms.clamp(min=1e-6)).clamp(max=1.0)
   354: 
   355:         # Evaluate all samples and pick the best
   356:         cost = self.cost_function(
   357:             rearrange(actions, "t b a -> b a t"), obs_init
   358:         )
   359:         best_idx = cost.argmin()
   360: 
   361:         return PlanningResult(actions=actions[:, best_idx])
   362: # EDITABLE REGION END
   363: 
   364: 
```

### `cem` baseline — editable region  [READ-ONLY — reference implementation]

In `eb_jepa/custom_planner.py`:

```python
Lines 323–385:
   320: # ============================================================================
   321: 
   322: # EDITABLE REGION START
   323: class CustomPlanner(Planner):
   324:     """CEM (Cross-Entropy Method) planner for JEPA world models."""
   325: 
   326:     def __init__(self, unroll, action_dim=2, plan_length=15,
   327:                  num_samples=200, n_iters=20, **kwargs):
   328:         super().__init__(unroll)
   329:         self.action_dim = action_dim
   330:         self.plan_length = plan_length
   331:         self.num_samples = num_samples
   332:         self.n_iters = n_iters
   333:         self.num_elites = max(10, num_samples // 10)
   334:         self.var_scale = 1.5
   335:         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   336: 
   337:     @torch.no_grad()
   338:     def plan(self, obs_init, steps_left=None, eval_mode=True,
   339:              t0=False, plan_vis_path=None):
   340:         from einops import rearrange
   341: 
   342:         plan_length = min(self.plan_length, steps_left) if steps_left else self.plan_length
   343: 
   344:         mean = torch.zeros(plan_length, self.action_dim, device=self.device)
   345:         std = self.var_scale * torch.ones(plan_length, self.action_dim, device=self.device)
   346:         actions = torch.empty(plan_length, self.num_samples, self.action_dim, device=self.device)
   347: 
   348:         losses = []
   349:         elite_means = []
   350:         elite_stds = []
   351: 
   352:         for _ in range(self.n_iters):
   353:             actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
   354:                 plan_length, self.num_samples, self.action_dim, device=self.device,
   355:             )
   356: 
   357:             # Clip action norms
   358:             max_norm = 2.45
   359:             eps = 1e-6
   360:             norms = actions.norm(dim=-1, keepdim=True)
   361:             max_norms = torch.ones_like(norms) * max_norm
   362:             min_norms = torch.ones_like(norms) * 0
   363:             coeff = torch.min(torch.max(norms, min_norms), max_norms) / (norms + eps)
   364:             actions = actions * coeff
   365: 
   366:             cost = self.cost_function(
   367:                 rearrange(actions, "t b a -> b a t"), obs_init
   368:             ).unsqueeze(1)
   369:             losses.append(cost.min().item())
   370: 
   371:             elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
   372:             elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]
   373: 
   374:             elite_means.append(elite_loss.mean().item())
   375:             elite_stds.append(elite_loss.std().item())
   376: 
   377:             mean = torch.mean(elite_actions, dim=1)
   378:             std = torch.std(elite_actions, dim=1)
   379: 
   380:         return PlanningResult(
   381:             actions=mean,
   382:             losses=torch.tensor(losses).detach().unsqueeze(-1),
   383:             prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
   384:             prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
   385:         )
   386: # EDITABLE REGION END
   387: 
   388: 
```

### `mppi` baseline — editable region  [READ-ONLY — reference implementation]

In `eb_jepa/custom_planner.py`:

```python
Lines 323–401:
   320: # ============================================================================
   321: 
   322: # EDITABLE REGION START
   323: class CustomPlanner(Planner):
   324:     """MPPI (Model Predictive Path Integral) planner for JEPA world models."""
   325: 
   326:     def __init__(self, unroll, action_dim=2, plan_length=15,
   327:                  num_samples=200, n_iters=20, **kwargs):
   328:         super().__init__(unroll)
   329:         self.action_dim = action_dim
   330:         self.plan_length = plan_length
   331:         self.num_samples = num_samples
   332:         self.n_iters = n_iters
   333:         # Match upstream MPPIPlanner defaults — planning_mppi.yaml sets
   334:         # var_scale=1.5 but MPPIPlanner doesn't accept that kwarg, so the
   335:         # effective config is max_std=2 (class default), temperature=0.005,
   336:         # num_elites=20 (yaml). Mirror that here.
   337:         self.num_elites = max(20, num_samples // 10)
   338:         self.max_std = 2.0
   339:         self.temperature = 0.005
   340:         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   341: 
   342:     @torch.no_grad()
   343:     def plan(self, obs_init, steps_left=None, eval_mode=True,
   344:              t0=False, plan_vis_path=None):
   345:         from einops import rearrange
   346: 
   347:         plan_length = min(self.plan_length, steps_left) if steps_left else self.plan_length
   348: 
   349:         mean = torch.zeros(plan_length, self.action_dim, device=self.device)
   350:         std = self.max_std * torch.ones(plan_length, self.action_dim, device=self.device)
   351:         actions = torch.empty(plan_length, self.num_samples, self.action_dim, device=self.device)
   352: 
   353:         losses = []
   354:         elite_means = []
   355:         elite_stds = []
   356: 
   357:         for _ in range(self.n_iters):
   358:             actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
   359:                 plan_length, self.num_samples, self.action_dim, device=self.device,
   360:             )
   361: 
   362:             cost = self.cost_function(
   363:                 rearrange(actions, "t b a -> b a t"), obs_init
   364:             ).unsqueeze(1)
   365:             losses.append(cost.min().item())
   366: 
   367:             elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
   368:             elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]
   369: 
   370:             elite_means.append(elite_loss.mean().item())
   371:             elite_stds.append(elite_loss.std().item())
   372: 
   373:             # MPPI weighted update
   374:             min_cost = cost.min(0)[0]
   375:             score = torch.exp(
   376:                 self.temperature * (min_cost - elite_loss[:, 0])
   377:             )
   378:             score /= score.sum(0) + 1e-9
   379:             mean = torch.sum(
   380:                 score.unsqueeze(0).unsqueeze(2) * elite_actions, dim=1
   381:             )
   382:             std = torch.sqrt(
   383:                 torch.sum(
   384:                     score.unsqueeze(0).unsqueeze(2)
   385:                     * (elite_actions - mean.unsqueeze(1)) ** 2,
   386:                     dim=1,
   387:                 )
   388:             )
   389: 
   390:         # Select action via weighted sampling
   391:         score_np = score.cpu().numpy()
   392:         selected = elite_actions[
   393:             :, np.random.choice(np.arange(score_np.shape[0]), p=score_np)
   394:         ]
   395: 
   396:         return PlanningResult(
   397:             actions=selected,
   398:             losses=torch.tensor(losses).detach().unsqueeze(-1),
   399:             prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
   400:             prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
   401:         )
   402: # EDITABLE REGION END
   403: 
   404: 
```

### `icem` baseline — editable region  [READ-ONLY — reference implementation]

In `eb_jepa/custom_planner.py`:

```python
Lines 323–502:
   320: # ============================================================================
   321: 
   322: # EDITABLE REGION START
   323: class CustomPlanner(Planner):
   324:     """iCEM (Pinneri et al., CoRL 2020 / PMLR 2021) - colored noise + elite reuse + momentum."""
   325: 
   326:     def __init__(self, unroll, action_dim=2, plan_length=15,
   327:                  num_samples=900, n_iters=10, **kwargs):
   328:         super().__init__(unroll)
   329:         self.action_dim = action_dim
   330:         self.plan_length = plan_length
   331:         # The task harness passes CEM/MPPI defaults (200, 20). Override them
   332:         # here to match Pinneri et al. Table S4's 4000-budget iCEM setting:
   333:         # 900 samples, 10 iterations, K=10%, gamma=1.25 -> ~4101 fresh
   334:         # trajectories after decay.
   335:         self.num_samples = 900
   336:         self.n_iters = 10
   337:         self.elites_size = max(10, self.num_samples // 10)
   338:         # iCEM paper / reference-code defaults
   339:         self.fraction_elites_reused = 0.3
   340:         self.factor_decrease_num = 1.25
   341:         self.alpha = 0.1              # momentum smoothing factor
   342:         self.noise_beta = 2.5         # learned-model PlaNet setting
   343:         self.var_scale = 1.5
   344:         self.max_norm = 2.45
   345:         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   346:         # Persistent state for across-env-step shift (paper Sec 3.2 / Fig S9).
   347:         # The Planner instance lives across all plan() calls within
   348:         # an evaluation; t0=True at the start of each episode triggers reset.
   349:         self._mean = None
   350:         self._std = None
   351:         self._kept_elites = None
   352: 
   353:     def _colored_noise(self, T, N, A):
   354:         # rFFT-based 1/f^noise_beta Gaussian, matching
   355:         # colorednoise.powerlaw_psd_gaussian (used by the iCEM repo).
   356:         if T <= 1:
   357:             return torch.randn(T, N, A, device=self.device)
   358:         freqs = torch.fft.rfftfreq(T, device=self.device)
   359:         # Avoid the DC zero-frequency blow-up the same way colorednoise does:
   360:         freqs = freqs.clone()
   361:         freqs[0] = 1.0 / T
   362:         s_scale = freqs ** (-self.noise_beta / 2.0)
   363:         w = s_scale[1:].clone()
   364:         if T % 2 == 0:
   365:             w[-1] = w[-1] / 2
   366:         sigma = 2.0 * torch.sqrt((w ** 2).sum()) / T
   367:         # Random amplitudes in frequency domain, one spectrum per (N, A) trajectory.
   368:         nf = s_scale.shape[0]
   369:         sr = torch.randn(N, A, nf, device=self.device) * s_scale
   370:         si = torch.randn(N, A, nf, device=self.device) * s_scale
   371:         si[..., 0] = 0.0
   372:         # The colorednoise.powerlaw_psd_gaussian reference compensates for
   373:         # the dropped imaginary parts at DC (and Nyquist when T is even) by
   374:         # scaling the corresponding real parts by sqrt(2). Without this fix
   375:         # the DC drift component is ~0.71x of the reference, weakening the
   376:         # long-trajectory directional bias colored noise is meant to provide.
   377:         sr[..., 0] = sr[..., 0] * (2 ** 0.5)
   378:         if T % 2 == 0:
   379:             si[..., -1] = 0.0
   380:             sr[..., -1] = sr[..., -1] * (2 ** 0.5)
   381:         spec = torch.complex(sr, si)
   382:         noise = torch.fft.irfft(spec, n=T, dim=-1) / sigma
   383:         # Return as [T, N, A] to match the CEM baseline's sampling convention.
   384:         return noise.permute(2, 0, 1)
   385: 
   386:     @torch.no_grad()
   387:     def plan(self, obs_init, steps_left=None, eval_mode=True,
   388:              t0=False, plan_vis_path=None):
   389:         from einops import rearrange
   390: 
   391:         plan_length = min(self.plan_length, steps_left) if steps_left else self.plan_length
   392: 
   393:         # iCEM "shift" mechanism (paper Sec 3.2 / icem.py:165-175). At episode
   394:         # start (t0=True) or when plan_length changes, reset to fresh state.
   395:         # Otherwise shift mean/kept-elites left by 1 timestep. Per Alg. 1 /
   396:         # Suppl. E.1, repeat the last mean timestep and reset std everywhere
   397:         # to sigma_init.
   398:         need_reset = (t0 or self._mean is None
   399:                       or self._mean.shape[0] != plan_length)
   400:         if need_reset:
   401:             mean = torch.zeros(plan_length, self.action_dim, device=self.device)
   402:             std = self.var_scale * torch.ones(plan_length, self.action_dim, device=self.device)
   403:             prev_elites = None
   404:         else:
   405:             mean = torch.empty_like(self._mean)
   406:             mean[:-1] = self._mean[1:]
   407:             mean[-1] = self._mean[-1]
   408:             std = self.var_scale * torch.ones_like(self._std)
   409:             # Shift kept elites: drop the executed timestep and append a
   410:             # freshly sampled final timestep from the reset distribution.
   411:             if self._kept_elites is not None:
   412:                 ke = torch.zeros_like(self._kept_elites)
   413:                 ke[:-1] = self._kept_elites[1:]
   414:                 K = self._kept_elites.shape[1]
   415:                 last_noise = self._colored_noise(1, K, self.action_dim)[0]
   416:                 ke[-1] = mean[-1] + std[-1] * last_noise
   417:                 prev_elites = ke
   418:             else:
   419:                 prev_elites = None
   420: 
   421:         best_actions = None
   422:         best_cost = torch.tensor(float("inf"), device=self.device)
   423:         losses, elite_means, elite_stds = [], [], []
   424: 
   425:         num_sim_traj = self.num_samples
   426:         prev_iter_elites = None
   427:         prev_iter_cost = None
   428: 
   429:         for i in range(self.n_iters):
   430:             # Sample decay per iCEM reference:
   431:             # num_sim_traj = max(elites_size * 2, num_sim_traj / factor_decrease_num)
   432:             if i > 0:
   433:                 num_sim_traj = max(self.elites_size * 2,
   434:                                    int(num_sim_traj / self.factor_decrease_num))
   435: 
   436:             n_reused = int(self.elites_size * self.fraction_elites_reused)
   437: 
   438:             noise = self._colored_noise(plan_length, num_sim_traj, self.action_dim)
   439:             actions = mean.unsqueeze(1) + std.unsqueeze(1) * noise
   440:             reused_cost = None
   441:             if i == 0 and prev_elites is not None and n_reused > 0:
   442:                 actions = torch.cat([actions, prev_elites[:, :n_reused]], dim=1)
   443:             elif i > 0 and prev_iter_elites is not None and n_reused > 0:
   444:                 actions = torch.cat([actions, prev_iter_elites[:, :n_reused]], dim=1)
   445:                 reused_cost = prev_iter_cost[:n_reused]
   446: 
   447:             # Clip action norms (consistent with CEM baseline and planning_cem.yaml)
   448:             norms = actions.norm(dim=-1, keepdim=True)
   449:             coeff = (self.max_norm / norms.clamp(min=1e-6)).clamp(max=1.0)
   450:             actions = actions * coeff
   451: 
   452:             if reused_cost is None:
   453:                 cost = self.cost_function(
   454:                     rearrange(actions, "t b a -> b a t"), obs_init
   455:                 )
   456:             else:
   457:                 fresh_actions = actions[:, :num_sim_traj]
   458:                 fresh_cost = self.cost_function(
   459:                     rearrange(fresh_actions, "t b a -> b a t"), obs_init
   460:                 )
   461:                 cost = torch.cat([fresh_cost, reused_cost], dim=0)
   462:             losses.append(cost.min().item())
   463: 
   464:             # Best-so-far across all iterations (iCEM "memory" within a plan).
   465:             min_idx = cost.argmin()
   466:             if cost[min_idx] < best_cost:
   467:                 best_cost = cost[min_idx]
   468:                 best_actions = actions[:, min_idx].clone()
   469: 
   470:             elite_idxs = torch.topk(-cost, self.elites_size, dim=0).indices
   471:             elite_actions = actions[:, elite_idxs]
   472:             elite_cost = cost[elite_idxs]
   473:             elite_means.append(elite_cost.mean().item())
   474:             elite_stds.append(elite_cost.std().item())
   475: 
   476:             # Momentum update of mean/std (alpha=0.1 per iCEM reference).
   477:             new_mean = elite_actions.mean(dim=1)
   478:             new_std = elite_actions.std(dim=1)
   479:             mean = (1.0 - self.alpha) * new_mean + self.alpha * mean
   480:             std = (1.0 - self.alpha) * new_std + self.alpha * std
   481: 
   482:             prev_iter_elites = elite_actions.detach()
   483:             prev_iter_cost = elite_cost.detach()
   484: 
   485:         # Return best-ever trajectory if it beats the final mean (iCEM keeps best).
   486:         final_mean_cost = self.cost_function(
   487:             rearrange(mean.unsqueeze(1), "t b a -> b a t"), obs_init
   488:         )[0]
   489:         out = best_actions if best_cost < final_mean_cost else mean
   490: 
   491:         # Save state for the next env step's shift (paper Sec 3.2).
   492:         # prev_iter_elites is the last iteration's elite_actions tensor [T, K, A].
   493:         self._mean = mean.detach()
   494:         self._std = std.detach()
   495:         self._kept_elites = prev_iter_elites.detach() if prev_iter_elites is not None else None
   496: 
   497:         return PlanningResult(
   498:             actions=out,
   499:             losses=torch.tensor(losses).detach().unsqueeze(-1),
   500:             prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
   501:             prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
   502:         )
   503: # EDITABLE REGION END
   504: 
   505: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
