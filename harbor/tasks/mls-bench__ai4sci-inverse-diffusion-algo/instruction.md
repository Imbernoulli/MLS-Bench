# MLS-Bench: ai4sci-inverse-diffusion-algo

# Task: Inverse Problem Algorithm Design with Diffusion Priors

## Research Question
Design a novel algorithm for solving scientific inverse problems using pre-trained diffusion model priors. Given a forward operator A and observation `y = A(x) + noise`, the algorithm should reconstruct `x` by leveraging a learned diffusion prior `p(x)`.

## Background
Diffusion models learn rich priors `p(x)` over signal distributions. For inverse problems, we want to sample from the posterior `p(x|y) ∝ p(y|x) p(x)`. Existing approaches include:

- **DPS — Diffusion Posterior Sampling** (Chung et al., "Diffusion Posterior Sampling for General Noisy Inverse Problems", ICLR 2023; arXiv:2209.14687). Uses the score `∇_x log p(x)` from the diffusion model and adds measurement guidance `∇_x log p(y|x)` at each denoising step. Code: https://github.com/DPS2022/diffusion-posterior-sampling.
- **REDDiff — Variational / Regularization-by-Denoising-Diffusion** (Mardani, Song, Kautz, Vahdat, "A Variational Perspective on Solving Inverse Problems with Diffusion Models", ICLR 2024; arXiv:2305.04391). Variational formulation that yields a regularization-by-denoising update where denoisers at different timesteps concurrently impose structural constraints. Code: https://github.com/NVlabs/RED-diff.
- **LGD — Loss-Guided Diffusion** (Song et al., "Loss-Guided Diffusion Models for Plug-and-Play Controllable Generation", ICML 2023). Estimates the guidance term via Monte Carlo sampling around the denoised estimate to reduce bias of point-estimate approximations.

## What to Implement
Implement the `Custom` class in `algo/custom.py`. You must implement:
1. `__init__`: Set up your algorithm (schedulers, optimizers, hyperparameters).
2. `inference(observation, num_samples)`: Given observation `y`, return reconstructed `x`.

## Available Components
- `self.net(x, sigma)` → denoised estimate (Tweedie's formula: E[x_0 | x_t]).
- `self.forward_op.forward(x)` → compute `A(x)`.
- `self.forward_op.gradient(x, y, return_loss=True)` → `(∇_x ||A(x) - y||², loss)`.
- `self.forward_op.loss(x, y)` → `||A(x) - y||²`.
- `Scheduler(num_steps, schedule, timestep, scaling)` → diffusion noise schedule.
- `DiffusionSampler(scheduler).sample(model, x_start)` → unconditional sampling.

The pretrained denoiser, the forward-operator definitions, and the evaluation problems are fixed; the algorithm only chooses how to combine these pieces.

## Editable Region
The entire `algo/custom.py` file is editable. You may define any helper classes/functions within this file.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/InverseBench/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `InverseBench/algo/custom.py`
- editable: **entire file**


Other files you may **read** for context (do not modify):
- `InverseBench/algo/base.py`
- `InverseBench/utils/scheduler.py`
- `InverseBench/utils/diffusion.py`
- `InverseBench/inverse_problems/base.py`
- `InverseBench/inverse_problems/blackhole.py`
- `InverseBench/inverse_problems/image_restore.py`


## Readable Context


### `InverseBench/algo/custom.py`  [EDITABLE — entire file only]

```python
     1: import torch
     2: from tqdm import tqdm
     3: from algo.base import Algo
     4: from utils.scheduler import Scheduler
     5: from utils.diffusion import DiffusionSampler
     6: import numpy as np
     7: 
     8: 
     9: class Custom(Algo):
    10:     """Custom algorithm for solving inverse problems with diffusion priors.
    11: 
    12:     Available utilities:
    13:         - self.net: pre-trained diffusion model.
    14:             - self.net(x, sigma) returns denoised estimate (Tweedie's formula).
    15:             - self.net.img_channels, self.net.img_resolution: image shape info.
    16:         - self.forward_op: forward operator A of the inverse problem.
    17:             - self.forward_op.forward(x): compute A(x).
    18:             - self.forward_op.gradient(x, y, return_loss=True): gradient of ||A(x)-y||^2
    19:               w.r.t. x, returns (grad, loss_value).
    20:             - self.forward_op.loss(x, y): ||A(x)-y||^2, shape (batch,).
    21:             - self.forward_op.device: device of the operator.
    22:         - Scheduler: noise schedule for diffusion process.
    23:             Scheduler(num_steps, schedule, timestep, scaling, sigma_max, sigma_min, ...)
    24:             Properties: .sigma_steps, .factor_steps, .scaling_factor, .scaling_steps, .num_steps
    25:         - DiffusionSampler: unconditional diffusion sampler.
    26:             DiffusionSampler(scheduler).sample(model, x_start, SDE=False) -> denoised x.
    27: 
    28:     Args (from config):
    29:         diffusion_scheduler_config: dict for Scheduler constructor.
    30:         guidance_scale: float, step size for measurement guidance.
    31:         sde: bool, whether to use SDE (stochastic) or ODE (deterministic) sampling.
    32:         num_optim_steps: int, number of optimization steps (for optimization-based methods).
    33:         observation_weight: float, weight for data fidelity term.
    34:         base_lambda: float, regularization strength.
    35:         base_lr: float, learning rate for optimization-based methods.
    36:         num_mc_samples: int, number of MC samples for gradient estimation.
    37:     """
    38: 
    39:     def __init__(self, net, forward_op,
    40:                  diffusion_scheduler_config=None,
    41:                  guidance_scale=10.0,
    42:                  sde=True,
    43:                  num_optim_steps=1000,
    44:                  observation_weight=1.0,
    45:                  base_lambda=0.25,
    46:                  base_lr=0.5,
    47:                  num_mc_samples=10,
    48:                  **kwargs):
    49:         super(Custom, self).__init__(net, forward_op)
    50:         # TODO: Initialize your algorithm components here.
    51:         # Store any hyperparameters and create schedulers/optimizers as needed.
    52:         # Example: self.scheduler = Scheduler(**diffusion_scheduler_config)
    53:         pass
    54: 
    55:     def inference(self, observation, num_samples=1, **kwargs):
    56:         """Solve the inverse problem: given observation y, recover x such that A(x) ≈ y.
    57: 
    58:         Args:
    59:             observation: measured data y = A(x_true) + noise.
    60:             num_samples: number of reconstruction samples to generate.
    61: 
    62:         Returns:
    63:             x_recon: reconstructed signal, shape (num_samples, C, H, W).
    64: 
    65:         TODO: Implement your inverse problem solving algorithm here.
    66:         You should use the diffusion prior (self.net) and the forward operator
    67:         (self.forward_op) to reconstruct x from the observation y.
    68:         """
    69:         device = self.forward_op.device
    70:         # Initialize from random noise
    71:         x = torch.randn(num_samples, self.net.img_channels,
    72:                          self.net.img_resolution, self.net.img_resolution,
    73:                          device=device)
    74:         return x
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `dps` baseline — editable region  [READ-ONLY — reference implementation]

In `InverseBench/algo/custom.py`:

```python
     1: import os
     2: import torch
     3: from tqdm import tqdm
     4: from algo.base import Algo
     5: from utils.scheduler import Scheduler
     6: from utils.diffusion import DiffusionSampler
     7: import numpy as np
     8: 
     9: 
    10: class Custom(Algo):
    11:     """DPS: Diffusion Posterior Sampling.
    12:     Score-based guidance using the gradient of the data likelihood.
    13:     Requires forward_op.gradient() — best for differentiable forward operators.
    14:     """
    15: 
    16:     # Per-problem optimized hyperparameters
    17:     # inv-scatter: linear forward op, gradient clean → high guidance works
    18:     # navier-stokes: PDE solver forward op, gradient VERY noisy/unstable →
    19:     #   low guidance + gradient clipping to prevent NaN divergence
    20:     # blackhole: non-trivial forward op → moderate guidance
    21:     # clip_grad: ONLY enable for problems where the forward-op gradient is
    22:     #   numerically unstable (e.g. NS PDE solver producing NaN). Clipping the
    23:     #   raw ll_grad norm before the 1/sqrt(loss_scale) rescaling changes the
    24:     #   effective guidance scale, so leave it OFF for well-behaved problems
    25:     #   (inv-scatter / blackhole) to preserve their tuned guidance_scale.
    26:     PROBLEM_CONFIGS = {
    27:         'inv-scatter': {'guidance_scale': 50.0, 'clip_grad': False},
    28:         'blackhole': {'guidance_scale': 1e-3, 'clip_grad': False},
    29:         # FFHQ256 box-inpaint with sigma_noise=0.05. The default 50.0 (tuned for
    30:         # the inv-scatter forward op) explodes here because the pixel-domain
    31:         # data fitting loss is much smaller. guidance_scale=1.0 matches typical
    32:         # DPS values for natural-image inverse problems.
    33:         'inpainting': {'guidance_scale': 1.0, 'clip_grad': False},
    34:     }
    35: 
    36:     def __init__(self, net, forward_op,
    37:                  diffusion_scheduler_config=None,
    38:                  guidance_scale=50.0,
    39:                  sde=True,
    40:                  **kwargs):
    41:         super(Custom, self).__init__(net, forward_op)
    42:         # Apply per-problem overrides
    43:         env = os.environ.get('ENV', '')
    44:         self.clip_grad = False
    45:         if env in self.PROBLEM_CONFIGS:
    46:             cfg = self.PROBLEM_CONFIGS[env]
    47:             guidance_scale = cfg.get('guidance_scale', guidance_scale)
    48:             self.clip_grad = cfg.get('clip_grad', False)
    49:         self.scale = guidance_scale
    50:         self.diffusion_scheduler_config = diffusion_scheduler_config or {
    51:             'num_steps': 1000, 'schedule': 'vp', 'timestep': 'vp', 'scaling': 'vp'
    52:         }
    53:         # Override num_steps for expensive problems
    54:         if env in self.PROBLEM_CONFIGS and 'num_steps' in self.PROBLEM_CONFIGS[env]:
    55:             self.diffusion_scheduler_config['num_steps'] = self.PROBLEM_CONFIGS[env]['num_steps']
    56:         self.scheduler = Scheduler(**self.diffusion_scheduler_config)
    57:         self.sde = sde
    58: 
    59:     def inference(self, observation, num_samples=1, **kwargs):
    60:         device = self.forward_op.device
    61:         if num_samples > 1:
    62:             observation = observation.repeat(num_samples, 1, 1, 1)
    63:         x_initial = torch.randn(
    64:             num_samples, self.net.img_channels,
    65:             self.net.img_resolution, self.net.img_resolution,
    66:             device=device
    67:         ) * self.scheduler.sigma_max
    68:         x_next = x_initial
    69:         x_next.requires_grad = True
    70: 
    71:         pbar = tqdm(range(self.scheduler.num_steps))
    72: 
    73:         for i in pbar:
    74:             x_cur = x_next.detach().requires_grad_(True)
    75: 
    76:             sigma = self.scheduler.sigma_steps[i]
    77:             factor = self.scheduler.factor_steps[i]
    78:             scaling_factor = self.scheduler.scaling_factor[i]
    79: 
    80:             denoised = self.net(
    81:                 x_cur / self.scheduler.scaling_steps[i],
    82:                 torch.as_tensor(sigma).to(x_cur.device)
    83:             )
    84:             gradient, loss_scale = self.forward_op.gradient(
    85:                 denoised, observation, return_loss=True
    86:             )
    87: 
    88:             ll_grad = torch.autograd.grad(denoised, x_cur, gradient)[0]
    89:             # Clip gradient to prevent NaN (only needed for NS solver / acoustic);
    90:             # for well-behaved problems (inv-scatter, blackhole) this would
    91:             # corrupt the tuned guidance scale, so the clip is opt-in.
    92:             if self.clip_grad:
    93:                 grad_norm = ll_grad.norm()
    94:                 max_grad_norm = 1.0
    95:                 if grad_norm > max_grad_norm:
    96:                     ll_grad = ll_grad * (max_grad_norm / grad_norm)
    97:             # Always replace NaN/Inf gradients with zero (cheap + safe)
    98:             ll_grad = torch.nan_to_num(ll_grad, nan=0.0, posinf=0.0, neginf=0.0)
    99:             ll_grad = ll_grad * 0.5 / torch.sqrt(loss_scale).clamp(min=1e-6)
   100: 
   101:             score = (
   102:                 (denoised - x_cur / self.scheduler.scaling_steps[i])
   103:                 / sigma ** 2 / self.scheduler.scaling_steps[i]
   104:             )
   105:             pbar.set_description(
   106:                 f'Iteration {i + 1}/{self.scheduler.num_steps}. '
   107:                 f'Data fitting loss: {torch.sqrt(loss_scale)}'
   108:             )
   109: 
   110:             if self.sde:
   111:                 epsilon = torch.randn_like(x_cur)
   112:                 x_next = (x_cur * scaling_factor + factor * score
   113:                           + np.sqrt(factor) * epsilon)
   114:             else:
   115:                 x_next = x_cur * scaling_factor + factor * score * 0.5
   116:             x_next -= ll_grad * self.scale
   117:         return x_next
```

### `reddiff` baseline — editable region  [READ-ONLY — reference implementation]

In `InverseBench/algo/custom.py`:

```python
     1: import os
     2: import torch
     3: import tqdm
     4: from algo.base import Algo
     5: from utils.scheduler import Scheduler
     6: from utils.diffusion import DiffusionSampler
     7: import numpy as np
     8: 
     9: 
    10: class Custom(Algo):
    11:     """REDDiff: Regularization by Denoising with Diffusion priors.
    12:     Optimization-based approach using diffusion score as regularizer.
    13:     """
    14: 
    15:     # Per-problem task-local hyperparameters, initialized from InverseBench-style
    16:     # inverse problem settings and then adjusted for this benchmark harness.
    17:     # 'inpainting' is intentionally omitted: the default __init__ values
    18:     # (observation_weight=1500, base_lr=0.04, base_lambda=5e-4) already work
    19:     # well on FFHQ256 box-inpaint (REDDiff achieves PSNR~22 with these), so
    20:     # adding an override here would only risk regressing the result.
    21:     PROBLEM_CONFIGS = {
    22:         'inv-scatter': {'observation_weight': 1500.0, 'base_lr': 0.04, 'base_lambda': 5e-4},
    23:         'blackhole': {'observation_weight': 1e-4, 'base_lr': 1e-2, 'base_lambda': 0.25},
    24:     }
    25: 
    26:     def __init__(self, net, forward_op,
    27:                  num_steps=1000,
    28:                  observation_weight=1500.0,
    29:                  base_lambda=5e-4,
    30:                  base_lr=0.04,
    31:                  lambda_scheduling_type='constant',
    32:                  **kwargs):
    33:         super(Custom, self).__init__(net, forward_op)
    34:         # Apply per-problem overrides
    35:         env = os.environ.get('ENV', '')
    36:         if env in self.PROBLEM_CONFIGS:
    37:             cfg = self.PROBLEM_CONFIGS[env]
    38:             observation_weight = cfg.get('observation_weight', observation_weight)
    39:             base_lr = cfg.get('base_lr', base_lr)
    40:             base_lambda = cfg.get('base_lambda', base_lambda)
    41:             num_steps = cfg.get('num_steps', num_steps)
    42:         self.net.eval().requires_grad_(False)
    43: 
    44:         self.scheduler = Scheduler(
    45:             num_steps=num_steps, schedule='vp',
    46:             timestep='vp', scaling='vp'
    47:         )
    48:         self.base_lr = base_lr
    49:         self.observation_weight = observation_weight
    50:         if lambda_scheduling_type == 'linear':
    51:             self.lambda_fn = lambda sigma: sigma * base_lambda
    52:         elif lambda_scheduling_type == 'sqrt':
    53:             self.lambda_fn = lambda sigma: torch.sqrt(sigma) * base_lambda
    54:         elif lambda_scheduling_type == 'constant':
    55:             self.lambda_fn = lambda sigma: base_lambda
    56:         else:
    57:             raise NotImplementedError
    58: 
    59:     def pred_epsilon(self, model, x, sigma):
    60:         sigma = torch.as_tensor(sigma).to(x.device)
    61:         d = model(x, sigma)
    62:         return (x - d) / sigma
    63: 
    64:     def inference(self, observation, num_samples=1, **kwargs):
    65:         device = self.forward_op.device
    66:         num_steps = self.scheduler.num_steps
    67:         pbar = tqdm.trange(num_steps)
    68:         if num_samples > 1:
    69:             observation = observation.repeat(num_samples, 1, 1, 1)
    70: 
    71:         mu = torch.zeros(
    72:             num_samples, self.net.img_channels,
    73:             self.net.img_resolution, self.net.img_resolution,
    74:             device=device
    75:         ).requires_grad_(True)
    76:         optimizer = torch.optim.Adam([mu], lr=self.base_lr, betas=(0.9, 0.99))
    77: 
    78:         for step in pbar:
    79:             with torch.no_grad():
    80:                 sigma = self.scheduler.sigma_steps[step]
    81:                 scaling = self.scheduler.scaling_steps[step]
    82:                 epsilon = torch.randn_like(mu)
    83:                 xt = scaling * (mu + sigma * epsilon)
    84:                 pred_epsilon = self.pred_epsilon(self.net, xt, sigma).detach()
    85: 
    86:             lam = self.lambda_fn(sigma)
    87:             optimizer.zero_grad()
    88: 
    89:             gradient, loss_scale = self.forward_op.gradient(
    90:                 mu, observation, return_loss=True
    91:             )
    92:             gradient = (gradient * self.observation_weight
    93:                         + lam * (pred_epsilon - epsilon))
    94:             mu.grad = gradient
    95: 
    96:             optimizer.step()
    97:             pbar.set_description(
    98:                 f'Iteration {step + 1}/{num_steps}. '
    99:                 f'Data fitting loss: {torch.sqrt(loss_scale)}'
   100:             )
   101:         return mu
```

### `lgd` baseline — editable region  [READ-ONLY — reference implementation]

In `InverseBench/algo/custom.py`:

```python
     1: import os
     2: import torch
     3: from tqdm import tqdm
     4: from algo.base import Algo
     5: from utils.scheduler import Scheduler
     6: from utils.diffusion import DiffusionSampler
     7: import numpy as np
     8: 
     9: 
    10: class Custom(Algo):
    11:     """LGD: Loss-Guided Diffusion.
    12:     Uses Monte Carlo gradient estimation for measurement guidance.
    13:     """
    14: 
    15:     # Per-problem task-local hyperparameters, initialized from InverseBench-style
    16:     # inverse problem settings and then adjusted for this benchmark harness.
    17:     # inpainting: ffhq256 box-mask, sigma_noise=0.05. Pixel-domain image prior
    18:     #   has raw loss_scale ~1e3 at early steps so the 3200 default (tuned for
    19:     #   the electromagnetic inv-scatter forward op) diverges immediately. Use
    20:     #   guidance_scale=1.0 in the normalized image range, matching typical
    21:     #   DPS-family values for natural-image inverse problems. num_mc_samples=5
    22:     #   is a conservative middle ground — higher reduces variance but costs
    23:     #   memory linearly in the samples dimension.
    24:     PROBLEM_CONFIGS = {
    25:         'inv-scatter': {'guidance_scale': 3200.0, 'num_mc_samples': 20},
    26:         'navier-stokes': {'guidance_scale': 3e-3, 'num_mc_samples': 3},
    27:         'blackhole': {'guidance_scale': 1e-3, 'num_mc_samples': 5},
    28:         'acoustic': {'guidance_scale': 1.0, 'num_mc_samples': 3, 'num_steps': 100},
    29:         'inpainting': {'guidance_scale': 1.0, 'num_mc_samples': 5},
    30:     }
    31: 
    32:     def __init__(self, net, forward_op,
    33:                  diffusion_scheduler_config=None,
    34:                  guidance_scale=3200.0,
    35:                  num_mc_samples=20,
    36:                  batch_grad=True,
    37:                  sde=True,
    38:                  **kwargs):
    39:         super(Custom, self).__init__(net, forward_op)
    40:         # Apply per-problem overrides
    41:         env = os.environ.get('ENV', '')
    42:         if env in self.PROBLEM_CONFIGS:
    43:             cfg = self.PROBLEM_CONFIGS[env]
    44:             guidance_scale = cfg.get('guidance_scale', guidance_scale)
    45:             num_mc_samples = cfg.get('num_mc_samples', num_mc_samples)
    46:         self.scale = guidance_scale
    47:         self.diffusion_scheduler_config = diffusion_scheduler_config or {
    48:             'num_steps': 1000, 'schedule': 'vp', 'timestep': 'vp', 'scaling': 'vp'
    49:         }
    50:         # Override num_steps for expensive problems
    51:         if env in self.PROBLEM_CONFIGS and 'num_steps' in self.PROBLEM_CONFIGS[env]:
    52:             self.diffusion_scheduler_config['num_steps'] = self.PROBLEM_CONFIGS[env]['num_steps']
    53:         self.scheduler = Scheduler(**self.diffusion_scheduler_config)
    54:         self.sde = sde
    55:         self.num_samples = num_mc_samples
    56:         self.batch_grad = batch_grad
    57: 
    58:     def inference(self, observation, num_samples=1, **kwargs):
    59:         device = self.forward_op.device
    60:         x_initial = torch.randn(
    61:             num_samples, self.net.img_channels,
    62:             self.net.img_resolution, self.net.img_resolution,
    63:             device=device
    64:         ) * self.scheduler.sigma_max
    65:         x_next = x_initial
    66:         x_next.requires_grad = True
    67:         pbar = tqdm(range(self.scheduler.num_steps))
    68: 
    69:         for i in pbar:
    70:             x_cur = x_next.detach().requires_grad_(True)
    71: 
    72:             sigma = self.scheduler.sigma_steps[i]
    73:             factor = self.scheduler.factor_steps[i]
    74:             scaling_factor = self.scheduler.scaling_factor[i]
    75:             rt = sigma / np.sqrt(1 + sigma ** 2)
    76: 
    77:             denoised = self.net(
    78:                 x_cur / self.scheduler.scaling_steps[i],
    79:                 torch.as_tensor(sigma).to(x_cur.device)
    80:             )
    81: 
    82:             samples = denoised + torch.randn(
    83:                 (self.num_samples, *denoised.shape[1:]), device=device
    84:             ) * rt
    85: 
    86:             if self.batch_grad:
    87:                 gradient, loss_scale = self.forward_op.gradient(
    88:                     samples, observation, return_loss=True
    89:                 )
    90:                 avg_loss = loss_scale
    91:             else:
    92:                 gradients = torch.empty(
    93:                     (self.num_samples, *denoised.shape[1:]), device=device
    94:                 )
    95:                 losses = np.empty(self.num_samples)
    96:                 for j in range(self.num_samples):
    97:                     gradient, loss_scale = self.forward_op.gradient(
    98:                         samples[j:j+1], observation, return_loss=True
    99:                     )
   100:                     gradients[j] = gradient
   101:                     losses[j] = loss_scale
   102:                 avg_loss = losses.mean()
   103:                 gradient = gradients
   104: 
   105:             avg_grad = torch.mean(gradient, dim=0, keepdim=True).detach()
   106: 
   107:             ll_grad = torch.autograd.grad(denoised, x_cur, avg_grad)[0]
   108:             ll_grad = ll_grad * 0.5 / torch.sqrt(avg_loss)
   109: 
   110:             score = (
   111:                 (denoised - x_cur / self.scheduler.scaling_steps[i])
   112:                 / sigma ** 2 / self.scheduler.scaling_steps[i]
   113:             )
   114:             pbar.set_description(
   115:                 f'Iteration {i + 1}/{self.scheduler.num_steps}. '
   116:                 f'Data fitting loss: {torch.sqrt(loss_scale)}'
   117:             )
   118: 
   119:             if self.sde:
   120:                 epsilon = torch.randn_like(x_cur)
   121:                 x_next = (x_cur * scaling_factor + factor * score
   122:                           + np.sqrt(factor) * epsilon)
   123:             else:
   124:                 x_next = x_cur * scaling_factor + factor * score * 0.5
   125:             x_next -= ll_grad * self.scale
   126: 
   127:         return x_next
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
