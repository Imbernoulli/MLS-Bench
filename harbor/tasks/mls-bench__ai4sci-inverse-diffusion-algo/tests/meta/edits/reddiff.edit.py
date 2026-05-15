"""REDDiff baseline — rigorous codebase edit ops.
Replaces entire custom.py with Regularization by Denoising implementation.
Reference: algo/reddiff.py (Mardani et al., 2023)
"""

_FILE = "InverseBench/algo/custom.py"

_CONTENT = """\
import os
import torch
import tqdm
from algo.base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler
import numpy as np


class Custom(Algo):
    \"\"\"REDDiff: Regularization by Denoising with Diffusion priors.
    Optimization-based approach using diffusion score as regularizer.
    \"\"\"

    # Per-problem task-local hyperparameters, initialized from InverseBench-style
    # inverse problem settings and then adjusted for this benchmark harness.
    # 'inpainting' is intentionally omitted: the default __init__ values
    # (observation_weight=1500, base_lr=0.04, base_lambda=5e-4) already work
    # well on FFHQ256 box-inpaint (REDDiff achieves PSNR~22 with these), so
    # adding an override here would only risk regressing the result.
    PROBLEM_CONFIGS = {
        'inv-scatter': {'observation_weight': 1500.0, 'base_lr': 0.04, 'base_lambda': 5e-4},
        'blackhole': {'observation_weight': 1e-4, 'base_lr': 1e-2, 'base_lambda': 0.25},
    }

    def __init__(self, net, forward_op,
                 num_steps=1000,
                 observation_weight=1500.0,
                 base_lambda=5e-4,
                 base_lr=0.04,
                 lambda_scheduling_type='constant',
                 **kwargs):
        super(Custom, self).__init__(net, forward_op)
        # Apply per-problem overrides
        env = os.environ.get('ENV', '')
        if env in self.PROBLEM_CONFIGS:
            cfg = self.PROBLEM_CONFIGS[env]
            observation_weight = cfg.get('observation_weight', observation_weight)
            base_lr = cfg.get('base_lr', base_lr)
            base_lambda = cfg.get('base_lambda', base_lambda)
            num_steps = cfg.get('num_steps', num_steps)
        self.net.eval().requires_grad_(False)

        self.scheduler = Scheduler(
            num_steps=num_steps, schedule='vp',
            timestep='vp', scaling='vp'
        )
        self.base_lr = base_lr
        self.observation_weight = observation_weight
        if lambda_scheduling_type == 'linear':
            self.lambda_fn = lambda sigma: sigma * base_lambda
        elif lambda_scheduling_type == 'sqrt':
            self.lambda_fn = lambda sigma: torch.sqrt(sigma) * base_lambda
        elif lambda_scheduling_type == 'constant':
            self.lambda_fn = lambda sigma: base_lambda
        else:
            raise NotImplementedError

    def pred_epsilon(self, model, x, sigma):
        sigma = torch.as_tensor(sigma).to(x.device)
        d = model(x, sigma)
        return (x - d) / sigma

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        num_steps = self.scheduler.num_steps
        pbar = tqdm.trange(num_steps)
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)

        mu = torch.zeros(
            num_samples, self.net.img_channels,
            self.net.img_resolution, self.net.img_resolution,
            device=device
        ).requires_grad_(True)
        optimizer = torch.optim.Adam([mu], lr=self.base_lr, betas=(0.9, 0.99))

        for step in pbar:
            with torch.no_grad():
                sigma = self.scheduler.sigma_steps[step]
                scaling = self.scheduler.scaling_steps[step]
                epsilon = torch.randn_like(mu)
                xt = scaling * (mu + sigma * epsilon)
                pred_epsilon = self.pred_epsilon(self.net, xt, sigma).detach()

            lam = self.lambda_fn(sigma)
            optimizer.zero_grad()

            gradient, loss_scale = self.forward_op.gradient(
                mu, observation, return_loss=True
            )
            gradient = (gradient * self.observation_weight
                        + lam * (pred_epsilon - epsilon))
            mu.grad = gradient

            optimizer.step()
            pbar.set_description(
                f'Iteration {step + 1}/{num_steps}. '
                f'Data fitting loss: {torch.sqrt(loss_scale)}'
            )
        return mu
"""

_YAML_FILE = "InverseBench/configs/algorithm/custom.yaml"
_YAML_CONTENT = """\
name: Custom
method:
  _target_: algo.custom.Custom
  num_steps: 1000
  observation_weight: 1500.0
  base_lr: 0.04
  base_lambda: 5e-4
  lambda_scheduling_type: 'constant'
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 1,
        "end_line": 74,
        "content": _CONTENT,
    },
    {
        "op": "replace",
        "file": _YAML_FILE,
        "start_line": 1,
        "end_line": 100,
        "content": _YAML_CONTENT,
    },
]
