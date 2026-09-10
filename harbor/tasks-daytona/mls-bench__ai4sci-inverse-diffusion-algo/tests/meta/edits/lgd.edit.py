"""LGD baseline — rigorous codebase edit ops.
Replaces entire custom.py with Loss-Guided Diffusion implementation.
Reference: algo/lgd.py (Song et al., 2023)
"""

_FILE = "InverseBench/algo/custom.py"

_CONTENT = """\
import os
import torch
from tqdm import tqdm
from algo.base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler
import numpy as np


class Custom(Algo):
    \"\"\"LGD: Loss-Guided Diffusion.
    Uses Monte Carlo gradient estimation for measurement guidance.
    \"\"\"

    # Per-problem task-local hyperparameters, initialized from InverseBench-style
    # inverse problem settings and then adjusted for this benchmark harness.
    # inpainting: ffhq256 box-mask, sigma_noise=0.05. Pixel-domain image prior
    #   has raw loss_scale ~1e3 at early steps so the 3200 default (tuned for
    #   the electromagnetic inv-scatter forward op) diverges immediately. Use
    #   guidance_scale=1.0 in the normalized image range, matching typical
    #   DPS-family values for natural-image inverse problems. num_mc_samples=5
    #   is a conservative middle ground — higher reduces variance but costs
    #   memory linearly in the samples dimension.
    PROBLEM_CONFIGS = {
        'inv-scatter': {'guidance_scale': 3200.0, 'num_mc_samples': 20},
        'navier-stokes': {'guidance_scale': 3e-3, 'num_mc_samples': 3},
        'blackhole': {'guidance_scale': 1e-3, 'num_mc_samples': 5},
        'acoustic': {'guidance_scale': 1.0, 'num_mc_samples': 3, 'num_steps': 100},
        'inpainting': {'guidance_scale': 1.0, 'num_mc_samples': 5},
    }

    def __init__(self, net, forward_op,
                 diffusion_scheduler_config=None,
                 guidance_scale=3200.0,
                 num_mc_samples=20,
                 batch_grad=True,
                 sde=True,
                 **kwargs):
        super(Custom, self).__init__(net, forward_op)
        # Apply per-problem overrides
        env = os.environ.get('ENV', '')
        if env in self.PROBLEM_CONFIGS:
            cfg = self.PROBLEM_CONFIGS[env]
            guidance_scale = cfg.get('guidance_scale', guidance_scale)
            num_mc_samples = cfg.get('num_mc_samples', num_mc_samples)
        self.scale = guidance_scale
        self.diffusion_scheduler_config = diffusion_scheduler_config or {
            'num_steps': 1000, 'schedule': 'vp', 'timestep': 'vp', 'scaling': 'vp'
        }
        # Override num_steps for expensive problems
        if env in self.PROBLEM_CONFIGS and 'num_steps' in self.PROBLEM_CONFIGS[env]:
            self.diffusion_scheduler_config['num_steps'] = self.PROBLEM_CONFIGS[env]['num_steps']
        self.scheduler = Scheduler(**self.diffusion_scheduler_config)
        self.sde = sde
        self.num_samples = num_mc_samples
        self.batch_grad = batch_grad

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        x_initial = torch.randn(
            num_samples, self.net.img_channels,
            self.net.img_resolution, self.net.img_resolution,
            device=device
        ) * self.scheduler.sigma_max
        x_next = x_initial
        x_next.requires_grad = True
        pbar = tqdm(range(self.scheduler.num_steps))

        for i in pbar:
            x_cur = x_next.detach().requires_grad_(True)

            sigma = self.scheduler.sigma_steps[i]
            factor = self.scheduler.factor_steps[i]
            scaling_factor = self.scheduler.scaling_factor[i]
            rt = sigma / np.sqrt(1 + sigma ** 2)

            denoised = self.net(
                x_cur / self.scheduler.scaling_steps[i],
                torch.as_tensor(sigma).to(x_cur.device)
            )

            samples = denoised + torch.randn(
                (self.num_samples, *denoised.shape[1:]), device=device
            ) * rt

            if self.batch_grad:
                gradient, loss_scale = self.forward_op.gradient(
                    samples, observation, return_loss=True
                )
                avg_loss = loss_scale
            else:
                gradients = torch.empty(
                    (self.num_samples, *denoised.shape[1:]), device=device
                )
                losses = np.empty(self.num_samples)
                for j in range(self.num_samples):
                    gradient, loss_scale = self.forward_op.gradient(
                        samples[j:j+1], observation, return_loss=True
                    )
                    gradients[j] = gradient
                    losses[j] = loss_scale
                avg_loss = losses.mean()
                gradient = gradients

            avg_grad = torch.mean(gradient, dim=0, keepdim=True).detach()

            ll_grad = torch.autograd.grad(denoised, x_cur, avg_grad)[0]
            ll_grad = ll_grad * 0.5 / torch.sqrt(avg_loss)

            score = (
                (denoised - x_cur / self.scheduler.scaling_steps[i])
                / sigma ** 2 / self.scheduler.scaling_steps[i]
            )
            pbar.set_description(
                f'Iteration {i + 1}/{self.scheduler.num_steps}. '
                f'Data fitting loss: {torch.sqrt(loss_scale)}'
            )

            if self.sde:
                epsilon = torch.randn_like(x_cur)
                x_next = (x_cur * scaling_factor + factor * score
                          + np.sqrt(factor) * epsilon)
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5
            x_next -= ll_grad * self.scale

        return x_next
"""

_YAML_FILE = "InverseBench/configs/algorithm/custom.yaml"
_YAML_CONTENT = """\
name: Custom
method:
  _target_: algo.custom.Custom
  diffusion_scheduler_config:
    num_steps: 1000
    schedule: 'vp'
    timestep: 'vp'
    scaling: 'vp'
  num_mc_samples: 20
  guidance_scale: 3200.0
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
