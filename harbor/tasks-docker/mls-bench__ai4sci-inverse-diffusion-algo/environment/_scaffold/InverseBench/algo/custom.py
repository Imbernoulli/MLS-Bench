import torch
from tqdm import tqdm
from algo.base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler
import numpy as np


class Custom(Algo):
    """Custom algorithm for solving inverse problems with diffusion priors.

    Available utilities:
        - self.net: pre-trained diffusion model.
            - self.net(x, sigma) returns denoised estimate (Tweedie's formula).
            - self.net.img_channels, self.net.img_resolution: image shape info.
        - self.forward_op: forward operator A of the inverse problem.
            - self.forward_op.forward(x): compute A(x).
            - self.forward_op.gradient(x, y, return_loss=True): gradient of ||A(x)-y||^2
              w.r.t. x, returns (grad, loss_value).
            - self.forward_op.loss(x, y): ||A(x)-y||^2, shape (batch,).
            - self.forward_op.device: device of the operator.
        - Scheduler: noise schedule for diffusion process.
            Scheduler(num_steps, schedule, timestep, scaling, sigma_max, sigma_min, ...)
            Properties: .sigma_steps, .factor_steps, .scaling_factor, .scaling_steps, .num_steps
        - DiffusionSampler: unconditional diffusion sampler.
            DiffusionSampler(scheduler).sample(model, x_start, SDE=False) -> denoised x.

    Args (from config):
        diffusion_scheduler_config: dict for Scheduler constructor.
        guidance_scale: float, step size for measurement guidance.
        sde: bool, whether to use SDE (stochastic) or ODE (deterministic) sampling.
        num_optim_steps: int, number of optimization steps (for optimization-based methods).
        observation_weight: float, weight for data fidelity term.
        base_lambda: float, regularization strength.
        base_lr: float, learning rate for optimization-based methods.
        num_mc_samples: int, number of MC samples for gradient estimation.
    """

    def __init__(self, net, forward_op,
                 diffusion_scheduler_config=None,
                 guidance_scale=10.0,
                 sde=True,
                 num_optim_steps=1000,
                 observation_weight=1.0,
                 base_lambda=0.25,
                 base_lr=0.5,
                 num_mc_samples=10,
                 **kwargs):
        super(Custom, self).__init__(net, forward_op)
        # TODO: Initialize your algorithm components here.
        # Store any hyperparameters and create schedulers/optimizers as needed.
        # Example: self.scheduler = Scheduler(**diffusion_scheduler_config)
        pass

    def inference(self, observation, num_samples=1, **kwargs):
        """Solve the inverse problem: given observation y, recover x such that A(x) ≈ y.

        Args:
            observation: measured data y = A(x_true) + noise.
            num_samples: number of reconstruction samples to generate.

        Returns:
            x_recon: reconstructed signal, shape (num_samples, C, H, W).

        TODO: Implement your inverse problem solving algorithm here.
        You should use the diffusion prior (self.net) and the forward operator
        (self.forward_op) to reconstruct x from the observation y.
        """
        device = self.forward_op.device
        # Initialize from random noise
        x = torch.randn(num_samples, self.net.img_channels,
                         self.net.img_resolution, self.net.img_resolution,
                         device=device)
        return x
