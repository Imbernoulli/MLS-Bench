"""
Based on: https://github.com/crowsonkb/k-diffusion
"""

import numpy as np
import torch
from piq import LPIPS
from tqdm.auto import tqdm
import torch.distributed as dist


from .nn import mean_flat, append_dims, append_zero
from .random_util import BatchedSeedGenerator


class NoiseSchedule:
    def __init__(self):
        raise NotImplementedError

    def get_f_g2(self, t):
        raise NotImplementedError

    def get_alpha_rho(self, t):
        raise NotImplementedError

    def get_abc(self, t):
        alpha_t, alpha_bar_t, rho_t, rho_bar_t = self.get_alpha_rho(t)
        a_t, b_t, c_t = (
            (alpha_bar_t * rho_t**2) / self.rho_T**2,
            (alpha_t * rho_bar_t**2) / self.rho_T**2,
            (alpha_t * rho_bar_t * rho_t) / self.rho_T,
        )
        return a_t, b_t, c_t


class VPNoiseSchedule(NoiseSchedule):
    def __init__(self, beta_d=2, beta_min=0.1):
        self.beta_d, self.beta_min = beta_d, beta_min
        self.alpha_fn = lambda t: np.e ** (-0.5 * beta_min * t - 0.25 * beta_d * t**2)
        self.alpha_T = self.alpha_fn(1)
        self.rho_fn = lambda t: (np.e ** (beta_min * t + 0.5 * beta_d * t**2) - 1).sqrt()
        self.rho_T = self.rho_fn(torch.DoubleTensor([1])).item()

        self.f_fn = lambda t: (-0.5 * beta_min - 0.5 * beta_d * t)
        self.g2_fn = lambda t: (beta_min + beta_d * t)

    def get_f_g2(self, t):
        t = t.to(torch.float64)
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_rho(self, t):
        t = t.to(torch.float64)
        alpha_t = self.alpha_fn(t)
        alpha_bar_t = alpha_t / self.alpha_T
        rho_t = self.rho_fn(t)
        rho_bar_t = (self.rho_T**2 - rho_t**2).sqrt()
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t


class VENoiseSchedule(NoiseSchedule):
    def __init__(self, sigma_max=80.0):
        self.sigma_max = sigma_max
        self.alpha_fn = lambda t: torch.ones_like(t)
        self.alpha_T = 1
        self.rho_fn = lambda t: t
        self.rho_T = sigma_max

        self.f_fn = lambda t: torch.zeros_like(t)
        self.g2_fn = lambda t: 2 * t

    def get_f_g2(self, t):
        t = t.to(torch.float64)
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_rho(self, t):
        t = t.to(torch.float64)
        alpha_t = self.alpha_fn(t)
        alpha_bar_t = alpha_t / self.alpha_T
        rho_t = self.rho_fn(t)
        rho_bar_t = (self.rho_T**2 - rho_t**2).sqrt()
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t


class I2SBNoiseSchedule(NoiseSchedule):
    def __init__(self, n_timestep=1000, beta_min=0.1, beta_max=1.0):
        self.n_timestep, self.linear_start, self.linear_end = (
            n_timestep,
            beta_min / n_timestep,
            beta_max / n_timestep,
        )
        betas = (
            torch.linspace(
                self.linear_start**0.5,
                self.linear_end**0.5,
                n_timestep,
                dtype=torch.float64,
            ).cuda()
            ** 2
        )
        betas = torch.cat(
            [
                betas[: self.n_timestep // 2],
                torch.flip(betas[: self.n_timestep // 2], dims=(0,)),
            ]
        )
        std_fwd = torch.sqrt(torch.cumsum(betas, dim=0))
        std_bwd = torch.sqrt(torch.flip(torch.cumsum(torch.flip(betas, dims=(0,)), dim=0), dims=(0,)))

        self.alpha_fn = lambda t: torch.ones_like(t).float()
        self.alpha_T = 1
        self.rho_fn = lambda t: std_fwd[t]
        self.rho_T = std_fwd[-1]
        self.rho_bar_fn = lambda t: std_bwd[t]

        self.f_fn = lambda t: torch.zeros_like(t).float()
        self.g2_fn = lambda t: betas[t]

    def get_f_g2(self, t):
        t = ((self.n_timestep - 1) * t).round().long()
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_rho(self, t):
        t = ((self.n_timestep - 1) * t).round().long()
        alpha_t = self.alpha_fn(t)
        alpha_bar_t = alpha_t / self.alpha_T
        rho_t = self.rho_fn(t)
        rho_bar_t = self.rho_bar_fn(t)
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t


class PreCond:
    def __init__(self, ns):
        raise NotImplementedError

    def _get_scalings_and_weightings(self, t):
        raise NotImplementedError

    def get_scalings_and_weightings(self, t, ndim):
        c_skip, c_in, c_out, c_noise, weightings = self._get_scalings_and_weightings(t)
        c_skip, c_in, c_out, weightings = [append_dims(item, ndim) for item in [c_skip, c_in, c_out, weightings]]
        return c_skip, c_in, c_out, c_noise, weightings


class I2SBPreCond(PreCond):
    def __init__(self, ns, n_timestep=1000, t0=1e-4, T=1.0):
        self.ns = ns
        self.n_timestep = n_timestep
        self.noise_levels = torch.linspace(t0, T, n_timestep).cuda() * n_timestep

    def _get_scalings_and_weightings(self, t):
        _, _, rho_t, _ = self.ns.get_alpha_rho(t)
        c_skip = torch.ones_like(t)
        c_in = torch.ones_like(t)
        c_out = -rho_t
        c_noise = self.noise_levels[((self.n_timestep - 1) * t).round().long()]
        weightings = 1 / c_out**2
        return c_skip, c_in, c_out, c_noise, weightings


class DDBMPreCond(PreCond):
    def __init__(self, ns, sigma_data, cov_xy):
        self.ns, self.sigma_data, self.cov_xy = ns, sigma_data, cov_xy
        self.sigma_data_end = sigma_data

    def _get_scalings_and_weightings(self, t):
        a_t, b_t, c_t = self.ns.get_abc(t)
        A = a_t**2 * self.sigma_data_end**2 + b_t**2 * self.sigma_data**2 + 2 * a_t * b_t * self.cov_xy + c_t**2
        c_in = 1 / (A) ** 0.5
        c_skip = (b_t * self.sigma_data**2 + a_t * self.cov_xy) / A
        c_out = (
            a_t**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * c_t**2
        ) ** 0.5 * c_in
        c_noise = 1000 * 0.25 * torch.log(t + 1e-44)
        weightings = 1 / c_out**2
        return c_skip, c_in, c_out, c_noise, weightings


class KarrasDenoiser:
    def __init__(
        self,
        noise_schedule,
        precond,
        t_max=1.0,
        t_min=0.0001,
        loss_norm="lpips",
    ):

        self.t_max = t_max
        self.t_min = t_min

        self.noise_schedule = noise_schedule
        self.precond = precond

        self.loss_norm = loss_norm
        if loss_norm == "lpips":
            self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")

    def bridge_sample(self, x0, xT, t, noise):
        a_t, b_t, c_t = [append_dims(item, x0.ndim) for item in self.noise_schedule.get_abc(t)]
        samples = a_t * xT + b_t * x0 + c_t * noise
        return samples

    def denoise(self, model, x_t, t, **model_kwargs):
        c_skip, c_in, c_out, c_noise, weightings = self.precond.get_scalings_and_weightings(t, x_t.ndim)
        model_output = model(c_in * x_t, c_noise, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised, weightings

    def training_bridge_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        assert model_kwargs is not None
        xT = model_kwargs["xT"]
        mask = model_kwargs.pop("mask", None)
        if noise is None:
            noise = torch.randn_like(x_start)
        t = torch.minimum(t, torch.ones_like(t) * self.t_max)
        terms = {}

        x_t = self.bridge_sample(x_start, xT, t, noise)

        _, denoised, weights = self.denoise(model, x_t, t, **model_kwargs)

        if mask is not None:
            terms["xs_mse"] = mean_flat(mask * (denoised - x_start) ** 2)
            terms["mse"] = mean_flat(weights * mask * (denoised - x_start) ** 2)
        else:
            terms["xs_mse"] = mean_flat((denoised - x_start) ** 2)
            terms["mse"] = mean_flat(weights * (denoised - x_start) ** 2)

        terms["loss"] = terms["mse"]

        return terms


def karras_sample(
    diffusion,
    model,
    x_T,
    x_0,
    steps,
    mask=None,
    clip_denoised=True,
    model_kwargs=None,
    device=None,
    rho=7.0,
    sampler="heun",
    churn_step_ratio=0.0,
    eta=0.0,
    order=2,
    seed=None,
):
    assert sampler in [
        "heun",
        "ground_truth",
        "dbim",
        "dbim_high_order",
    ], "only these sampler is supported currently"

    if sampler == "heun":
        ts = get_sigmas_karras(steps, diffusion.t_min, diffusion.t_max - 1e-4, rho, device=device)
    else:
        ts = get_sigmas_uniform(steps, diffusion.t_min, diffusion.t_max - 1e-3, device=device)

    sample_fn = {
        "heun": sample_heun,
        "ground_truth": sample_ground_truth,
        "dbim": sample_dbim,
        "dbim_high_order": sample_dbim_high_order,
    }[sampler]

    sampler_args = dict(churn_step_ratio=churn_step_ratio, mask=mask, eta=eta, x_0=x_0, order=order, seed=seed)

    _nfe_budget = steps + 1
    _nfe_counter = [0]
    def denoiser(x_t, sigma):
        if _nfe_counter[0] >= _nfe_budget:
            raise RuntimeError(
                f'NFE_BUDGET_EXCEEDED: used {_nfe_counter[0]+1} denoiser calls '
                f'but budget is {_nfe_budget}. Do not double-denoise / Heun-correct '
                f'beyond the allowed NFE.'
            )
        _nfe_counter[0] += 1
        _, denoised, _ = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    x_0, path, nfe, pred_x0, sigmas, noise = sample_fn(
        denoiser,
        diffusion,
        x_T,
        ts,
        **sampler_args,
    )
    if dist.get_rank() == 0:
        print("nfe:", nfe)

    return (
        x_0.clamp(-1, 1),
        [x.clamp(-1, 1) for x in path],
        nfe,
        [x.clamp(-1, 1) for x in pred_x0],
        sigmas,
        noise,
    )


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    return torch.linspace(t_max, t_min, n + 1).to(device)


@torch.no_grad()
def sample_dbim_high_order(
    denoiser,
    diffusion,
    x,
    ts,
    mask=None,
    order=2,
    lower_order_final=True,
    seed=None,
    **kwargs,
):
    if order not in [2, 3]:
        raise NotImplementedError("Not supported")
    x_T = x
    path = []
    pred_x0 = []

    ones = x.new_ones([x.shape[0]])
    indices = range(len(ts) - 1)
    indices = tqdm(indices, disable=(dist.get_rank() != 0))

    nfe = 0
    x0_hat = denoiser(x, diffusion.t_max * ones)
    generator = BatchedSeedGenerator(seed)
    noise = generator.randn_like(x0_hat)
    first_noise = noise
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)
    path.append(x.detach().cpu())
    pred_x0.append(x0_hat.detach().cpu())
    nfe += 1

    u = diffusion.t_max
    if u == 1.0:
        u -= 5e-5
    u = [u for _ in range(order - 1)]
    xu_hat = [x0_hat.detach().clone() for _ in range(order - 1)]

    for _, i in enumerate(indices):
        s = ts[i]
        t = ts[i + 1]

        # First Order Update, t < s
        if (lower_order_final and i + 1 == len(ts) - 1) or (i == 0):
            if dist.get_rank() == 0:
                print("Step order 1")
            a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]

            tmp_var = c_t / c_s
            coeff_xs = tmp_var
            coeff_x0_hat = b_t - tmp_var * b_s
            coeff_xT = a_t - tmp_var * a_s

            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            nfe += 1
            x_old = x
            x = coeff_xs * x_old + coeff_x0_hat * x0_hat + coeff_xT * x_T

        # Second Order Update, t < s < u
        elif order == 2 or i == 1:
            if dist.get_rank() == 0:
                print("Step order 2")
            a_u, b_u, c_u = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(u[-1] * ones)]
            a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u, lambda_s, lambda_t = (
                torch.log(b_u / c_u),
                torch.log(b_s / c_s),
                torch.log(b_t / c_t),
            )

            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            nfe += 1
            h = lambda_t - lambda_s
            h2 = lambda_s - lambda_u
            integral = torch.exp(lambda_t) * (
                (1 - torch.exp(-h)) * x0_hat + (torch.exp(-h) + h - 1) * (x0_hat - xu_hat[-1]) / h2
            )
            x_old = x
            x = x_old * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        elif order == 3:
            if dist.get_rank() == 0:
                print("Step order 3")
            a_u1, b_u1, c_u1 = [
                append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(u[-1] * ones)
            ]
            a_u2, b_u2, c_u2 = [
                append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(u[-2] * ones)
            ]
            a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u2, lambda_u1, lambda_s, lambda_t = (
                torch.log(b_u2 / c_u2),
                torch.log(b_u1 / c_u1),
                torch.log(b_s / c_s),
                torch.log(b_t / c_t),
            )
            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            nfe += 1

            h = lambda_t - lambda_s
            h1 = lambda_s - lambda_u1
            h2 = lambda_u1 - lambda_u2
            dx0_hat = ((x0_hat - xu_hat[-1]) * (2 * h1 + h2) / h1 - (xu_hat[-1] - xu_hat[-2]) * h1 / h2) / (h1 + h2)
            d2x0_hat = 2 * ((x0_hat - xu_hat[-1]) / h1 - (xu_hat[-1] - xu_hat[-2]) / h2) / (h1 + h2)
            integral = torch.exp(lambda_t) * (
                (1 - torch.exp(-h)) * x0_hat
                + (torch.exp(-h) + h - 1) * dx0_hat
                + (h**2 / 2 - h + 1 - torch.exp(-h)) * d2x0_hat
            )
            x_old = x
            x = x_old * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        u.append(s)
        u.pop(0)
@torch.no_grad()
def sample_dbim(
    denoiser,
    diffusion,
    x,
    ts,
    eta=1.0,
    mask=None,
    seed=None,
    **kwargs,
):
    # =================================================================================
    # 🚨 CRITICAL CONSTRAINTS - DO NOT IGNORE! 🚨
    # 1. Function Signature: You must NOT modify the function name, arguments, or return structure.
    # 2. NFE Match (FATAL I/O ERROR): The framework uses the final returned `nfe` to locate
    #    generated files (e.g., expecting `samples_..._nfe5.npz`). You MUST return
    #    `nfe = len(ts) - 1` regardless of the internal call count.
    # =================================================================================
    
    # TODO: Implement your novel sampling kernel here.
    # Ensure the return structure is: return x, path, nfe, pred_x0, ts, first_noise
    
    raise NotImplementedError("Custom sampler not implemented yet.")


@torch.no_grad()
def sample_ground_truth(
    denoiser,
    diffusion,
    x,
    ts,
    x0=None,
    **kwargs,
):
    assert x0 is not None
    x_T = x
    path = []
    pred_x0 = []

    ones = x.new_ones([x.shape[0]])
    indices = range(len(ts) - 1)
    indices = tqdm(indices, disable=(dist.get_rank() != 0))

    nfe = 0
    x0_hat = denoiser(x, diffusion.t_max * ones)
    noise = torch.randn_like(x0)
    first_noise = noise
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)
    path.append(x.detach().cpu())
    pred_x0.append(x0_hat.detach().cpu())
    nfe += 1

    for _, i in enumerate(indices):
        s = ts[i]
        t = ts[i + 1]

        x0_hat = denoiser(x, s * ones)
        noise = torch.randn_like(x0)
        x = diffusion.bridge_sample(x0, x_T, t * ones, noise)

        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())
        nfe += 1

    return x, path, nfe, pred_x0, ts, first_noise


def get_d(denoiser, noise_schedule, x, x_T, t, stochastic):
    ones = x.new_ones([x.shape[0]])
    f_t, g2_t = [append_dims(item, x.ndim) for item in noise_schedule.get_f_g2(t * ones)]
    alpha_t, alpha_bar_t, _, rho_bar_t = [append_dims(item, x.ndim) for item in noise_schedule.get_alpha_rho(t * ones)]
    a_t, b_t, c_t = [append_dims(item, x.ndim) for item in noise_schedule.get_abc(t * ones)]
    denoised = denoiser(x, t * ones)
    grad_logq = -(x - (a_t * x_T + b_t * denoised)) / c_t**2
    grad_logpxTlxt = -(x - alpha_bar_t * x_T) / (alpha_t**2 * rho_bar_t**2)
    d = f_t * x - g2_t * ((0.5 if not stochastic else 1) * grad_logq - grad_logpxTlxt)
    return d, g2_t, denoised


def ddbm_simulate(denoiser, noise_schedule, x, x_T, t_cur, t_next, stochastic, second_order=False):
    dt = t_next - t_cur
    if isinstance(noise_schedule, I2SBNoiseSchedule):
        dt = dt * (noise_schedule.n_timestep - 1)
    d, g2_t, pred_x0 = get_d(denoiser, noise_schedule, x, x_T, t_cur, stochastic)
    x_new = x + d * dt + (0 if not stochastic else 1) * torch.randn_like(x) * ((dt).abs() ** 0.5) * g2_t.sqrt()
    if second_order:
        d_2, _, pred_x0 = get_d(denoiser, noise_schedule, x_new, x_T, t_next, stochastic)
        d_prime = (d + d_2) / 2
        x_new = (
            x + d_prime * dt + (0 if not stochastic else 1) * torch.randn_like(x) * ((dt).abs() ** 0.5) * g2_t.sqrt()
        )
    return x_new, pred_x0


@torch.no_grad()
def sample_heun(
    denoiser,
    diffusion,
    x,
    ts,
    churn_step_ratio=0.0,
    **kwargs,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    x_T = x
    path = []
    pred_x0 = []

    indices = range(len(ts) - 1)

    indices = tqdm(indices, disable=(dist.get_rank() != 0))

    nfe = 0
    assert churn_step_ratio < 1

    for _, i in enumerate(indices):

        if churn_step_ratio > 0:
            # 1 step euler
            t_hat = (ts[i + 1] - ts[i]) * churn_step_ratio + ts[i]
            x, _pred_x0 = ddbm_simulate(
                denoiser,
                diffusion.noise_schedule,
                x,
                x_T,
                ts[i],
                t_hat,
                stochastic=True,
            )
            nfe += 1
            path.append(x.detach().cpu())
            pred_x0.append(_pred_x0.detach().cpu())
        else:
            t_hat = ts[i]

        # heun step
        if ts[i + 1] == 0:
            x, _pred_x0 = ddbm_simulate(
                denoiser,
                diffusion.noise_schedule,
                x,
                x_T,
                t_hat,
                ts[i + 1],
                stochastic=False,
            )
            nfe += 1
        else:
            # Heun's method
            x, _pred_x0 = ddbm_simulate(
                denoiser,
                diffusion.noise_schedule,
                x,
                x_T,
                t_hat,
                ts[i + 1],
                stochastic=False,
                second_order=True,
            )
            nfe += 2

        path.append(x.detach().cpu())
        pred_x0.append(_pred_x0.detach().cpu())

    return x, path, nfe, pred_x0, ts, None
