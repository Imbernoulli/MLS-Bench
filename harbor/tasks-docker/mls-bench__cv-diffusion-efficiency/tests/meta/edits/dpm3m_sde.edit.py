"""DPM++ 3M SDE baseline.

Third-order multistep exponential integrator with Karras sigmas and Langevin
noise. This restores the historical baseline edit used for the recorded
dpm3m_sde anchor.
"""

_SD_FILE = "CFGpp-main/latent_diffusion.py"
_SDXL_FILE = "CFGpp-main/latent_sdxl.py"

_DPM3M_SDE_SD = '''@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    """DPM-Solver++(3M) SDE with Karras schedule."""

    def __init__(self,
                 solver_config: Dict,
                 model_key:str="runwayml/stable-diffusion-v1-5",
                 device: Optional[torch.device]=None,
                 **kwargs):
        super().__init__(solver_config, model_key, device, **kwargs)

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               cfg_guidance=7.5,
               prompt=["",""],
               callback_fn=None,
               **kwargs):
        t_fn = lambda sigma: sigma.log().neg()

        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
        sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)

        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=sigmas).to(torch.float16)

        eta = 1.2
        denoised_1, denoised_2 = None, None
        h_1, h_2 = None, None

        pbar = tqdm(self.scheduler.timesteps, desc="DPM++3M-SDE")
        for i, _ in enumerate(pbar):
            sigma = sigmas[i]
            new_t = self.timestep(sigma).to(self.device)

            with torch.no_grad():
                denoised, _ = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)

            if sigmas[i + 1] == 0:
                x = denoised
            else:
                t, s = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
                h = s - t
                h_eta = h * (eta + 1)

                x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

                if denoised_1 is not None:
                    phi_2 = h_eta.neg().expm1() / h_eta + 1

                    if denoised_2 is None:
                        r = h_1 / h
                        d = (denoised - denoised_1) / r
                        x = x + phi_2 * d
                    else:
                        r0 = h_1 / h
                        r1 = h_2 / h_1
                        d1_0 = (denoised - denoised_1) / r0
                        d1_1 = (denoised_1 - denoised_2) / r1
                        d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                        d2 = (d1_0 - d1_1) / (r0 + r1)
                        phi_3 = phi_2 / h_eta - 0.5
                        x = x + phi_2 * d1 + phi_3 * d2

                if eta > 0:
                    noise = torch.randn_like(x)
                    x = x + noise * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt()

                denoised_2 = denoised_1
                denoised_1 = denoised
                h_2 = h_1
                h_1 = h

            if callback_fn is not None:
                callback_kwargs = {'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, new_t, callback_kwargs)
                x = callback_kwargs["zt"]

        z0t = x
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
'''

_DPM3M_SDE_SDXL = '''@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(SDXL):
    """DPM-Solver++(3M) SDE with Karras schedule for SDXL."""
    quantize = True

    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        **kwargs):
        t_fn = lambda sigma: sigma.log().neg()

        total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
        sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)

        latent_dim = (1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor)
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=latent_dim,
                                   sigmas=sigmas).to(torch.float16)

        eta = 1.2
        denoised_1, denoised_2 = None, None
        h_1, h_2 = None, None

        pbar = tqdm(self.scheduler.timesteps, desc="SDXL-DPM++3M-SDE")
        for i, _ in enumerate(pbar):
            sigma = sigmas[i]
            new_t = self.sigma_to_t(sigma).to(self.device)
            c_in = (1 / (sigma ** 2 + 1)).sqrt()
            c_out = -sigma

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(
                    x * c_in, new_t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            denoised = x + c_out * noise_pred

            if sigmas[i + 1] == 0:
                x = denoised
            else:
                t, s = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
                h = s - t
                h_eta = h * (eta + 1)

                x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

                if denoised_1 is not None:
                    phi_2 = h_eta.neg().expm1() / h_eta + 1

                    if denoised_2 is None:
                        r = h_1 / h
                        d = (denoised - denoised_1) / r
                        x = x + phi_2 * d
                    else:
                        r0 = h_1 / h
                        r1 = h_2 / h_1
                        d1_0 = (denoised - denoised_1) / r0
                        d1_1 = (denoised_1 - denoised_2) / r1
                        d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                        d2 = (d1_0 - d1_1) / (r0 + r1)
                        phi_3 = phi_2 / h_eta - 0.5
                        x = x + phi_2 * d1 + phi_3 * d2

                if eta > 0:
                    noise = torch.randn_like(x)
                    x = x + noise * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt()

                denoised_2 = denoised_1
                denoised_1 = denoised
                h_2 = h_1
                h_1 = h

            if callback_fn is not None:
                callback_kwargs = {'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, new_t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]

        return x
'''

OPS = [
    {
        "op": "replace",
        "file": _SD_FILE,
        "start_line": 621,
        "end_line": 679,
        "content": _DPM3M_SDE_SD,
    },
    {
        "op": "replace",
        "file": _SDXL_FILE,
        "start_line": 713,
        "end_line": 755,
        "content": _DPM3M_SDE_SDXL,
    },
]
