"""DPM++ 2S Ancestral baseline — from CFGpp-main/latent_diffusion.py dpm++_2s_a.

Standard CFG version with karras sigma schedule and ancestral sampling.
Reference: vendor/external_packages/CFGpp-main/latent_diffusion.py line 393.
"""

_SD_FILE = "CFGpp-main/latent_diffusion.py"
_SDXL_FILE = "CFGpp-main/latent_sdxl.py"

_DPM2S_SD = """\
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    \"\"\"DPM++ 2S Ancestral sampler with standard CFG.\"\"\"
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
        sigma_fn = lambda t: t.neg().exp()

        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
        sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)

        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=sigmas).to(torch.float16)

        pbar = tqdm(self.scheduler.timesteps, desc="DPM++2S")
        for i, _ in enumerate(pbar):
            sigma = sigmas[i]
            new_t = self.timestep(sigma).to(self.device)

            with torch.no_grad():
                denoised, _ = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)

            sigma_down, sigma_up = self.get_ancestral_step(sigmas[i], sigmas[i + 1])
            if sigma_down == 0:
                d = self.to_d(x, sigmas[i], denoised)
                x = denoised + d * sigma_down
            else:
                t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
                r = 1 / 2
                h = t_next - t
                s = t + r * h
                x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised

                with torch.no_grad():
                    sigma_s = sigma_fn(s)
                    t_2 = self.timestep(sigma_s).to(self.device)
                    denoised_2, _ = self.kdiffusion_x_to_denoised(x_2, sigma_s, uc, c, cfg_guidance, t_2)

                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2

            if sigmas[i + 1] > 0:
                x = x + torch.randn_like(x) * sigma_up

            if callback_fn is not None:
                callback_kwargs = {'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, new_t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]

        img = self.decode(x)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
"""

_DPM2S_SDXL = """\
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(SDXL):
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
        sigma_fn = lambda t: t.neg().exp()

        alphas = self.scheduler.alphas_cumprod[self.scheduler.timesteps.int().cpu()].cpu()
        sigmas = (1-alphas).sqrt() / alphas.sqrt()

        zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        x = zt * sigmas[0]

        pbar = tqdm(self.scheduler.timesteps[:-1].int(), desc='SDXL-DPM++2S')
        for i, _ in enumerate(pbar):
            at = alphas[i]
            sigma = sigmas[i]
            c_in = at.sqrt()
            c_out = -sigma

            new_t = self.sigma_to_t(sigma).to(self.device)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(x * c_in, new_t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            denoised = x + c_out * noise_pred

            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
            if sigma_down == 0:
                d = (x - denoised) / sigma
                x = denoised + d * sigma_down
            else:
                t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
                r = 1 / 2
                h = t_next - t
                s = t + r * h
                x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised

                sigma_s = sigma_fn(s)
                at_s_idx = min(int((sigma_s**2 / (1 + sigma_s**2)) * len(alphas)), len(alphas)-1)
                c_in_2 = (1 / (1 + sigma_s**2)).sqrt()
                c_out_2 = -sigma_s
                t_2 = self.sigma_to_t(sigma_s).to(self.device)

                with torch.no_grad():
                    noise_uc_2, noise_c_2 = self.predict_noise(x_2 * c_in_2, t_2, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
                    noise_pred_2 = noise_uc_2 + cfg_guidance * (noise_c_2 - noise_uc_2)

                denoised_2 = x_2 + c_out_2 * noise_pred_2
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2

            if sigmas[i + 1] > 0:
                x = x + torch.randn_like(x) * sigma_up

            if callback_fn is not None:
                callback_kwargs = {'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, new_t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]

        return x
"""

OPS = [
    {
        "op": "replace",
        "file": _SD_FILE,
        "start_line": 621,
        "end_line": 679,
        "content": _DPM2S_SD,
    },
    {
        "op": "replace",
        "file": _SDXL_FILE,
        "start_line": 713,
        "end_line": 755,
        "content": _DPM2S_SDXL,
    },
]
