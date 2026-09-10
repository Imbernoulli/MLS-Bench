"""DPM++ 2M baseline edit — Second-order multistep sampler.

Replaces the custom template with DPM++ 2M implementation for both
SD v1.5 (latent_diffusion.py) and SDXL (latent_sdxl.py).
"""

_SD_FILE = "CFGpp-main/latent_diffusion.py"
_SDXL_FILE = "CFGpp-main/latent_sdxl.py"

_DPM2M_SD = """\
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    \"\"\"
    DPM++ 2M sampler with CFG++.
    Second-order multistep method - more efficient than first-order.
    \"\"\"
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

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize zT
        zt = self.initialize_latent()
        zt = zt.requires_grad_()

        old_z0t = None  # Buffer for second-order correction

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="DPM++2M")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # Tweedie: estimate clean image
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # DPM++ 2M update: second-order multistep
            if old_z0t is None or step == len(self.scheduler.timesteps) - 1:
                # First step or last step: use first-order
                zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc
            else:
                # Second-order correction using previous estimate
                z0t_corrected = z0t + 0.5 * (z0t - old_z0t)
                zt = at_prev.sqrt() * z0t_corrected + (1-at_prev).sqrt() * noise_uc

            old_z0t = z0t.clone()

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # Decode final latent
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
"""

_DPM2M_SDXL = """\
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(SDXL):
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        **kwargs):
        zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))

        old_z0t = None

        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
        for step, t in enumerate(pbar):
            next_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[next_t]

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            if old_z0t is None or step == len(self.scheduler.timesteps) - 1:
                zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_uc
            else:
                z0t_corrected = z0t + 0.5 * (z0t - old_z0t)
                zt = at_next.sqrt() * z0t_corrected + (1-at_next).sqrt() * noise_uc

            old_z0t = z0t.clone()

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        return z0t
"""

OPS = [
    {
        "op": "replace",
        "file": _SD_FILE,
        "start_line": 621,
        "end_line": 679,
        "content": _DPM2M_SD,
    },
    {
        "op": "replace",
        "file": _SDXL_FILE,
        "start_line": 713,
        "end_line": 755,
        "content": _DPM2M_SDXL,
    },
]
