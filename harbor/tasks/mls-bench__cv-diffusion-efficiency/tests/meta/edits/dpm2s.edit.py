"""DPM++ 2S baseline edit — Second-order singlestep sampler.

Replaces the custom template with DPM++ 2S implementation for both
SD v1.5 (latent_diffusion.py) and SDXL (latent_sdxl.py).
DPM++ 2S uses Heun's method. With NFE=20, timesteps are halved to 10,
with 2 model evaluations per step.
"""

_SD_FILE = "CFGpp-main/latent_diffusion.py"
_SDXL_FILE = "CFGpp-main/latent_sdxl.py"

_DPM2S_SD = """\
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    \"\"\"
    DPM++ 2S sampler with CFG++.
    Second-order singlestep (Heun's method) - higher quality per step.
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

        # Halve timesteps: 2 model evals per step to stay within NFE budget
        timesteps = self.scheduler.timesteps[::2]
        double_skip = 2 * self.skip

        # Sampling
        pbar = tqdm(timesteps, desc="DPM++2S")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - double_skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # Tweedie: estimate clean image
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # First prediction (Euler step to next timestep)
            zt_euler = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc

            # DPM++ 2S: Heun's method for second-order accuracy
            if step < len(timesteps) - 1:
                # Evaluate at the predicted point (endpoint of Euler step)
                with torch.no_grad():
                    noise_uc_2, noise_c_2 = self.predict_noise(zt_euler, t - double_skip, uc, c)
                    noise_pred_2 = noise_uc_2 + cfg_guidance * (noise_c_2 - noise_uc_2)

                z0t_2 = (zt_euler - (1-at_prev).sqrt() * noise_pred_2) / at_prev.sqrt()

                # Average the two estimates (Heun's method)
                z0t_avg = 0.5 * (z0t + z0t_2)
                zt = at_prev.sqrt() * z0t_avg + (1-at_prev).sqrt() * noise_uc
            else:
                # Last step: just use first-order
                zt = zt_euler

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

_DPM2S_SDXL = """\
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

        timesteps = self.scheduler.timesteps.int()[::2]
        double_skip = 2 * self.skip

        pbar = tqdm(timesteps, desc='SDXL-DPM++2S')
        for step, t in enumerate(pbar):
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[t - double_skip]

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
            zt_euler = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_uc

            if step < len(timesteps) - 1:
                with torch.no_grad():
                    noise_uc_2, noise_c_2 = self.predict_noise(zt_euler, t - double_skip, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
                    noise_pred_2 = noise_uc_2 + cfg_guidance * (noise_c_2 - noise_uc_2)

                z0t_2 = (zt_euler - (1-at_next).sqrt() * noise_pred_2) / at_next.sqrt()
                z0t_avg = 0.5 * (z0t + z0t_2)
                zt = at_next.sqrt() * z0t_avg + (1-at_next).sqrt() * noise_uc
            else:
                zt = zt_euler

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
