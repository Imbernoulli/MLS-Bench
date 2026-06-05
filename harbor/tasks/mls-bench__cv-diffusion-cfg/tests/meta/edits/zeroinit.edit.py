"""Zero-init + Rescaled CFG baseline — zero-init K=2 + Imagen std-rescale.

Combines two known techniques:
1. Zero-init: skip guidance (w=0) for the first K=2 timesteps, then
   apply cfg_guidance=7.5 for the rest. Avoids over-correction in the
   high-noise regime where (noise_c - noise_uc) is unstable.
2. Imagen Rescaled CFG (Lin et al 2024): normalize noise_pred std to
   match noise_c std, then mix with phi=0.7 against the un-rescaled
   noise_pred. Fixes the over-saturation problem of vanilla CFG at
   high guidance.

Renoises with `noise_pred` (standard CFG flavor).
"""

_SD_FILE = "CFGpp-main/latent_diffusion.py"
_SDXL_FILE = "CFGpp-main/latent_sdxl.py"

_ZEROINIT_SD = """\
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    \"\"\"
    DDIM solver for SD with CFG and Zero-init for first K steps.
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
        # Zero-init natural scale: standard CFG strength (7.5). The zero-init
        # trick only delays the first K steps; the rest runs at normal scale.
        # Hardcoded as method design.
        cfg_guidance = 7.5

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize zT
        zt = self.initialize_latent()
        zt = zt.requires_grad_()

        # Zero-init parameter
        K = 2  # First K steps use guidance=0

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)

                # Zero-init: w=0 for first K steps, then normal CFG
                w = 0.0 if step < K else cfg_guidance
                noise_pred = noise_uc + w * (noise_c - noise_uc)

                # Imagen Rescaled CFG (Lin et al 2024) — only when guidance is active
                if w > 0:
                    rescale_phi = 0.7
                    std_c = noise_c.std(dim=list(range(1, noise_c.ndim)), keepdim=True)
                    std_pred = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)
                    noise_pred_rescaled = noise_pred * (std_c / std_pred)
                    noise_pred = rescale_phi * noise_pred_rescaled + (1 - rescale_phi) * noise_pred

            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise - standard CFG renoising
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
"""

_ZEROINIT_SDXL = """\
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
        # Zero-init natural scale — hardcoded as method design
        cfg_guidance = 7.5
        zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))

        K = 2  # First K steps use guidance=0

        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
        for step, t in enumerate(pbar):
            next_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[next_t]

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)

                # Zero-init: w=0 for first K steps, then normal CFG
                w = 0.0 if step < K else cfg_guidance
                noise_pred = noise_uc + w * (noise_c - noise_uc)

                # Imagen Rescaled CFG (Lin et al 2024)
                if w > 0:
                    rescale_phi = 0.7
                    std_c = noise_c.std(dim=list(range(1, noise_c.ndim)), keepdim=True)
                    std_pred = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)
                    noise_pred_rescaled = noise_pred * (std_c / std_pred)
                    noise_pred = rescale_phi * noise_pred_rescaled + (1 - rescale_phi) * noise_pred

            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # Standard CFG renoising
            zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_pred

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
        "content": _ZEROINIT_SD,
    },
    {
        "op": "replace",
        "file": _SDXL_FILE,
        "start_line": 713,
        "end_line": 755,
        "content": _ZEROINIT_SDXL,
    },
]
