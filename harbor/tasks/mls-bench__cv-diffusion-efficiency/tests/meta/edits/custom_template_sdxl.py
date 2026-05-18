@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(SDXL):
    # TODO: Implement your improved method here.
    #
    # Your goal is to improve the CFG mechanism for SDXL to achieve better
    # text-image alignment (measured by CLIP score).
    #
    # Key methods you need to implement:
    # - reverse_process: Main sampling function that generates latents
    #
    # Available helper methods from parent class:
    # - self.initialize_latent(size=(1, 4, H//vae_scale, W//vae_scale))
    # - self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
    # - self.scheduler.alphas_cumprod[t]: Get alpha_t value
    # - self.scheduler.timesteps: List of timesteps
    # - self.skip: Timestep skip value
    # - self.vae_scale_factor: VAE downscaling factor
    #
    # The baseline CFG++ uses unconditional noise (noise_uc) for renoising.
    # You should modify the sampling logic to improve upon this approach.

    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        **kwargs):
        # TODO: Implement your improved reverse process here.
        #
        #
        #
        # Consider modifications to the CFG formula, adaptive guidance scales,
        # or alternative renoising strategies to improve generation quality.
        raise NotImplementedError("You need to implement reverse_process")







