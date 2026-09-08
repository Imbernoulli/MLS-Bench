@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    # TODO: Implement your improved method here.
    #
    # Your goal is to improve the CFG mechanism to achieve better text-image
    # alignment (measured by CLIP score) while maintaining or improving sample quality.
    #
    # Key methods you need to implement:
    # - __init__: Initialize the solver
    # - sample: Main sampling function that generates images
    #
    # Available helper methods from parent class:
    # - self.get_text_embed(null_prompt, prompt): Get text embeddings
    # - self.initialize_latent(): Initialize latent variable zT
    # - self.predict_noise(zt, t, uc, c): Predict noise at timestep t
    # - self.alpha(t): Get alpha_t value
    # - self.decode(z): Decode latent to image
    # - self.scheduler.timesteps: List of timesteps to iterate over
    #
    # The baseline CFG++ uses unconditional noise (noise_uc) for renoising to keep
    # the trajectory on the data manifold. You should modify the sampling logic to
    # improve upon this approach.

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
        # TODO: Implement your improved sampling method here.
        #
        #
        #
        #
        # Consider modifications to the CFG formula, adaptive guidance scales,
        # or alternative renoising strategies to improve generation quality.
        raise NotImplementedError("You need to implement the sample method")















