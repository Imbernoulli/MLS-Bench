@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    # TODO: Implement your improved sampling method here.
    #
    # You should implement an improved sampling algorithm that achieves better
    # image quality (FID) with a fixed budget of NFE=50 steps.
    #
    # Key methods you need to implement:
    # - __init__: Initialize the solver
    # - sample: Main sampling function with your update rule
    #
    # Available helper methods from parent class:
    # - self.get_text_embed(null_prompt, prompt): Get text embeddings
    # - self.initialize_latent(): Initialize latent variable zT
    # - self.predict_noise(zt, t, uc, c): Predict noise at timestep t
    # - self.alpha(t): Get alpha_t value (sqrt of cumulative product of alphas)
    # - self.decode(z): Decode latent to image
    # - self.scheduler.timesteps: List of timesteps to iterate over
    #
    # Focus on optimizing the update rule in the sampling loop to achieve better
    # quality with the fixed NFE budget.

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
        # TODO: Implement your efficient sampling method here.
        #
        #
        #
        #
        # This method should generate high-quality images with minimal sampling steps.
        # Consider different update rules, adaptive step sizes, or combining multiple
        # methods to achieve better performance.
        raise NotImplementedError("You need to implement the sample method")













