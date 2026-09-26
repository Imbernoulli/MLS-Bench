# MLS-Bench: cv-diffusion-efficiency

# Diffusion Model: Sampler Efficiency Optimization

## Objective

Design a sampling algorithm for text-to-image diffusion models that achieves
high generation quality with a fixed budget of NFE = 50 denoiser evaluations.

## Background

Diffusion models generate images by iteratively denoising from random noise.
Different samplers differ in how they update the latent after each model
prediction. The general structure of one step is:

```python
for step, t in enumerate(timesteps):
    # 1. Predict noise.
    noise_pred = model(zt, t, text_embedding)
    # 2. Estimate clean image (Tweedie's formula).
    z0t = (zt - sigma_t * noise_pred) / alpha_t
    # 3. Update to next step (this differs across samplers).
    zt_next = update_rule(zt, z0t, noise_pred, t, t_next)
```

Reference families:

- **DDIM** (Song et al., ICLR 2021, arXiv:2010.02502) — first-order ODE
  solver, deterministic, simple update rule.
- **DPM-Solver++** (Lu et al., 2022, arXiv:2211.01095) — high-order solvers
  for the diffusion ODE in data-prediction form.
  - **DPM-Solver++(2M)** — second-order multistep variant, reuses the
    previous denoiser output.
  - **DPM-Solver++(2S)** — second-order singlestep variant, smaller
    high-order error constant.
  - **DPM-Solver++(3M) SDE** — third-order multistep stochastic variant for
    guided sampling.

A useful method may use time-dependent coefficients, history (multistep),
predictor-corrector structure, or guidance-aware renoising — but it must
respect the fixed function-evaluation budget.

## Implementation Contract

Implement the update rule for both Stable Diffusion v1.5 and SDXL by editing
the marked editable regions of two files:

1. **`latent_diffusion.py`** — `BaseDDIMCFGpp` class for SD v1.5
   (`sample()` method). Available helpers:
   `self.get_text_embed()`, `self.initialize_latent()`,
   `self.predict_noise()`, `self.alpha(t)`.
2. **`latent_sdxl.py`** — `BaseDDIMCFGpp` class for SDXL
   (`reverse_process()` method). Available helpers:
   `self.initialize_latent(size=...)`, `self.predict_noise()`,
   `self.scheduler.alphas_cumprod[t]`.

The contribution must respect a fixed budget of **NFE = 50** denoiser calls
per sample.

The budget is measured, not trusted. The `self.unet` that the fixed base-class
`__init__` gives the solver is a counting wrapper (callable like the UNet, with
its `config`, `dtype` and `device`); the raw network is never handed to the
solver, so every UNet forward the sampler makes, through `self.predict_noise()`
or `self.unet(...)` directly, is counted for each image. One NFE is one UNet
forward on the image's latent: the batched unconditional + conditional pair of
classifier-free guidance counts as 1, and so does a single-branch call. A
forward over more than two latent rows counts ceil(rows / 2). A run that spends
more than 50 NFE on any image is rejected and records no FID; spending fewer is
allowed. All denoiser evaluations must go through `self.unet` /
`self.predict_noise()`, and a solver whose `self.unet` is not the one its
base-class `__init__` set up is rejected.

## Baselines

| Baseline    | Description |
|-------------|-------------|
| `ddim`      | DDIM (Song et al., ICLR 2021, arXiv:2010.02502). First-order deterministic. |
| `dpm3m_sde` | DPM-Solver++(3M) SDE multistep variant (Lu et al., 2022, arXiv:2211.01095). |
| `dpm2s`     | DPM-Solver++(2S) second-order singlestep variant (same paper). Two evaluations per step, so 25 steps. |

## Fixed Pipeline

The training and evaluation pipeline (models, weights, prompt set, and metric computation) is fixed by the harness and not editable. Only the marked editable regions of the two solver files may be changed, and the fixed function-evaluation budget stated in the Implementation Contract must be respected.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/CFGpp-main/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `CFGpp-main/latent_diffusion.py`
- editable lines **621–677**
- `CFGpp-main/latent_sdxl.py`
- editable lines **722–764**




## Readable Context


### `CFGpp-main/latent_diffusion.py`  [EDITABLE — lines 621–677 only]

```python
     1: """
     2: This module includes LDM-based inverse problem solvers.
     3: Forward operators follow DPS and DDRM/DDNM.
     4: """
     5: 
     6: from typing import Any, Callable, Dict, Optional
     7: 
     8: import torch
     9: from diffusers import DDIMScheduler, StableDiffusionPipeline
    10: from tqdm import tqdm
    11: 
    12: ####### Factory #######
    13: __SOLVER__ = {}
    14: 
    15: def register_solver(name: str):
    16:     def wrapper(cls):
    17:         if __SOLVER__.get(name, None) is not None:
    18:             raise ValueError(f"Solver {name} already registered.")
    19:         __SOLVER__[name] = cls
    20:         return cls
    21:     return wrapper
    22: 
    23: def get_solver(name: str, **kwargs):
    24:     if name not in __SOLVER__:
    25:         raise ValueError(f"Solver {name} does not exist.")
    26:     return __SOLVER__[name](**kwargs)
    27: 
    28: ########################
    29: 
    30: def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    31:     """Calculates the noise level (sigma_down) to step down to and the amount
    32:     of noise to add (sigma_up) when doing an ancestral sampling step."""
    33:     if not eta:
    34:         return sigma_to, 0.
    35:     sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    36:     sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    37:     return sigma_down, sigma_up
    38: 
    39: 
    40: def append_zero(x):
    41:     return torch.cat([x, x.new_zeros([1])])
    42: 
    43: 
    44: def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    45:     """Constructs the noise schedule of Karras et al. (2022)."""
    46:     ramp = torch.linspace(0, 1, n+1, device=device)[:-1]
    47:     min_inv_rho = sigma_min ** (1 / rho)
    48:     max_inv_rho = sigma_max ** (1 / rho)
    49:     sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    50:     return append_zero(sigmas).to(device)
    51: 
    52: ########################
    53: 
    54: class StableDiffusion():
    55:     def __init__(self,
    56:                  solver_config: Dict,
    57:                  model_key:str="runwayml/stable-diffusion-v1-5",
    58:                  device: Optional[torch.device]=None,
    59:                  **kwargs):
    60:         self.device = device
    61: 
    62:         self.dtype = kwargs.get("pipe_dtype", torch.float16)
    63:         pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype).to(device)
    64:         self.vae = pipe.vae
    65:         self.tokenizer = pipe.tokenizer
    66:         self.text_encoder = pipe.text_encoder
    67:         self.unet = pipe.unet
    68: 
    69:         self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
    70:         self.total_alphas = self.scheduler.alphas_cumprod.clone()
    71:         
    72:         self.sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
    73:         self.log_sigmas = self.sigmas.log()
    74:         
    75:         total_timesteps = len(self.scheduler.timesteps)
    76:         self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
    77:         self.skip = total_timesteps // solver_config.num_sampling
    78: 
    79:         self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
    80:         self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])
    81: 
    82:     def __call__(self, *args: Any, **kwargs: Any) -> Any:
    83:         self.sample(*args, **kwargs)
    84: 
    85:     def sample(self, *args: Any, **kwargs: Any) -> Any:
    86:         raise NotImplementedError("Solver must implement sample() method.")
    87: 
    88:     def alpha(self, t):
    89:         at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
    90:         return at
    91: 
    92:     @torch.no_grad()
    93:     def get_text_embed(self, null_prompt, prompt):
    94:         """
    95:         Get text embedding.
    96:         args:
    97:             null_prompt (str): null text
    98:             prompt (str): guidance text
    99:         """
   100:         # null text embedding (negation)
   101:         null_text_input = self.tokenizer(null_prompt,
   102:                                          padding='max_length',
   103:                                          max_length=self.tokenizer.model_max_length,
   104:                                          return_tensors="pt",)
   105:         null_text_embed = self.text_encoder(null_text_input.input_ids.to(self.device))[0]
   106: 
   107:         # text embedding (guidance)
   108:         text_input = self.tokenizer(prompt,
   109:                                     padding='max_length',
   110:                                     max_length=self.tokenizer.model_max_length,
   111:                                     return_tensors="pt",
   112:                                     truncation=True)
   113:         text_embed = self.text_encoder(text_input.input_ids.to(self.device))[0]
   114: 
   115:         return null_text_embed, text_embed
   116: 
   117:     def encode(self, x):
   118:         """
   119:         xt -> zt
   120:         """
   121:         return self.vae.encode(x).latent_dist.sample() * 0.18215
   122: 
   123:     def decode(self, zt):
   124:         """
   125:         zt -> xt
   126:         """
   127:         zt = 1/0.18215 * zt
   128:         img = self.vae.decode(zt).sample.float()
   129:         return img
   130: 
   131:     def predict_noise(self,
   132:                       zt: torch.Tensor,
   133:                       t: torch.Tensor,
   134:                       uc: torch.Tensor,
   135:                       c: torch.Tensor):
   136:         """
   137:         compuate epsilon_theta for null and condition
   138:         args:
   139:             zt (torch.Tensor): latent features
   140:             t (torch.Tensor): timestep
   141:             uc (torch.Tensor): null-text embedding
   142:             c (torch.Tensor): text embedding
   143:         """
   144:         t_in = t.unsqueeze(0)
   145:         if uc is None:
   146:             noise_c = self.unet(zt, t_in, encoder_hidden_states=c)['sample']
   147:             noise_uc = noise_c
   148:         elif c is None:
   149:             noise_uc = self.unet(zt, t_in, encoder_hidden_states=uc)['sample']
   150:             noise_c = noise_uc
   151:         else:
   152:             c_embed = torch.cat([uc, c], dim=0)
   153:             z_in = torch.cat([zt] * 2)
   154:             t_in = torch.cat([t_in] * 2)
   155:             noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed)['sample']
   156:             noise_uc, noise_c = noise_pred.chunk(2)
   157: 
   158:         return noise_uc, noise_c
   159: 
   160:     @torch.no_grad()
   161:     def inversion(self,
   162:                   z0: torch.Tensor,
   163:                   uc: torch.Tensor,
   164:                   c: torch.Tensor,
   165:                   cfg_guidance: float=1.0):
   166: 
   167:         # initialize z_0
   168:         zt = z0.clone().to(self.device)
   169: 
   170:         # loop
   171:         pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM Inversion')
   172:         for _, t in enumerate(pbar):
   173:             at = self.alpha(t)
   174:             at_prev = self.alpha(t - self.skip)
   175: 
   176:             noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
   177:             noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   178: 
   179:             z0t = (zt - (1-at_prev).sqrt() * noise_pred) / at_prev.sqrt()
   180:             zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred
   181: 
   182:         return zt
   183: 
   184:     def initialize_latent(self,
   185:                           method: str='random',
   186:                           src_img: Optional[torch.Tensor]=None,
   187:                           **kwargs):
   188:         if method == 'ddim':
   189:             z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
   190:                                kwargs.get('uc'),
   191:                                kwargs.get('c'),
   192:                                cfg_guidance=kwargs.get('cfg_guidance', 0.0))
   193:         elif method == 'npi':
   194:             z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
   195:                                kwargs.get('c'),
   196:                                kwargs.get('c'),
   197:                                cfg_guidance=1.0)
   198:         elif method == 'random':
   199:             size = kwargs.get('latent_dim', (1, 4, 64, 64))
   200:             z = torch.randn(size).to(self.device)
   201:         elif method == 'random_kdiffusion':
   202:             size = kwargs.get('latent_dim', (1, 4, 64, 64))
   203:             sigmas = kwargs.get('sigmas', [14.6146])
   204:             z = torch.randn(size).to(self.device)
   205:             z = z * (sigmas[0] ** 2 + 1) ** 0.5
   206:         else:
   207:             raise NotImplementedError
   208: 
   209:         return z.requires_grad_()
   210:     
   211:     def timestep(self, sigma):
   212:         log_sigma = sigma.log()
   213:         dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
   214:         return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)
   215: 
   216:     def to_d(self, x, sigma, denoised):
   217:         '''converts a denoiser output to a Karras ODE derivative'''
   218:         return (x - denoised) / sigma.item()
   219:     
   220:     def get_ancestral_step(self, sigma_from, sigma_to, eta=1.):
   221:         """Calculates the noise level (sigma_down) to step down to and the amount
   222:         of noise to add (sigma_up) when doing an ancestral sampling step."""
   223:         if not eta:
   224:             return sigma_to, 0.
   225:         sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
   226:         sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
   227:         return sigma_down, sigma_up
   228:     
   229:     def calculate_input(self, x, sigma):
   230:         return x / (sigma ** 2 + 1) ** 0.5
   231:     
   232:     def calculate_denoised(self, x, model_pred, sigma):
   233:         return x - model_pred * sigma
   234:     
   235:     def kdiffusion_x_to_denoised(self, x, sigma, uc, c, cfg_guidance, t):
   236:         xc = self.calculate_input(x, sigma)
   237:         noise_uc, noise_c = self.predict_noise(xc, t, uc, c)
   238:         noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   239:         denoised = self.calculate_denoised(x, noise_pred, sigma)
   240:         # Keep the tuple shape; hide the pure unconditional denoised prediction.
   241:         return denoised, denoised
   242: 
   243: ###########################################
   244: # Base version
   245: ###########################################
   246: 
   247: @register_solver("ddim")
   248: class BaseDDIM(StableDiffusion):
   249:     """
   250:     Basic DDIM solver for SD.
   251:     Useful for text-to-image generation
   252:     """
   253: 
   254:     @torch.autocast(device_type='cuda', dtype=torch.float16)
   255:     def sample(self,
   256:                cfg_guidance=7.5,
   257:                prompt=["",""],
   258:                callback_fn=None,
   259:                **kwargs):
   260:         """
   261:         Main function that defines each solver.
   262:         This will generate samples without considering measurements.
   263:         """
   264: 
   265:         # Text embedding
   266:         uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
   267: 
   268:         # Initialize zT
   269:         zt = self.initialize_latent()
   270:         zt = zt.requires_grad_()
   271: 
   272:         # Sampling
   273:         pbar = tqdm(self.scheduler.timesteps, desc="SD")
   274:         for step, t in enumerate(pbar):
   275:             at = self.alpha(t)
   276:             at_prev = self.alpha(t - self.skip)
   277: 
   278:             with torch.no_grad():
   279:                 noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
   280:                 noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   281: 
   282:             # tweedie
   283:             z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
   284: 
   285:             # add noise
   286:             zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred
   287: 
   288:             if callback_fn is not None:
   289:                 callback_kwargs = {'z0t': z0t.detach(),
   290:                                     'zt': zt.detach(),
   291:                                     'decode': self.decode}
   292:                 callback_kwargs = callback_fn(step, t, callback_kwargs)
   293:                 z0t = callback_kwargs["z0t"]
   294:                 zt = callback_kwargs["zt"]
   295: 
   296:         # for the last step, do not add noise
   297:         img = self.decode(z0t)
   298:         img = (img / 2 + 0.5).clamp(0, 1)
   299:         return img.detach().cpu()
   300:     
   301:     
   302: @register_solver("euler")
   303: class EulerCFGSolver(StableDiffusion):
   304:     """
   305:     Karras Euler (VE casted)
   306:     """
   307:     @torch.autocast(device_type='cuda', dtype=torch.float16)
   308:     def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
   309:         # Text embedding
   310:         uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
   311: 
   312:         # perpare alphas and sigmas
   313:         timesteps = reversed(torch.linspace(0, 1000, len(self.scheduler.timesteps)+1).long())
   314:         # convert to karras sigma scheduler
   315:         total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
   316:         sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
   317:         # initialize
   318:         x = self.initialize_latent(method="random_kdiffusion",
   319:                                    latent_dim=(1, 4, 64, 64),
   320:                                    sigmas=sigmas).to(torch.float16)
   321: 
   322:         # Sampling
   323:         pbar = tqdm(self.scheduler.timesteps, desc="SD")
   324:         for i, _ in enumerate(pbar):
   325:             sigma = sigmas[i]
   326:             t = self.timestep(sigma).to(self.device)
   327:             
   328:             with torch.no_grad():
   329:                 denoised, _ = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, t)
   330:             
   331:             d = self.to_d(x, sigma, denoised)
   332:             # Euler method
   333:             x = denoised + d * sigmas[i+1]
   334: 
   335:             if callback_fn is not None:
   336:                 callback_kwargs = {'z0t': denoised.detach(),
   337:                                     'zt': x.detach(),
   338:                                     'decode': self.decode}
   339:                 callback_kwargs = callback_fn(i, t, callback_kwargs)
   340:                 z0t = callback_kwargs["z0t"]
   341:                 zt = callback_kwargs["zt"]
   342: 
   343:         # for the last step, do not add noise
   344:         img = self.decode(denoised)
   345:         img = (img / 2 + 0.5).clamp(0, 1)
   346:         return img.detach().cpu()
   347:     
   348:     
   349: @register_solver("euler_a")
   350: class EulerAncestralCFGSolver(StableDiffusion):
   351:     """
   352:     Karras Euler (VE casted) + Ancestral sampling
   353:     """
   354:     @torch.autocast(device_type='cuda', dtype=torch.float16)
   355:     def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
   356:         # Text embedding
   357:         uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
   358:         # convert to karras sigma scheduler
   359:         total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
   360:         sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
   361:         # initialize
   362:         x = self.initialize_latent(method="random_kdiffusion",
   363:                                    latent_dim=(1, 4, 64, 64),
   364:                                    sigmas=sigmas).to(torch.float16)
   365:         # Sampling
   366:         pbar = tqdm(self.scheduler.timesteps, desc="SD")
   367:         for i, _ in enumerate(pbar):
   368:             sigma = sigmas[i]
   369:             t = self.timestep(sigma).to(self.device)
   370:             sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
   371:             with torch.no_grad():
   372:                 denoised, _ = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, t)
   373:             
   374:             # Euler method
   375:             d = self.to_d(x, sigma, denoised)
   376:             x = denoised + d * sigma_down
   377:             
   378:             if sigmas[i + 1] > 0:
   379:                 x = x + torch.randn_like(x) * sigma_up
   380: 
   381:             if callback_fn is not None:
   382:                 callback_kwargs = {'z0t': denoised.detach(),
   383:                                     'zt': x.detach(),
   384:                                     'decode': self.decode}
   385:                 callback_kwargs = callback_fn(i, t, callback_kwargs)
   386: 
   387:         # for the last step, do not add noise
   388:         img = self.decode(denoised)
   389:         img = (img / 2 + 0.5).clamp(0, 1)
   390:         return img.detach().cpu()
   391:     
   392:     
   393: @register_solver("dpm++_2s_a")
   394: class DPMpp2sAncestralCFGSolver(StableDiffusion):
   395:     @torch.autocast(device_type='cuda', dtype=torch.float16)
   396:     def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
   397:         t_fn = lambda sigma: sigma.log().neg()
   398:         sigma_fn = lambda t: t.neg().exp()
   399:         # Text embedding
   400:         uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
   401:         # convert to karras sigma scheduler
   402:         total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
   403:         sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
   404:         # initialize
   405:         x = self.initialize_latent(method="random_kdiffusion",
   406:                                    latent_dim=(1, 4, 64, 64),
   407:                                    sigmas=sigmas).to(torch.float16)
   408:         # Sampling
   409:         pbar = tqdm(self.scheduler.timesteps, desc="SD")
   410:         for i, _ in enumerate(pbar):
   411:             sigma = sigmas[i]
   412:             new_t = self.timestep(sigma).to(self.device)
   413:             
   414:             with torch.no_grad():
   415:                 denoised, _ = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)
   416: 
   417:             sigma_down, sigma_up = self.get_ancestral_step(sigmas[i], sigmas[i + 1])
   418:             if sigma_down == 0:
   419:                 # Euler method
   420:                 d = self.to_d(x, sigmas[i], denoised)
   421:                 x = denoised + d * sigma_down
   422:             else:
   423:                 # DPM-Solver++(2S)
   424:                 t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
   425:                 r = 1 / 2
   426:                 h = t_next - t
   427:                 s = t + r * h
   428:                 x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
   429:                 
   430:                 with torch.no_grad():
   431:                     sigma_s = sigma_fn(s)
   432:                     t_2 = self.timestep(sigma_s).to(self.device)
   433:                     denoised_2, _ = self.kdiffusion_x_to_denoised(x_2, sigma_s, uc, c, cfg_guidance, t_2)
   434:                 
   435:                 x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
   436:             # Noise addition
   437:             if sigmas[i + 1] > 0:
   438:                 x = x + torch.randn_like(x) * sigma_up
   439: 
   440:             if callback_fn is not None:
   441:                 callback_kwargs = { 'z0t': denoised.detach(),
   442:                                     'zt': x.detach(),
   443:                                     'decode': self.decode}
   444:                 callback_kwargs = callback_fn(i, new_t, callback_kwargs)
   445:                 denoised = callback_kwargs["z0t"]
   446:                 x = callback_kwargs["zt"]
   447:         
   448:         # for the last step, do not add noise
   449:         img = self.decode(x)
   450:         img = (img / 2 + 0.5).clamp(0, 1)
   451:         return img.detach().cpu()
   452:     
   453:     
   454: @register_solver("dpm++_2m")
   455: class DPMpp2mCFGSolver(StableDiffusion):
   456:     @torch.autocast(device_type='cuda', dtype=torch.float16)
   457:     def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
   458:         t_fn = lambda sigma: sigma.log().neg()
   459:         sigma_fn = lambda t: t.neg().exp()
   460:         # Text embedding
   461:         uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
   462:         # convert to karras sigma scheduler
   463:         total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
   464:         sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
   465:         # initialize
   466:         x = self.initialize_latent(method="random_kdiffusion",
   467:                                    latent_dim=(1, 4, 64, 64),
   468:                                    sigmas=sigmas).to(torch.float16)
   469:         old_denoised = None # buffer
   470:         # Sampling
   471:         pbar = tqdm(self.scheduler.timesteps, desc="SD")
   472:         for i, _ in enumerate(pbar):
   473:             sigma = sigmas[i]
   474:             new_t = self.timestep(sigma).to(self.device)
   475:             
   476:             with torch.no_grad():
   477:                 denoised, _ = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)
   478: 
   479:             # solve ODE one step
   480:             t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i+1])
   481:             h = t_next - t
   482:             if old_denoised is None or sigmas[i+1] == 0:
   483:                 x = denoised + self.to_d(x, sigmas[i], denoised) * sigmas[i+1]
   484:             else:
   485:                 h_last = t - t_fn(sigmas[i-1])
   486:                 r = h_last / h
   487:                 extra1 = -torch.exp(-h) * denoised - (-h).expm1() * (denoised - old_denoised) / (2*r)
   488:                 extra2 = torch.exp(-h) * x
   489:                 x = denoised + extra1 + extra2
   490:             old_denoised = denoised
   491: 
   492:             if callback_fn is not None:
   493:                 callback_kwargs = { 'z0t': denoised.detach(),
   494:                                     'zt': x.detach(),
   495:                                     'decode': self.decode}
   496:                 callback_kwargs = callback_fn(i, new_t, callback_kwargs)
   497:                 denoised = callback_kwargs["z0t"]
   498:                 x = callback_kwargs["zt"]
   499:         
   500:         # for the last step, do not add noise

[truncated: showing at most 500 lines / 60000 bytes from CFGpp-main/latent_diffusion.py]
```

### `CFGpp-main/latent_sdxl.py`  [EDITABLE — lines 722–764 only]

```python
     1: from typing import Any, Optional, Tuple
     2: import os
     3: from safetensors.torch import load_file
     4: 
     5: import torch
     6: from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
     7: from diffusers.models.attention_processor import (AttnProcessor2_0,
     8:                                                   LoRAAttnProcessor2_0,
     9:                                                   LoRAXFormersAttnProcessor,
    10:                                                   XFormersAttnProcessor)
    11: from tqdm import tqdm
    12: from latent_diffusion import get_sigmas_karras, get_ancestral_step, append_zero
    13: 
    14: ####### Factory #######
    15: __SOLVER__ = {}
    16: 
    17: def register_solver(name: str):
    18:     def wrapper(cls):
    19:         if __SOLVER__.get(name, None) is not None:
    20:             raise ValueError(f"Solver {name} already registered.")
    21:         __SOLVER__[name] = cls
    22:         return cls
    23:     return wrapper
    24: 
    25: def get_solver(name: str, **kwargs):
    26:     if name not in __SOLVER__:
    27:         raise ValueError(f"Solver {name} does not exist.")
    28:     return __SOLVER__[name](**kwargs)
    29: 
    30: ########################
    31: 
    32: class SDXL():
    33:     def __init__(self, 
    34:                  solver_config: dict,
    35:                  model_key:str="stabilityai/stable-diffusion-xl-base-1.0",
    36:                  dtype=torch.float16,
    37:                  device='cuda'):
    38: 
    39:         self.device = device
    40:         # Offline image ships SDXL unet as safetensors (often only the fp16
    41:         # variant) — the default from_pretrained looks for diffusion_pytorch_
    42:         # model.bin and dies (OSError: no .bin found). Prefer the fp16 variant
    43:         # safetensors, fall back to standard safetensors.
    44:         try:
    45:             pipe = StableDiffusionXLPipeline.from_pretrained(
    46:                 model_key, torch_dtype=dtype, variant="fp16", use_safetensors=True).to(device)
    47:         except Exception:
    48:             pipe = StableDiffusionXLPipeline.from_pretrained(
    49:                 model_key, torch_dtype=dtype, use_safetensors=True).to(device)
    50:         self.dtype = dtype
    51: 
    52:         # avoid overflow in float16
    53:         self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(device)
    54: 
    55:         self.tokenizer_1 = pipe.tokenizer
    56:         self.tokenizer_2 = pipe.tokenizer_2
    57:         self.text_enc_1 = pipe.text_encoder
    58:         self.text_enc_2 = pipe.text_encoder_2
    59:         self.unet = pipe.unet
    60: 
    61:         self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    62:         self.default_sample_size = self.unet.config.sample_size
    63: 
    64:         # sampling parameters
    65:         self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
    66:         self.total_alphas = self.scheduler.alphas_cumprod.clone()
    67: 
    68:         self.sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
    69:         self.log_sigmas = self.sigmas.log()
    70: 
    71:         N_ts = len(self.scheduler.timesteps)
    72:         self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
    73:         self.skip = N_ts // solver_config.num_sampling
    74: 
    75:         self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
    76:         self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])
    77: 
    78:     def __call__(self, *args: Any, **kwargs: Any) -> Any:
    79:         self.sample(*args, **kwargs)
    80: 
    81:     def alpha(self, t):
    82:         at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
    83:         return at
    84: 
    85:     @torch.no_grad()
    86:     def _text_embed(self, prompt, tokenizer, text_enc, clip_skip):
    87:         text_inputs = tokenizer(
    88:             prompt,
    89:             padding='max_length',
    90:             max_length=tokenizer.model_max_length,
    91:             truncation=True,
    92:             return_tensors='pt')
    93:         text_input_ids = text_inputs.input_ids
    94:         prompt_embeds = text_enc(text_input_ids.to(self.device), output_hidden_states=True)
    95: 
    96:         pool_prompt_embeds = prompt_embeds[0]
    97:         if clip_skip is None:
    98:             prompt_embeds = prompt_embeds.hidden_states[-2]
    99:         else:
   100:             # +2 because SDXL always indexes from the penultimate layer.
   101:             prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
   102:         return prompt_embeds, pool_prompt_embeds
   103: 
   104:     @torch.no_grad()
   105:     def get_text_embed(self, null_prompt_1, prompt_1, null_prompt_2=None, prompt_2=None, clip_skip=None):
   106:         '''
   107:         At this time, assume that batch_size = 1.
   108:         We should extend the code to batch_size > 1.
   109:         '''        
   110:         # Encode the prompts
   111:         # if prompt_2 is None, set same as prompt_1
   112:         prompt_1 = [prompt_1] if isinstance(prompt_1, str) else prompt_1
   113:         null_prompt_1 = [null_prompt_1] if isinstance(null_prompt_1, str) else null_prompt_1
   114: 
   115: 
   116:         prompt_embed_1, pool_prompt_embed = self._text_embed(prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
   117:         if prompt_2 is None:
   118:             prompt_embed = [prompt_embed_1]
   119:         else:
   120:             # Comment on diffusers' source code:
   121:             # "We are only ALWAYS interested in the pooled output of the final text encoder"
   122:             # i.e. we overwrite the pool_prompt_embed with the new one
   123:             prompt_embed_2, pool_prompt_embed = self._text_embed(prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
   124:             prompt_embed = [prompt_embed_1, prompt_embed_2]
   125:         
   126:         null_embed_1, pool_null_embed = self._text_embed(null_prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
   127:         if null_prompt_2 is None:
   128:             null_embed = [null_embed_1]
   129:         else:
   130:             null_embed_2, pool_null_embed = self._text_embed(null_prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
   131:             null_embed = [null_embed_1, null_embed_2]
   132: 
   133:         # concat embeds from two encoders
   134:         null_prompt_embeds = torch.concat(null_embed, dim=-1)
   135:         prompt_embeds = torch.concat(prompt_embed, dim=-1)
   136: 
   137:         return null_prompt_embeds, prompt_embeds, pool_null_embed, pool_prompt_embed            
   138: 
   139:     # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
   140:     def upcast_vae(self):
   141:         dtype = self.vae.dtype
   142:         self.vae.to(dtype=torch.float32)
   143:         use_torch_2_0_or_xformers = isinstance(
   144:             self.vae.decoder.mid_block.attentions[0].processor,
   145:             (
   146:                 AttnProcessor2_0,
   147:                 XFormersAttnProcessor,
   148:                 LoRAXFormersAttnProcessor,
   149:                 LoRAAttnProcessor2_0,
   150:             ),
   151:         )
   152:         # if xformers or torch_2_0 is used attention block does not need
   153:         # to be in float32 which can save lots of memory
   154:         if use_torch_2_0_or_xformers:
   155:             self.vae.post_quant_conv.to(dtype)
   156:             self.vae.decoder.conv_in.to(dtype)
   157:             self.vae.decoder.mid_block.to(dtype)
   158: 
   159:     @torch.no_grad()
   160:     def encode(self, x):
   161:         return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor 
   162: 
   163:     # @torch.no_grad() 
   164:     def decode(self, zt):
   165:         # make sure the VAE is in float32 mode, as it overflows in float16
   166:         # needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
   167: 
   168:         # if needs_upcasting:
   169:         #     self.upcast_vae()
   170:         #     zt = zt.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
   171: 
   172:         image = self.vae.decode(zt / self.vae.config.scaling_factor).sample.float()
   173:         return image
   174: 
   175: 
   176:     def predict_noise(self, zt, t, uc, c, added_cond_kwargs):
   177:         t_in = t.unsqueeze(0)
   178:         if uc is None:
   179:             noise_c = self.unet(zt, t_in, encoder_hidden_states=c,
   180:                                    added_cond_kwargs=added_cond_kwargs)['sample']
   181:             noise_uc = noise_c
   182:         elif c is None:
   183:             noise_uc = self.unet(zt, t_in, encoder_hidden_states=uc,
   184:                                    added_cond_kwargs=added_cond_kwargs)['sample']
   185:             noise_c = noise_uc
   186:         else:
   187:             c_embed = torch.cat([uc, c], dim=0)
   188:             z_in = torch.cat([zt] * 2)
   189:             t_in = torch.cat([t_in] * 2)
   190:             noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed,
   191:                                    added_cond_kwargs=added_cond_kwargs)['sample']
   192:             noise_uc, noise_c = noise_pred.chunk(2)
   193: 
   194:         return noise_uc, noise_c
   195: 
   196:     def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim):
   197:         add_time_ids = list(original_size+crops_coords_top_left+target_size)
   198:         passed_add_embed_dim = (
   199:             self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
   200:         )
   201:         expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
   202: 
   203:         assert expected_add_embed_dim == passed_add_embed_dim, (
   204:              f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
   205:         )
   206:         add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
   207:         return add_time_ids
   208: 
   209:     @torch.autocast(device_type='cuda', dtype=torch.float16)
   210:     def sample(self,
   211:                prompt1 = ["", ""],
   212:                prompt2 = ["", ""],
   213:                cfg_guidance:float=5.0,
   214:                original_size: Optional[Tuple[int, int]]=None,
   215:                crops_coords_top_left: Tuple[int, int]=(0, 0),
   216:                target_size: Optional[Tuple[int, int]]=None,
   217:                negative_original_size: Optional[Tuple[int, int]]=None,
   218:                negative_crops_coords_top_left: Tuple[int, int]=(0, 0),
   219:                negative_target_size: Optional[Tuple[int, int]]=None,
   220:                clip_skip: Optional[int]=None,
   221:                **kwargs):
   222: 
   223:         # 0. Default height and width to unet
   224:         height = self.default_sample_size * self.vae_scale_factor
   225:         width = self.default_sample_size * self.vae_scale_factor
   226: 
   227:         original_size = original_size or (height, width)
   228:         target_size = target_size or (height, width)
   229: 
   230:         # embedding
   231:         (null_prompt_embeds,
   232:          prompt_embeds,
   233:          pool_null_embed,
   234:          pool_prompt_embed) = self.get_text_embed(prompt1[0], prompt1[1], prompt2[0], prompt2[1], clip_skip)
   235: 
   236:         # prepare kwargs for SDXL
   237:         add_text_embeds = pool_prompt_embed
   238:         add_time_ids = self._get_add_time_ids(
   239:             original_size,
   240:             crops_coords_top_left,
   241:             target_size,
   242:             dtype=prompt_embeds.dtype,
   243:             text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
   244:         )
   245: 
   246:         if negative_original_size is not None and negative_target_size is not None:
   247:             negative_add_time_ids = self._get_add_time_ids(
   248:                 negative_original_size,
   249:                 negative_crops_coords_top_left,
   250:                 negative_target_size,
   251:                 dtype=prompt_embeds.dtype,
   252:                 text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
   253:             )
   254:         else:
   255:             negative_add_time_ids = add_time_ids
   256:         negative_text_embeds = pool_null_embed 
   257: 
   258:         if cfg_guidance != 0.0 and cfg_guidance != 1.0:
   259:             # do cfg
   260:             add_text_embeds = torch.cat([negative_text_embeds, add_text_embeds], dim=0)
   261:             add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
   262: 
   263:         add_cond_kwargs = {
   264:             'text_embeds': add_text_embeds.to(self.device),
   265:             'time_ids': add_time_ids.to(self.device)
   266:         }
   267: 
   268:         # reverse sampling
   269:         zt = self.reverse_process(null_prompt_embeds, prompt_embeds, cfg_guidance, add_cond_kwargs, target_size, **kwargs)
   270: 
   271:         # decode
   272:         with torch.no_grad():
   273:             img = self.decode(zt)
   274:         img = (img / 2 + 0.5).clamp(0, 1)
   275:         return img.detach().cpu()
   276: 
   277:     def initialize_latent(self,
   278:                           method: str='random',
   279:                           src_img: Optional[torch.Tensor]=None,
   280:                           add_cond_kwargs: Optional[dict]=None,
   281:                           **kwargs):
   282:         if method == 'ddim':
   283:             assert src_img is not None, "src_img must be provided for inversion"
   284:             z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
   285:                                kwargs.get('uc'),
   286:                                kwargs.get('c'),
   287:                                kwargs.get('cfg_guidance', 0.0),
   288:                                add_cond_kwargs)
   289:         elif method == 'npi':
   290:             assert src_img is not None, "src_img must be provided for inversion"
   291:             z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
   292:                                kwargs.get('c'),
   293:                                kwargs.get('c'),
   294:                                1.0,
   295:                                add_cond_kwargs)
   296:         elif method == 'random':
   297:             size = kwargs.get('size', (1, 4, 128, 128))
   298:             z = torch.randn(size).to(self.device)
   299:         elif method == 'random_kdiffusion':
   300:             size = kwargs.get('latent_dim', (1, 4, 128, 128))
   301:             sigmas = kwargs.get('sigmas', [14.6146])
   302:             z = torch.randn(size).to(self.device)
   303:             z = z * (sigmas[0] ** 2 + 1) ** 0.5
   304:             #z = z * sigmas[0]
   305:         else: 
   306:             raise NotImplementedError
   307: 
   308:         return z.requires_grad_()
   309:     
   310:     def inversion(self, z0, uc, c, cfg_guidance, add_cond_kwargs):
   311:         # if we use cfg_guidance=0.0 or 1.0 for inversion, add_cond_kwargs must be splitted. 
   312:         if cfg_guidance == 0.0 or cfg_guidance == 1.0:
   313:             add_cond_kwargs['text_embeds'] = add_cond_kwargs['text_embeds'][-1].unsqueeze(0)
   314:             add_cond_kwargs['time_ids'] = add_cond_kwargs['time_ids'][-1].unsqueeze(0)
   315: 
   316:         zt = z0.clone().to(self.device)
   317:         pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM inversion')
   318:         for _, t in enumerate(pbar):
   319:             at = self.alpha(t)
   320:             at_prev = self.alpha(t - self.skip)
   321: 
   322:             with torch.no_grad():
   323:                 noise_uc, noise_c  = self.predict_noise(zt, t, uc, c, add_cond_kwargs)
   324:                 noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   325: 
   326:             z0t = (zt - (1-at_prev).sqrt() * noise_pred) / at_prev.sqrt()
   327:             zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred
   328: 
   329:         return zt
   330:     
   331:     def reverse_process(self, *args, **kwargs):
   332:         raise NotImplementedError
   333: 
   334:     # Belows are for K-diffusion sampling (euler, etc)
   335:     def calculate_input(self, x, sigma):
   336:         return x / (sigma ** 2 + 1) ** 0.5
   337:     
   338:     # Related to the Tweedie's formula in VE
   339:     def calculate_denoised(self, x, model_pred, sigma):
   340:         return x - model_pred * sigma
   341:     
   342:     def sigma_to_t(self, sigma, quantize=None):
   343:         '''Taken from k_diffusion/external.py'''
   344:         quantize = self.quantize if quantize is None else quantize
   345:         total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
   346:         dists = sigma - total_sigmas[:, None]
   347:         if quantize:
   348:             return dists.abs().argmin(dim=0).view(sigma.shape)
   349:         low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=total_sigmas.shape[0] - 2)
   350:         high_idx = low_idx + 1
   351:         low, high = total_sigmas[low_idx], total_sigmas[high_idx]
   352:         w = (low - sigma) / (low - high)
   353:         w = w.clamp(0, 1)
   354:         t = (1 - w) * low_idx + w * high_idx
   355:         return t.view(sigma.shape)
   356:     
   357:     def timestep(self, sigma):
   358:         log_sigma = sigma.log()
   359:         dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
   360:         return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)
   361: 
   362:     def to_d(self, x, sigma, denoised):
   363:         '''converts a denoiser output to a Karras ODE derivative'''
   364:         return (x - denoised) / sigma.item()
   365:     
   366:     def kdiffusion_zt_to_denoised(self, x, sigma, uc, c, cfg_guidance, t, add_cond_kwargs):
   367:         xc = self.calculate_input(x, sigma)
   368:         noise_uc, noise_c = self.predict_noise(xc, t, uc, c, add_cond_kwargs)
   369:         noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   370:         denoised = self.calculate_denoised(x, noise_pred, sigma)
   371:         uncond_denoised = self.calculate_denoised(x, noise_uc, sigma)
   372:         return denoised, uncond_denoised
   373: 
   374: 
   375: class SDXLLightning(SDXL):
   376:     def __init__(self, 
   377:                  solver_config: dict,
   378:                  base_model_key:str="stabilityai/stable-diffusion-xl-base-1.0",
   379:                  #light_model_ckpt:str="ckpt/sdxl_lightning_4step_unet.safetensors",
   380:                  light_model_ckpt:str="ckpt/LEOSAM HelloWorld 极速版_6.0 Lightning.safetensors",
   381:                  dtype=torch.float16,
   382:                  device='cuda'):
   383: 
   384:         self.device = device
   385: 
   386:         # load the student model
   387:         """
   388:         unet = UNet2DConditionModel.from_config(base_model_key, subfolder="unet").to("cuda", torch.float16)
   389:         ext = os.path.splitext(light_model_ckpt)[1]
   390:         if ext == ".safetensors":
   391:             state_dict = load_file(light_model_ckpt)
   392:         else:
   393:             state_dict = torch.load(light_model_ckpt, map_location="cpu")
   394:         print(unet.load_state_dict(state_dict, strict=True))
   395:         unet.requires_grad_(False)
   396:         self.unet = unet
   397:         """
   398: 
   399:         pipe = StableDiffusionXLPipeline.from_single_file(light_model_ckpt, torch_dtype=dtype).to(device)
   400:         self.unet = pipe.unet
   401:         #pipe = StableDiffusionXLPipeline.from_pretrained(base_model_key, unet=self.unet, torch_dtype=dtype).to(device)
   402:         self.dtype = dtype
   403: 
   404:         # avoid overflow in float16
   405:         self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(device)
   406: 
   407:         self.tokenizer_1 = pipe.tokenizer
   408:         self.tokenizer_2 = pipe.tokenizer_2
   409:         self.text_enc_1 = pipe.text_encoder
   410:         self.text_enc_2 = pipe.text_encoder_2
   411: 
   412:         self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
   413:         self.default_sample_size = self.unet.config.sample_size
   414: 
   415:         # sampling parameters
   416:         self.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
   417:         self.total_alphas = self.scheduler.alphas_cumprod.clone()
   418: 
   419:         self.sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
   420:         self.log_sigmas = self.sigmas.log()
   421: 
   422:         N_ts = len(self.scheduler.timesteps)
   423:         self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
   424:         self.skip = N_ts // solver_config.num_sampling
   425: 
   426:         #self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
   427:         self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod]).to(device)
   428: 
   429: 
   430: ###########################################
   431: # Base version
   432: ###########################################
   433: 
   434: @register_solver('ddim')
   435: class BaseDDIM(SDXL):
   436:     def reverse_process(self,
   437:                         null_prompt_embeds,
   438:                         prompt_embeds,
   439:                         cfg_guidance,
   440:                         add_cond_kwargs,
   441:                         shape=(1024, 1024),
   442:                         callback_fn=None,
   443:                         **kwargs):
   444:         #################################
   445:         # Sample region - where to change
   446:         #################################
   447:         # initialize zT
   448:         zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
   449:         
   450:         # sampling
   451:         pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
   452:         for step, t in enumerate(pbar):
   453:             next_t = t - self.skip
   454:             at = self.scheduler.alphas_cumprod[t]
   455:             at_next = self.scheduler.alphas_cumprod[next_t]
   456: 
   457:             with torch.no_grad():
   458:                 noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
   459:                 noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   460:             
   461:             # tweedie
   462:             z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
   463: 
   464:             # add noise
   465:             zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_pred
   466: 
   467:             if callback_fn is not None:
   468:                 callback_kwargs = { 'z0t': z0t.detach(),
   469:                                     'zt': zt.detach(),
   470:                                     'decode': self.decode}
   471:                 callback_kwargs = callback_fn(step, t, callback_kwargs)
   472:                 z0t = callback_kwargs["z0t"]
   473:                 zt = callback_kwargs["zt"]
   474: 
   475:         # for the last stpe, do not add noise
   476:         return z0t
   477: 
   478: @register_solver('euler')
   479: class Euler(SDXL):
   480:     quantize = True
   481:     """
   482:     Karras Euler (VE casted)
   483:     """
   484:     @torch.autocast(device_type='cuda', dtype=torch.float16)
   485:     def reverse_process(self,
   486:                         null_prompt_embeds,
   487:                         prompt_embeds,
   488:                         cfg_guidance,
   489:                         add_cond_kwargs,
   490:                         shape=(1024, 1024),
   491:                         callback_fn=None,
   492:                         **kwargs):
   493:         # convert to karras sigma scheduler
   494:         total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
   495:         sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
   496: 
   497:         # initialize
   498:         zt_dim = (1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor)
   499:         zt = self.initialize_latent(method="random_kdiffusion",
   500:                                    latent_dim=zt_dim,

[truncated: showing at most 500 lines / 60000 bytes from CFGpp-main/latent_sdxl.py]
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `ddim` baseline — editable region  [READ-ONLY — reference implementation]

In `CFGpp-main/latent_diffusion.py`:

```python
Lines 621–678:
   618: # CFG++ version
   619: ###########################################
   620: 
   621: @register_solver("ddim_cfg++")
   622: class BaseDDIMCFGpp(StableDiffusion):
   623:     """
   624:     DDIM sampler with CFG++.
   625:     First-order ODE solver - simple and deterministic.
   626:     """
   627:     def __init__(self,
   628:                  solver_config: Dict,
   629:                  model_key:str="runwayml/stable-diffusion-v1-5",
   630:                  device: Optional[torch.device]=None,
   631:                  **kwargs):
   632:         super().__init__(solver_config, model_key, device, **kwargs)
   633: 
   634:     @torch.autocast(device_type='cuda', dtype=torch.float16)
   635:     def sample(self,
   636:                cfg_guidance=7.5,
   637:                prompt=["",""],
   638:                callback_fn=None,
   639:                **kwargs):
   640: 
   641:         # Text embedding
   642:         uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
   643: 
   644:         # Initialize zT
   645:         zt = self.initialize_latent()
   646:         zt = zt.requires_grad_()
   647: 
   648:         # Sampling
   649:         pbar = tqdm(self.scheduler.timesteps, desc="DDIM")
   650:         for step, t in enumerate(pbar):
   651:             at = self.alpha(t)
   652:             at_prev = self.alpha(t - self.skip)
   653: 
   654:             with torch.no_grad():
   655:                 if cfg_guidance == 1.0:
   656:                     noise_pred = self.predict_noise(zt, t, None, c)[1]
   657:                 else:
   658:                     noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
   659:                     noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   660: 
   661:             # Tweedie: estimate clean image
   662:             z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
   663: 
   664:             # DDIM update: standard CFG renoising
   665:             zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred
   666: 
   667:             if callback_fn is not None:
   668:                 callback_kwargs = {'z0t': z0t.detach(),
   669:                                     'zt': zt.detach(),
   670:                                     'decode': self.decode}
   671:                 callback_kwargs = callback_fn(step, t, callback_kwargs)
   672:                 z0t = callback_kwargs["z0t"]
   673:                 zt = callback_kwargs["zt"]
   674: 
   675:         # Decode final latent
   676:         img = self.decode(z0t)
   677:         img = (img / 2 + 0.5).clamp(0, 1)
   678:         return img.detach().cpu()
   679:     
   680:     
   681: @register_solver("euler_cfg++")
```

### `dpm3m_sde` baseline — editable region  [READ-ONLY — reference implementation]

In `CFGpp-main/latent_diffusion.py`:

```python
Lines 621–706:
   618: # CFG++ version
   619: ###########################################
   620: 
   621: @register_solver("ddim_cfg++")
   622: class BaseDDIMCFGpp(StableDiffusion):
   623:     """DPM-Solver++(3M) SDE with Karras schedule."""
   624: 
   625:     def __init__(self,
   626:                  solver_config: Dict,
   627:                  model_key:str="runwayml/stable-diffusion-v1-5",
   628:                  device: Optional[torch.device]=None,
   629:                  **kwargs):
   630:         super().__init__(solver_config, model_key, device, **kwargs)
   631: 
   632:     @torch.autocast(device_type='cuda', dtype=torch.float16)
   633:     def sample(self,
   634:                cfg_guidance=7.5,
   635:                prompt=["",""],
   636:                callback_fn=None,
   637:                **kwargs):
   638:         t_fn = lambda sigma: sigma.log().neg()
   639: 
   640:         uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
   641: 
   642:         total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
   643:         sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
   644: 
   645:         x = self.initialize_latent(method="random_kdiffusion",
   646:                                    latent_dim=(1, 4, 64, 64),
   647:                                    sigmas=sigmas).to(torch.float16)
   648: 
   649:         eta = 1.2
   650:         denoised_1, denoised_2 = None, None
   651:         h_1, h_2 = None, None
   652: 
   653:         pbar = tqdm(self.scheduler.timesteps, desc="DPM++3M-SDE")
   654:         for i, _ in enumerate(pbar):
   655:             sigma = sigmas[i]
   656:             new_t = self.timestep(sigma).to(self.device)
   657: 
   658:             with torch.no_grad():
   659:                 denoised, _ = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)
   660: 
   661:             if sigmas[i + 1] == 0:
   662:                 x = denoised
   663:             else:
   664:                 t, s = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
   665:                 h = s - t
   666:                 h_eta = h * (eta + 1)
   667: 
   668:                 x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised
   669: 
   670:                 if denoised_1 is not None:
   671:                     phi_2 = h_eta.neg().expm1() / h_eta + 1
   672: 
   673:                     if denoised_2 is None:
   674:                         r = h_1 / h
   675:                         d = (denoised - denoised_1) / r
   676:                         x = x + phi_2 * d
   677:                     else:
   678:                         r0 = h_1 / h
   679:                         r1 = h_2 / h_1
   680:                         d1_0 = (denoised - denoised_1) / r0
   681:                         d1_1 = (denoised_1 - denoised_2) / r1
   682:                         d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
   683:                         d2 = (d1_0 - d1_1) / (r0 + r1)
   684:                         phi_3 = phi_2 / h_eta - 0.5
   685:                         x = x + phi_2 * d1 + phi_3 * d2
   686: 
   687:                 if eta > 0:
   688:                     noise = torch.randn_like(x)
   689:                     x = x + noise * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt()
   690: 
   691:                 denoised_2 = denoised_1
   692:                 denoised_1 = denoised
   693:                 h_2 = h_1
   694:                 h_1 = h
   695: 
   696:             if callback_fn is not None:
   697:                 callback_kwargs = {'z0t': denoised.detach(),
   698:                                     'zt': x.detach(),
   699:                                     'decode': self.decode}
   700:                 callback_kwargs = callback_fn(i, new_t, callback_kwargs)
   701:                 x = callback_kwargs["zt"]
   702: 
   703:         z0t = x
   704:         img = self.decode(z0t)
   705:         img = (img / 2 + 0.5).clamp(0, 1)
   706:         return img.detach().cpu()
   707:     
   708:     
   709: @register_solver("euler_cfg++")
```

### `dpm2s` baseline — editable region  [READ-ONLY — reference implementation]

In `CFGpp-main/latent_diffusion.py`:

```python
Lines 621–690:
   618: # CFG++ version
   619: ###########################################
   620: 
   621: @register_solver("ddim_cfg++")
   622: class BaseDDIMCFGpp(StableDiffusion):
   623:     """DPM++ 2S Ancestral sampler with standard CFG."""
   624:     def __init__(self,
   625:                  solver_config: Dict,
   626:                  model_key:str="runwayml/stable-diffusion-v1-5",
   627:                  device: Optional[torch.device]=None,
   628:                  **kwargs):
   629:         super().__init__(solver_config, model_key, device, **kwargs)
   630: 
   631:     @torch.autocast(device_type='cuda', dtype=torch.float16)
   632:     def sample(self,
   633:                cfg_guidance=7.5,
   634:                prompt=["",""],
   635:                callback_fn=None,
   636:                **kwargs):
   637:         t_fn = lambda sigma: sigma.log().neg()
   638:         sigma_fn = lambda t: t.neg().exp()
   639: 
   640:         uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
   641: 
   642:         # Two evaluations per step: 25 steps = 49 NFE (the last step is Euler).
   643:         n_steps = len(self.scheduler.timesteps) // 2
   644:         total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
   645:         sigmas = get_sigmas_karras(n_steps, total_sigmas.min(), total_sigmas.max(), rho=7.)
   646: 
   647:         x = self.initialize_latent(method="random_kdiffusion",
   648:                                    latent_dim=(1, 4, 64, 64),
   649:                                    sigmas=sigmas).to(torch.float16)
   650: 
   651:         pbar = tqdm(range(n_steps), desc="DPM++2S")
   652:         for i, _ in enumerate(pbar):
   653:             sigma = sigmas[i]
   654:             new_t = self.timestep(sigma).to(self.device)
   655: 
   656:             with torch.no_grad():
   657:                 denoised, _ = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)
   658: 
   659:             sigma_down, sigma_up = self.get_ancestral_step(sigmas[i], sigmas[i + 1])
   660:             if sigma_down == 0:
   661:                 d = self.to_d(x, sigmas[i], denoised)
   662:                 x = denoised + d * sigma_down
   663:             else:
   664:                 t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
   665:                 r = 1 / 2
   666:                 h = t_next - t
   667:                 s = t + r * h
   668:                 x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
   669: 
   670:                 with torch.no_grad():
   671:                     sigma_s = sigma_fn(s)
   672:                     t_2 = self.timestep(sigma_s).to(self.device)
   673:                     denoised_2, _ = self.kdiffusion_x_to_denoised(x_2, sigma_s, uc, c, cfg_guidance, t_2)
   674: 
   675:                 x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
   676: 
   677:             if sigmas[i + 1] > 0:
   678:                 x = x + torch.randn_like(x) * sigma_up
   679: 
   680:             if callback_fn is not None:
   681:                 callback_kwargs = {'z0t': denoised.detach(),
   682:                                     'zt': x.detach(),
   683:                                     'decode': self.decode}
   684:                 callback_kwargs = callback_fn(i, new_t, callback_kwargs)
   685:                 denoised = callback_kwargs["z0t"]
   686:                 x = callback_kwargs["zt"]
   687: 
   688:         img = self.decode(x)
   689:         img = (img / 2 + 0.5).clamp(0, 1)
   690:         return img.detach().cpu()
   691:     
   692:     
   693: @register_solver("euler_cfg++")
```

### `ddim` baseline — editable region  [READ-ONLY — reference implementation]

In `CFGpp-main/latent_sdxl.py`:

```python
Lines 722–757:
   719: # CFG++ version
   720: ###########################################
   721: 
   722: @register_solver("ddim_cfg++")
   723: class BaseDDIMCFGpp(SDXL):
   724:     def reverse_process(self,
   725:                         null_prompt_embeds,
   726:                         prompt_embeds,
   727:                         cfg_guidance,
   728:                         add_cond_kwargs,
   729:                         shape=(1024, 1024),
   730:                         callback_fn=None,
   731:                         **kwargs):
   732:         zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
   733: 
   734:         pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
   735:         for step, t in enumerate(pbar):
   736:             next_t = t - self.skip
   737:             at = self.scheduler.alphas_cumprod[t]
   738:             at_next = self.scheduler.alphas_cumprod[next_t]
   739: 
   740:             with torch.no_grad():
   741:                 noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
   742:                 noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   743: 
   744:             z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
   745: 
   746:             # DDIM: standard CFG renoising
   747:             zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_pred
   748: 
   749:             if callback_fn is not None:
   750:                 callback_kwargs = {'z0t': z0t.detach(),
   751:                                     'zt': zt.detach(),
   752:                                     'decode': self.decode}
   753:                 callback_kwargs = callback_fn(step, t, callback_kwargs)
   754:                 z0t = callback_kwargs["z0t"]
   755:                 zt = callback_kwargs["zt"]
   756: 
   757:         return z0t
   758: 
   759: @register_solver('euler_cfg++')
   760: class EulerCFGpp(SDXL):
```

### `dpm3m_sde` baseline — editable region  [READ-ONLY — reference implementation]

In `CFGpp-main/latent_sdxl.py`:

```python
Lines 722–806:
   719: # CFG++ version
   720: ###########################################
   721: 
   722: @register_solver("ddim_cfg++")
   723: class BaseDDIMCFGpp(SDXL):
   724:     """DPM-Solver++(3M) SDE with Karras schedule for SDXL."""
   725:     quantize = True
   726: 
   727:     def reverse_process(self,
   728:                         null_prompt_embeds,
   729:                         prompt_embeds,
   730:                         cfg_guidance,
   731:                         add_cond_kwargs,
   732:                         shape=(1024, 1024),
   733:                         callback_fn=None,
   734:                         **kwargs):
   735:         t_fn = lambda sigma: sigma.log().neg()
   736: 
   737:         total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
   738:         sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
   739: 
   740:         latent_dim = (1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor)
   741:         x = self.initialize_latent(method="random_kdiffusion",
   742:                                    latent_dim=latent_dim,
   743:                                    sigmas=sigmas).to(torch.float16)
   744: 
   745:         eta = 1.2
   746:         denoised_1, denoised_2 = None, None
   747:         h_1, h_2 = None, None
   748: 
   749:         pbar = tqdm(self.scheduler.timesteps, desc="SDXL-DPM++3M-SDE")
   750:         for i, _ in enumerate(pbar):
   751:             sigma = sigmas[i]
   752:             new_t = self.sigma_to_t(sigma).to(self.device)
   753:             c_in = (1 / (sigma ** 2 + 1)).sqrt()
   754:             c_out = -sigma
   755: 
   756:             with torch.no_grad():
   757:                 noise_uc, noise_c = self.predict_noise(
   758:                     x * c_in, new_t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
   759:                 noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   760: 
   761:             denoised = x + c_out * noise_pred
   762: 
   763:             if sigmas[i + 1] == 0:
   764:                 x = denoised
   765:             else:
   766:                 t, s = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
   767:                 h = s - t
   768:                 h_eta = h * (eta + 1)
   769: 
   770:                 x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised
   771: 
   772:                 if denoised_1 is not None:
   773:                     phi_2 = h_eta.neg().expm1() / h_eta + 1
   774: 
   775:                     if denoised_2 is None:
   776:                         r = h_1 / h
   777:                         d = (denoised - denoised_1) / r
   778:                         x = x + phi_2 * d
   779:                     else:
   780:                         r0 = h_1 / h
   781:                         r1 = h_2 / h_1
   782:                         d1_0 = (denoised - denoised_1) / r0
   783:                         d1_1 = (denoised_1 - denoised_2) / r1
   784:                         d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
   785:                         d2 = (d1_0 - d1_1) / (r0 + r1)
   786:                         phi_3 = phi_2 / h_eta - 0.5
   787:                         x = x + phi_2 * d1 + phi_3 * d2
   788: 
   789:                 if eta > 0:
   790:                     noise = torch.randn_like(x)
   791:                     x = x + noise * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt()
   792: 
   793:                 denoised_2 = denoised_1
   794:                 denoised_1 = denoised
   795:                 h_2 = h_1
   796:                 h_1 = h
   797: 
   798:             if callback_fn is not None:
   799:                 callback_kwargs = {'z0t': denoised.detach(),
   800:                                     'zt': x.detach(),
   801:                                     'decode': self.decode}
   802:                 callback_kwargs = callback_fn(i, new_t, callback_kwargs)
   803:                 denoised = callback_kwargs["z0t"]
   804:                 x = callback_kwargs["zt"]
   805: 
   806:         return x
   807: 
   808: @register_solver('euler_cfg++')
   809: class EulerCFGpp(SDXL):
```

### `dpm2s` baseline — editable region  [READ-ONLY — reference implementation]

In `CFGpp-main/latent_sdxl.py`:

```python
Lines 722–796:
   719: # CFG++ version
   720: ###########################################
   721: 
   722: @register_solver("ddim_cfg++")
   723: class BaseDDIMCFGpp(SDXL):
   724:     quantize = True
   725: 
   726:     def reverse_process(self,
   727:                         null_prompt_embeds,
   728:                         prompt_embeds,
   729:                         cfg_guidance,
   730:                         add_cond_kwargs,
   731:                         shape=(1024, 1024),
   732:                         callback_fn=None,
   733:                         **kwargs):
   734:         t_fn = lambda sigma: sigma.log().neg()
   735:         sigma_fn = lambda t: t.neg().exp()
   736: 
   737:         # Two evaluations per step: 26 timesteps (first and last kept) = 25 steps = 50 NFE.
   738:         ts = self.scheduler.timesteps.int().cpu()
   739:         ts = ts[torch.linspace(0, len(ts) - 1, len(ts) // 2 + 1).round().long()]
   740:         alphas = self.scheduler.alphas_cumprod[ts].cpu()
   741:         sigmas = (1-alphas).sqrt() / alphas.sqrt()
   742: 
   743:         zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
   744:         x = zt * sigmas[0]
   745: 
   746:         pbar = tqdm(ts[:-1], desc='SDXL-DPM++2S')
   747:         for i, _ in enumerate(pbar):
   748:             at = alphas[i]
   749:             sigma = sigmas[i]
   750:             c_in = at.sqrt()
   751:             c_out = -sigma
   752: 
   753:             new_t = self.sigma_to_t(sigma).to(self.device)
   754: 
   755:             with torch.no_grad():
   756:                 noise_uc, noise_c = self.predict_noise(x * c_in, new_t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
   757:                 noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   758: 
   759:             denoised = x + c_out * noise_pred
   760: 
   761:             sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
   762:             if sigma_down == 0:
   763:                 d = (x - denoised) / sigma
   764:                 x = denoised + d * sigma_down
   765:             else:
   766:                 t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
   767:                 r = 1 / 2
   768:                 h = t_next - t
   769:                 s = t + r * h
   770:                 x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
   771: 
   772:                 sigma_s = sigma_fn(s)
   773:                 at_s_idx = min(int((sigma_s**2 / (1 + sigma_s**2)) * len(alphas)), len(alphas)-1)
   774:                 c_in_2 = (1 / (1 + sigma_s**2)).sqrt()
   775:                 c_out_2 = -sigma_s
   776:                 t_2 = self.sigma_to_t(sigma_s).to(self.device)
   777: 
   778:                 with torch.no_grad():
   779:                     noise_uc_2, noise_c_2 = self.predict_noise(x_2 * c_in_2, t_2, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
   780:                     noise_pred_2 = noise_uc_2 + cfg_guidance * (noise_c_2 - noise_uc_2)
   781: 
   782:                 denoised_2 = x_2 + c_out_2 * noise_pred_2
   783:                 x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
   784: 
   785:             if sigmas[i + 1] > 0:
   786:                 x = x + torch.randn_like(x) * sigma_up
   787: 
   788:             if callback_fn is not None:
   789:                 callback_kwargs = {'z0t': denoised.detach(),
   790:                                     'zt': x.detach(),
   791:                                     'decode': self.decode}
   792:                 callback_kwargs = callback_fn(i, new_t, callback_kwargs)
   793:                 denoised = callback_kwargs["z0t"]
   794:                 x = callback_kwargs["zt"]
   795: 
   796:         return x
   797: 
   798: @register_solver('euler_cfg++')
   799: class EulerCFGpp(SDXL):
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

## Time Budget

You have **5 hours** of wall-clock time before submission, covering
everything you do here: reading the code, editing it, and any trial runs
you launch.

Good luck.
