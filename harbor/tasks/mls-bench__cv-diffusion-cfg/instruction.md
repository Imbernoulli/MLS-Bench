# MLS-Bench: cv-diffusion-cfg

# Diffusion Model: Classifier-Free Guidance Optimization

## Objective

Design a classifier-free guidance (CFG) method for text-to-image diffusion
that improves generation quality across Stable Diffusion model variants under
a fixed sampling pipeline.

## Background

Classifier-free guidance (Ho & Salimans, 2022, arXiv:2207.12598) combines
unconditional and conditional noise predictions to trade off prompt alignment
and image quality. The standard formula is:

```
noise_pred = noise_uc + cfg_scale * (noise_c - noise_uc)
```

where `noise_uc` is the unconditional noise prediction, `noise_c` is the
text-conditioned noise prediction, and `cfg_scale` is typically in the range
7.5 – 12.5 for high-quality samples.

Standard CFG has well-documented limitations: it can cause mode collapse, over-
saturated colours, off-manifold sampling trajectories that hurt invertibility,
and a sensitive dependence on guidance scale. Recent work proposes manifold-
constrained alternatives:

- **CFG++** (Chung et al., ICLR 2025, arXiv:2406.08070) — reformulates CFG as
  decomposed reverse diffusion sampling: instead of renoising with the
  guided prediction, renoise with the unconditional prediction, keeping the
  latent on the data manifold and enabling small guidance scales (0 < λ < 1).
- **Zero-init / skip-step variants** — skip the first few sampling steps
  before applying guidance to reduce trajectory error at the highest noise
  levels.

## Implementation Contract

Implement the guidance rule for both Stable Diffusion v1.5 and SDXL by editing
the marked editable regions of two files:

1. **`latent_diffusion.py`** — `BaseDDIMCFGpp` class for SD v1.5
   (`sample()` method). Available helpers:
   `self.get_text_embed()`, `self.initialize_latent()`,
   `self.predict_noise()`, `self.alpha(t)`.
2. **`latent_sdxl.py`** — `BaseDDIMCFGpp` class for SDXL
   (`reverse_process()` method). Available helpers:
   `self.initialize_latent(size=...)`, `self.predict_noise()`,
   `self.scheduler.alphas_cumprod[t]`.

The contribution may change how conditional and unconditional predictions are
combined, how the latent is renoised, or how guidance strength varies with
time, but it should not change the prompt set, model weights, the number of
allowed denoiser evaluations, or evaluation code.

## Baselines

| Baseline   | Description |
|------------|-------------|
| `cfg`      | Standard classifier-free guidance (Ho & Salimans, arXiv:2207.12598): renoise with the guided noise prediction. |
| `cfgpp`    | CFG++ (Chung et al., ICLR 2025, arXiv:2406.08070): renoise with the unconditional noise prediction, keeping the trajectory on the data manifold. |
| `zeroinit` | CFG++ with zero-initialization (skip the first K = 2 sampling steps before applying guidance). |

## Fixed Pipeline

- Models: Stable Diffusion v1.5 and SDXL (frozen weights).
- Sampling: fixed sampler call structure with a fixed step budget.
- Prompts: shared evaluation prompt set across all baselines.

## Evaluation

Evaluation runs the text-to-image sampling pipeline on the model variants
above. Metrics reported:

- **CLIP score** (cosine similarity between generated image and text prompt;
  higher is better).
- **FID** computed against a reference image set (lower is better).

Task scoring uses per-variant FID (lower is better). A good method should
improve image quality without sacrificing the prompt-following behaviour
provided by guidance.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/CFGpp-main/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `CFGpp-main/latent_diffusion.py`
- editable lines **621–679**
- `CFGpp-main/latent_sdxl.py`
- editable lines **713–755**




## Readable Context


### `CFGpp-main/latent_diffusion.py`  [EDITABLE — lines 621–679 only]

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
   240:         uncond_denoised = self.calculate_denoised(x, noise_uc, sigma)
   241:         return denoised, uncond_denoised
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

### `CFGpp-main/latent_sdxl.py`  [EDITABLE — lines 713–755 only]

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
    40:         pipe = StableDiffusionXLPipeline.from_pretrained(model_key, torch_dtype=dtype).to(device)
    41:         self.dtype = dtype
    42: 
    43:         # avoid overflow in float16
    44:         self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(device)
    45: 
    46:         self.tokenizer_1 = pipe.tokenizer
    47:         self.tokenizer_2 = pipe.tokenizer_2
    48:         self.text_enc_1 = pipe.text_encoder
    49:         self.text_enc_2 = pipe.text_encoder_2
    50:         self.unet = pipe.unet
    51: 
    52:         self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    53:         self.default_sample_size = self.unet.config.sample_size
    54: 
    55:         # sampling parameters
    56:         self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
    57:         self.total_alphas = self.scheduler.alphas_cumprod.clone()
    58: 
    59:         self.sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
    60:         self.log_sigmas = self.sigmas.log()
    61: 
    62:         N_ts = len(self.scheduler.timesteps)
    63:         self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
    64:         self.skip = N_ts // solver_config.num_sampling
    65: 
    66:         self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
    67:         self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])
    68: 
    69:     def __call__(self, *args: Any, **kwargs: Any) -> Any:
    70:         self.sample(*args, **kwargs)
    71: 
    72:     def alpha(self, t):
    73:         at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
    74:         return at
    75: 
    76:     @torch.no_grad()
    77:     def _text_embed(self, prompt, tokenizer, text_enc, clip_skip):
    78:         text_inputs = tokenizer(
    79:             prompt,
    80:             padding='max_length',
    81:             max_length=tokenizer.model_max_length,
    82:             truncation=True,
    83:             return_tensors='pt')
    84:         text_input_ids = text_inputs.input_ids
    85:         prompt_embeds = text_enc(text_input_ids.to(self.device), output_hidden_states=True)
    86: 
    87:         pool_prompt_embeds = prompt_embeds[0]
    88:         if clip_skip is None:
    89:             prompt_embeds = prompt_embeds.hidden_states[-2]
    90:         else:
    91:             # +2 because SDXL always indexes from the penultimate layer.
    92:             prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
    93:         return prompt_embeds, pool_prompt_embeds
    94: 
    95:     @torch.no_grad()
    96:     def get_text_embed(self, null_prompt_1, prompt_1, null_prompt_2=None, prompt_2=None, clip_skip=None):
    97:         '''
    98:         At this time, assume that batch_size = 1.
    99:         We should extend the code to batch_size > 1.
   100:         '''        
   101:         # Encode the prompts
   102:         # if prompt_2 is None, set same as prompt_1
   103:         prompt_1 = [prompt_1] if isinstance(prompt_1, str) else prompt_1
   104:         null_prompt_1 = [null_prompt_1] if isinstance(null_prompt_1, str) else null_prompt_1
   105: 
   106: 
   107:         prompt_embed_1, pool_prompt_embed = self._text_embed(prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
   108:         if prompt_2 is None:
   109:             prompt_embed = [prompt_embed_1]
   110:         else:
   111:             # Comment on diffusers' source code:
   112:             # "We are only ALWAYS interested in the pooled output of the final text encoder"
   113:             # i.e. we overwrite the pool_prompt_embed with the new one
   114:             prompt_embed_2, pool_prompt_embed = self._text_embed(prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
   115:             prompt_embed = [prompt_embed_1, prompt_embed_2]
   116:         
   117:         null_embed_1, pool_null_embed = self._text_embed(null_prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
   118:         if null_prompt_2 is None:
   119:             null_embed = [null_embed_1]
   120:         else:
   121:             null_embed_2, pool_null_embed = self._text_embed(null_prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
   122:             null_embed = [null_embed_1, null_embed_2]
   123: 
   124:         # concat embeds from two encoders
   125:         null_prompt_embeds = torch.concat(null_embed, dim=-1)
   126:         prompt_embeds = torch.concat(prompt_embed, dim=-1)
   127: 
   128:         return null_prompt_embeds, prompt_embeds, pool_null_embed, pool_prompt_embed            
   129: 
   130:     # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
   131:     def upcast_vae(self):
   132:         dtype = self.vae.dtype
   133:         self.vae.to(dtype=torch.float32)
   134:         use_torch_2_0_or_xformers = isinstance(
   135:             self.vae.decoder.mid_block.attentions[0].processor,
   136:             (
   137:                 AttnProcessor2_0,
   138:                 XFormersAttnProcessor,
   139:                 LoRAXFormersAttnProcessor,
   140:                 LoRAAttnProcessor2_0,
   141:             ),
   142:         )
   143:         # if xformers or torch_2_0 is used attention block does not need
   144:         # to be in float32 which can save lots of memory
   145:         if use_torch_2_0_or_xformers:
   146:             self.vae.post_quant_conv.to(dtype)
   147:             self.vae.decoder.conv_in.to(dtype)
   148:             self.vae.decoder.mid_block.to(dtype)
   149: 
   150:     @torch.no_grad()
   151:     def encode(self, x):
   152:         return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor 
   153: 
   154:     # @torch.no_grad() 
   155:     def decode(self, zt):
   156:         # make sure the VAE is in float32 mode, as it overflows in float16
   157:         # needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
   158: 
   159:         # if needs_upcasting:
   160:         #     self.upcast_vae()
   161:         #     zt = zt.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
   162: 
   163:         image = self.vae.decode(zt / self.vae.config.scaling_factor).sample.float()
   164:         return image
   165: 
   166: 
   167:     def predict_noise(self, zt, t, uc, c, added_cond_kwargs):
   168:         t_in = t.unsqueeze(0)
   169:         if uc is None:
   170:             noise_c = self.unet(zt, t_in, encoder_hidden_states=c,
   171:                                    added_cond_kwargs=added_cond_kwargs)['sample']
   172:             noise_uc = noise_c
   173:         elif c is None:
   174:             noise_uc = self.unet(zt, t_in, encoder_hidden_states=uc,
   175:                                    added_cond_kwargs=added_cond_kwargs)['sample']
   176:             noise_c = noise_uc
   177:         else:
   178:             c_embed = torch.cat([uc, c], dim=0)
   179:             z_in = torch.cat([zt] * 2)
   180:             t_in = torch.cat([t_in] * 2)
   181:             noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed,
   182:                                    added_cond_kwargs=added_cond_kwargs)['sample']
   183:             noise_uc, noise_c = noise_pred.chunk(2)
   184: 
   185:         return noise_uc, noise_c
   186: 
   187:     def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim):
   188:         add_time_ids = list(original_size+crops_coords_top_left+target_size)
   189:         passed_add_embed_dim = (
   190:             self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
   191:         )
   192:         expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
   193: 
   194:         assert expected_add_embed_dim == passed_add_embed_dim, (
   195:              f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
   196:         )
   197:         add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
   198:         return add_time_ids
   199: 
   200:     @torch.autocast(device_type='cuda', dtype=torch.float16)
   201:     def sample(self,
   202:                prompt1 = ["", ""],
   203:                prompt2 = ["", ""],
   204:                cfg_guidance:float=5.0,
   205:                original_size: Optional[Tuple[int, int]]=None,
   206:                crops_coords_top_left: Tuple[int, int]=(0, 0),
   207:                target_size: Optional[Tuple[int, int]]=None,
   208:                negative_original_size: Optional[Tuple[int, int]]=None,
   209:                negative_crops_coords_top_left: Tuple[int, int]=(0, 0),
   210:                negative_target_size: Optional[Tuple[int, int]]=None,
   211:                clip_skip: Optional[int]=None,
   212:                **kwargs):
   213: 
   214:         # 0. Default height and width to unet
   215:         height = self.default_sample_size * self.vae_scale_factor
   216:         width = self.default_sample_size * self.vae_scale_factor
   217: 
   218:         original_size = original_size or (height, width)
   219:         target_size = target_size or (height, width)
   220: 
   221:         # embedding
   222:         (null_prompt_embeds,
   223:          prompt_embeds,
   224:          pool_null_embed,
   225:          pool_prompt_embed) = self.get_text_embed(prompt1[0], prompt1[1], prompt2[0], prompt2[1], clip_skip)
   226: 
   227:         # prepare kwargs for SDXL
   228:         add_text_embeds = pool_prompt_embed
   229:         add_time_ids = self._get_add_time_ids(
   230:             original_size,
   231:             crops_coords_top_left,
   232:             target_size,
   233:             dtype=prompt_embeds.dtype,
   234:             text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
   235:         )
   236: 
   237:         if negative_original_size is not None and negative_target_size is not None:
   238:             negative_add_time_ids = self._get_add_time_ids(
   239:                 negative_original_size,
   240:                 negative_crops_coords_top_left,
   241:                 negative_target_size,
   242:                 dtype=prompt_embeds.dtype,
   243:                 text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
   244:             )
   245:         else:
   246:             negative_add_time_ids = add_time_ids
   247:         negative_text_embeds = pool_null_embed 
   248: 
   249:         if cfg_guidance != 0.0 and cfg_guidance != 1.0:
   250:             # do cfg
   251:             add_text_embeds = torch.cat([negative_text_embeds, add_text_embeds], dim=0)
   252:             add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
   253: 
   254:         add_cond_kwargs = {
   255:             'text_embeds': add_text_embeds.to(self.device),
   256:             'time_ids': add_time_ids.to(self.device)
   257:         }
   258: 
   259:         # reverse sampling
   260:         zt = self.reverse_process(null_prompt_embeds, prompt_embeds, cfg_guidance, add_cond_kwargs, target_size, **kwargs)
   261: 
   262:         # decode
   263:         with torch.no_grad():
   264:             img = self.decode(zt)
   265:         img = (img / 2 + 0.5).clamp(0, 1)
   266:         return img.detach().cpu()
   267: 
   268:     def initialize_latent(self,
   269:                           method: str='random',
   270:                           src_img: Optional[torch.Tensor]=None,
   271:                           add_cond_kwargs: Optional[dict]=None,
   272:                           **kwargs):
   273:         if method == 'ddim':
   274:             assert src_img is not None, "src_img must be provided for inversion"
   275:             z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
   276:                                kwargs.get('uc'),
   277:                                kwargs.get('c'),
   278:                                kwargs.get('cfg_guidance', 0.0),
   279:                                add_cond_kwargs)
   280:         elif method == 'npi':
   281:             assert src_img is not None, "src_img must be provided for inversion"
   282:             z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
   283:                                kwargs.get('c'),
   284:                                kwargs.get('c'),
   285:                                1.0,
   286:                                add_cond_kwargs)
   287:         elif method == 'random':
   288:             size = kwargs.get('size', (1, 4, 128, 128))
   289:             z = torch.randn(size).to(self.device)
   290:         elif method == 'random_kdiffusion':
   291:             size = kwargs.get('latent_dim', (1, 4, 128, 128))
   292:             sigmas = kwargs.get('sigmas', [14.6146])
   293:             z = torch.randn(size).to(self.device)
   294:             z = z * (sigmas[0] ** 2 + 1) ** 0.5
   295:             #z = z * sigmas[0]
   296:         else: 
   297:             raise NotImplementedError
   298: 
   299:         return z.requires_grad_()
   300:     
   301:     def inversion(self, z0, uc, c, cfg_guidance, add_cond_kwargs):
   302:         # if we use cfg_guidance=0.0 or 1.0 for inversion, add_cond_kwargs must be splitted. 
   303:         if cfg_guidance == 0.0 or cfg_guidance == 1.0:
   304:             add_cond_kwargs['text_embeds'] = add_cond_kwargs['text_embeds'][-1].unsqueeze(0)
   305:             add_cond_kwargs['time_ids'] = add_cond_kwargs['time_ids'][-1].unsqueeze(0)
   306: 
   307:         zt = z0.clone().to(self.device)
   308:         pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM inversion')
   309:         for _, t in enumerate(pbar):
   310:             at = self.alpha(t)
   311:             at_prev = self.alpha(t - self.skip)
   312: 
   313:             with torch.no_grad():
   314:                 noise_uc, noise_c  = self.predict_noise(zt, t, uc, c, add_cond_kwargs)
   315:                 noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   316: 
   317:             z0t = (zt - (1-at_prev).sqrt() * noise_pred) / at_prev.sqrt()
   318:             zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred
   319: 
   320:         return zt
   321:     
   322:     def reverse_process(self, *args, **kwargs):
   323:         raise NotImplementedError
   324: 
   325:     # Belows are for K-diffusion sampling (euler, etc)
   326:     def calculate_input(self, x, sigma):
   327:         return x / (sigma ** 2 + 1) ** 0.5
   328:     
   329:     # Related to the Tweedie's formula in VE
   330:     def calculate_denoised(self, x, model_pred, sigma):
   331:         return x - model_pred * sigma
   332:     
   333:     def sigma_to_t(self, sigma, quantize=None):
   334:         '''Taken from k_diffusion/external.py'''
   335:         quantize = self.quantize if quantize is None else quantize
   336:         total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
   337:         dists = sigma - total_sigmas[:, None]
   338:         if quantize:
   339:             return dists.abs().argmin(dim=0).view(sigma.shape)
   340:         low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=total_sigmas.shape[0] - 2)
   341:         high_idx = low_idx + 1
   342:         low, high = total_sigmas[low_idx], total_sigmas[high_idx]
   343:         w = (low - sigma) / (low - high)
   344:         w = w.clamp(0, 1)
   345:         t = (1 - w) * low_idx + w * high_idx
   346:         return t.view(sigma.shape)
   347:     
   348:     def timestep(self, sigma):
   349:         log_sigma = sigma.log()
   350:         dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
   351:         return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)
   352: 
   353:     def to_d(self, x, sigma, denoised):
   354:         '''converts a denoiser output to a Karras ODE derivative'''
   355:         return (x - denoised) / sigma.item()
   356:     
   357:     def kdiffusion_zt_to_denoised(self, x, sigma, uc, c, cfg_guidance, t, add_cond_kwargs):
   358:         xc = self.calculate_input(x, sigma)
   359:         noise_uc, noise_c = self.predict_noise(xc, t, uc, c, add_cond_kwargs)
   360:         noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   361:         denoised = self.calculate_denoised(x, noise_pred, sigma)
   362:         uncond_denoised = self.calculate_denoised(x, noise_uc, sigma)
   363:         return denoised, uncond_denoised
   364: 
   365: 
   366: class SDXLLightning(SDXL):
   367:     def __init__(self, 
   368:                  solver_config: dict,
   369:                  base_model_key:str="stabilityai/stable-diffusion-xl-base-1.0",
   370:                  #light_model_ckpt:str="ckpt/sdxl_lightning_4step_unet.safetensors",
   371:                  light_model_ckpt:str="ckpt/LEOSAM HelloWorld 极速版_6.0 Lightning.safetensors",
   372:                  dtype=torch.float16,
   373:                  device='cuda'):
   374: 
   375:         self.device = device
   376: 
   377:         # load the student model
   378:         """
   379:         unet = UNet2DConditionModel.from_config(base_model_key, subfolder="unet").to("cuda", torch.float16)
   380:         ext = os.path.splitext(light_model_ckpt)[1]
   381:         if ext == ".safetensors":
   382:             state_dict = load_file(light_model_ckpt)
   383:         else:
   384:             state_dict = torch.load(light_model_ckpt, map_location="cpu")
   385:         print(unet.load_state_dict(state_dict, strict=True))
   386:         unet.requires_grad_(False)
   387:         self.unet = unet
   388:         """
   389: 
   390:         pipe = StableDiffusionXLPipeline.from_single_file(light_model_ckpt, torch_dtype=dtype).to(device)
   391:         self.unet = pipe.unet
   392:         #pipe = StableDiffusionXLPipeline.from_pretrained(base_model_key, unet=self.unet, torch_dtype=dtype).to(device)
   393:         self.dtype = dtype
   394: 
   395:         # avoid overflow in float16
   396:         self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(device)
   397: 
   398:         self.tokenizer_1 = pipe.tokenizer
   399:         self.tokenizer_2 = pipe.tokenizer_2
   400:         self.text_enc_1 = pipe.text_encoder
   401:         self.text_enc_2 = pipe.text_encoder_2
   402: 
   403:         self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
   404:         self.default_sample_size = self.unet.config.sample_size
   405: 
   406:         # sampling parameters
   407:         self.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
   408:         self.total_alphas = self.scheduler.alphas_cumprod.clone()
   409: 
   410:         self.sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
   411:         self.log_sigmas = self.sigmas.log()
   412: 
   413:         N_ts = len(self.scheduler.timesteps)
   414:         self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
   415:         self.skip = N_ts // solver_config.num_sampling
   416: 
   417:         #self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
   418:         self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod]).to(device)
   419: 
   420: 
   421: ###########################################
   422: # Base version
   423: ###########################################
   424: 
   425: @register_solver('ddim')
   426: class BaseDDIM(SDXL):
   427:     def reverse_process(self,
   428:                         null_prompt_embeds,
   429:                         prompt_embeds,
   430:                         cfg_guidance,
   431:                         add_cond_kwargs,
   432:                         shape=(1024, 1024),
   433:                         callback_fn=None,
   434:                         **kwargs):
   435:         #################################
   436:         # Sample region - where to change
   437:         #################################
   438:         # initialize zT
   439:         zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
   440:         
   441:         # sampling
   442:         pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
   443:         for step, t in enumerate(pbar):
   444:             next_t = t - self.skip
   445:             at = self.scheduler.alphas_cumprod[t]
   446:             at_next = self.scheduler.alphas_cumprod[next_t]
   447: 
   448:             with torch.no_grad():
   449:                 noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
   450:                 noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   451:             
   452:             # tweedie
   453:             z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
   454: 
   455:             # add noise
   456:             zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_pred
   457: 
   458:             if callback_fn is not None:
   459:                 callback_kwargs = { 'z0t': z0t.detach(),
   460:                                     'zt': zt.detach(),
   461:                                     'decode': self.decode}
   462:                 callback_kwargs = callback_fn(step, t, callback_kwargs)
   463:                 z0t = callback_kwargs["z0t"]
   464:                 zt = callback_kwargs["zt"]
   465: 
   466:         # for the last stpe, do not add noise
   467:         return z0t
   468: 
   469: @register_solver('euler')
   470: class Euler(SDXL):
   471:     quantize = True
   472:     """
   473:     Karras Euler (VE casted)
   474:     """
   475:     @torch.autocast(device_type='cuda', dtype=torch.float16)
   476:     def reverse_process(self,
   477:                         null_prompt_embeds,
   478:                         prompt_embeds,
   479:                         cfg_guidance,
   480:                         add_cond_kwargs,
   481:                         shape=(1024, 1024),
   482:                         callback_fn=None,
   483:                         **kwargs):
   484:         # convert to karras sigma scheduler
   485:         total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
   486:         sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
   487: 
   488:         # initialize
   489:         zt_dim = (1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor)
   490:         zt = self.initialize_latent(method="random_kdiffusion",
   491:                                    latent_dim=zt_dim,
   492:                                    sigmas=sigmas).to(torch.float16)
   493:         
   494:         # sampling
   495:         pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
   496:         for step, t in enumerate(pbar):
   497:             sigma = sigmas[step]
   498:             t = self.timestep(sigma).to(self.device)
   499: 
   500:             with torch.no_grad():

[truncated: showing at most 500 lines / 60000 bytes from CFGpp-main/latent_sdxl.py]
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `cfg` baseline — editable region  [READ-ONLY — reference implementation]

In `CFGpp-main/latent_diffusion.py`:

```python
Lines 621–676:
   618: # CFG++ version
   619: ###########################################
   620: 
   621: @register_solver("ddim_cfg++")
   622: class BaseDDIMCFGpp(StableDiffusion):
   623:     """
   624:     DDIM solver for SD with standard CFG.
   625:     """
   626:     def __init__(self,
   627:                  solver_config: Dict,
   628:                  model_key:str="runwayml/stable-diffusion-v1-5",
   629:                  device: Optional[torch.device]=None,
   630:                  **kwargs):
   631:         super().__init__(solver_config, model_key, device, **kwargs)
   632: 
   633:     @torch.autocast(device_type='cuda', dtype=torch.float16)
   634:     def sample(self,
   635:                cfg_guidance=7.5,
   636:                prompt=["",""],
   637:                callback_fn=None,
   638:                **kwargs):
   639:         # Standard CFG needs higher guidance scale
   640:         cfg_guidance = 7.5
   641: 
   642:         # Text embedding
   643:         uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
   644: 
   645:         # Initialize zT
   646:         zt = self.initialize_latent()
   647:         zt = zt.requires_grad_()
   648: 
   649:         # Sampling
   650:         pbar = tqdm(self.scheduler.timesteps, desc="SD")
   651:         for step, t in enumerate(pbar):
   652:             at = self.alpha(t)
   653:             at_prev = self.alpha(t - self.skip)
   654: 
   655:             with torch.no_grad():
   656:                 noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
   657:                 noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   658: 
   659:             # tweedie
   660:             z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
   661: 
   662:             # add noise - STANDARD CFG: use noise_pred
   663:             zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred
   664: 
   665:             if callback_fn is not None:
   666:                 callback_kwargs = {'z0t': z0t.detach(),
   667:                                     'zt': zt.detach(),
   668:                                     'decode': self.decode}
   669:                 callback_kwargs = callback_fn(step, t, callback_kwargs)
   670:                 z0t = callback_kwargs["z0t"]
   671:                 zt = callback_kwargs["zt"]
   672: 
   673:         # for the last step, do not add noise
   674:         img = self.decode(z0t)
   675:         img = (img / 2 + 0.5).clamp(0, 1)
   676:         return img.detach().cpu()
   677:     
   678:     
   679: @register_solver("euler_cfg++")
```

### `cfgpp` baseline — editable region  [READ-ONLY — reference implementation]

In `CFGpp-main/latent_diffusion.py`:

```python
Lines 621–674:
   618: # CFG++ version
   619: ###########################################
   620: 
   621: @register_solver("ddim_cfg++")
   622: class BaseDDIMCFGpp(StableDiffusion):
   623:     """
   624:     DDIM solver for SD with CFG++.
   625:     """
   626:     def __init__(self,
   627:                  solver_config: Dict,
   628:                  model_key:str="runwayml/stable-diffusion-v1-5",
   629:                  device: Optional[torch.device]=None,
   630:                  **kwargs):
   631:         super().__init__(solver_config, model_key, device, **kwargs)
   632: 
   633:     @torch.autocast(device_type='cuda', dtype=torch.float16)
   634:     def sample(self,
   635:                cfg_guidance=7.5,
   636:                prompt=["",""],
   637:                callback_fn=None,
   638:                **kwargs):
   639: 
   640:         # Text embedding
   641:         uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
   642: 
   643:         # Initialize zT
   644:         zt = self.initialize_latent()
   645:         zt = zt.requires_grad_()
   646: 
   647:         # Sampling
   648:         pbar = tqdm(self.scheduler.timesteps, desc="SD")
   649:         for step, t in enumerate(pbar):
   650:             at = self.alpha(t)
   651:             at_prev = self.alpha(t - self.skip)
   652: 
   653:             with torch.no_grad():
   654:                 noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
   655:                 noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   656: 
   657:             # tweedie
   658:             z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
   659: 
   660:             # add noise - CFG++: use noise_uc to stay on manifold
   661:             zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc
   662: 
   663:             if callback_fn is not None:
   664:                 callback_kwargs = {'z0t': z0t.detach(),
   665:                                     'zt': zt.detach(),
   666:                                     'decode': self.decode}
   667:                 callback_kwargs = callback_fn(step, t, callback_kwargs)
   668:                 z0t = callback_kwargs["z0t"]
   669:                 zt = callback_kwargs["zt"]
   670: 
   671:         # for the last step, do not add noise
   672:         img = self.decode(z0t)
   673:         img = (img / 2 + 0.5).clamp(0, 1)
   674:         return img.detach().cpu()
   675:     
   676:     
   677: @register_solver("euler_cfg++")
```

### `zeroinit` baseline — editable region  [READ-ONLY — reference implementation]

In `CFGpp-main/latent_diffusion.py`:

```python
Lines 621–681:
   618: # CFG++ version
   619: ###########################################
   620: 
   621: @register_solver("ddim_cfg++")
   622: class BaseDDIMCFGpp(StableDiffusion):
   623:     """
   624:     DDIM solver for SD with CFG++ and Zero-init.
   625:     """
   626:     def __init__(self,
   627:                  solver_config: Dict,
   628:                  model_key:str="runwayml/stable-diffusion-v1-5",
   629:                  device: Optional[torch.device]=None,
   630:                  **kwargs):
   631:         super().__init__(solver_config, model_key, device, **kwargs)
   632: 
   633:     @torch.autocast(device_type='cuda', dtype=torch.float16)
   634:     def sample(self,
   635:                cfg_guidance=7.5,
   636:                prompt=["",""],
   637:                callback_fn=None,
   638:                **kwargs):
   639: 
   640:         # Text embedding
   641:         uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
   642: 
   643:         # Initialize zT
   644:         zt = self.initialize_latent()
   645:         zt = zt.requires_grad_()
   646: 
   647:         # Zero-init parameter
   648:         K = 2  # Skip first K steps
   649: 
   650:         # Sampling
   651:         pbar = tqdm(self.scheduler.timesteps, desc="SD")
   652:         for step, t in enumerate(pbar):
   653:             # Zero-init: skip first K steps
   654:             if step < K:
   655:                 continue
   656: 
   657:             at = self.alpha(t)
   658:             at_prev = self.alpha(t - self.skip)
   659: 
   660:             with torch.no_grad():
   661:                 noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
   662:                 noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   663: 
   664:             # tweedie
   665:             z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
   666: 
   667:             # add noise - CFG++: use noise_uc to stay on manifold
   668:             zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc
   669: 
   670:             if callback_fn is not None:
   671:                 callback_kwargs = {'z0t': z0t.detach(),
   672:                                     'zt': zt.detach(),
   673:                                     'decode': self.decode}
   674:                 callback_kwargs = callback_fn(step, t, callback_kwargs)
   675:                 z0t = callback_kwargs["z0t"]
   676:                 zt = callback_kwargs["zt"]
   677: 
   678:         # for the last step, do not add noise
   679:         img = self.decode(z0t)
   680:         img = (img / 2 + 0.5).clamp(0, 1)
   681:         return img.detach().cpu()
   682:     
   683:     
   684: @register_solver("euler_cfg++")
```

### `cfg` baseline — editable region  [READ-ONLY — reference implementation]

In `CFGpp-main/latent_sdxl.py`:

```python
Lines 713–751:
   710: # CFG++ version
   711: ###########################################
   712: 
   713: @register_solver("ddim_cfg++")
   714: class BaseDDIMCFGpp(SDXL):
   715:     def reverse_process(self,
   716:                         null_prompt_embeds,
   717:                         prompt_embeds,
   718:                         cfg_guidance,
   719:                         add_cond_kwargs,
   720:                         shape=(1024, 1024),
   721:                         callback_fn=None,
   722:                         **kwargs):
   723:         # Standard CFG needs higher guidance scale
   724:         cfg_guidance = 7.5
   725: 
   726:         zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
   727: 
   728:         pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
   729:         for step, t in enumerate(pbar):
   730:             next_t = t - self.skip
   731:             at = self.scheduler.alphas_cumprod[t]
   732:             at_next = self.scheduler.alphas_cumprod[next_t]
   733: 
   734:             with torch.no_grad():
   735:                 noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
   736:                 noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   737: 
   738:             z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
   739: 
   740:             # STANDARD CFG: use noise_pred
   741:             zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_pred
   742: 
   743:             if callback_fn is not None:
   744:                 callback_kwargs = {'z0t': z0t.detach(),
   745:                                     'zt': zt.detach(),
   746:                                     'decode': self.decode}
   747:                 callback_kwargs = callback_fn(step, t, callback_kwargs)
   748:                 z0t = callback_kwargs["z0t"]
   749:                 zt = callback_kwargs["zt"]
   750: 
   751:         return z0t
   752: 
   753: @register_solver('euler_cfg++')
   754: class EulerCFGpp(SDXL):
```

### `cfgpp` baseline — editable region  [READ-ONLY — reference implementation]

In `CFGpp-main/latent_sdxl.py`:

```python
Lines 713–748:
   710: # CFG++ version
   711: ###########################################
   712: 
   713: @register_solver("ddim_cfg++")
   714: class BaseDDIMCFGpp(SDXL):
   715:     def reverse_process(self,
   716:                         null_prompt_embeds,
   717:                         prompt_embeds,
   718:                         cfg_guidance,
   719:                         add_cond_kwargs,
   720:                         shape=(1024, 1024),
   721:                         callback_fn=None,
   722:                         **kwargs):
   723:         zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
   724: 
   725:         pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
   726:         for step, t in enumerate(pbar):
   727:             next_t = t - self.skip
   728:             at = self.scheduler.alphas_cumprod[t]
   729:             at_next = self.scheduler.alphas_cumprod[next_t]
   730: 
   731:             with torch.no_grad():
   732:                 noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
   733:                 noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   734: 
   735:             z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
   736: 
   737:             # CFG++: use noise_uc to stay on manifold
   738:             zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_uc
   739: 
   740:             if callback_fn is not None:
   741:                 callback_kwargs = {'z0t': z0t.detach(),
   742:                                     'zt': zt.detach(),
   743:                                     'decode': self.decode}
   744:                 callback_kwargs = callback_fn(step, t, callback_kwargs)
   745:                 z0t = callback_kwargs["z0t"]
   746:                 zt = callback_kwargs["zt"]
   747: 
   748:         return z0t
   749: 
   750: @register_solver('euler_cfg++')
   751: class EulerCFGpp(SDXL):
```

### `zeroinit` baseline — editable region  [READ-ONLY — reference implementation]

In `CFGpp-main/latent_sdxl.py`:

```python
Lines 713–753:
   710: # CFG++ version
   711: ###########################################
   712: 
   713: @register_solver("ddim_cfg++")
   714: class BaseDDIMCFGpp(SDXL):
   715:     def reverse_process(self,
   716:                         null_prompt_embeds,
   717:                         prompt_embeds,
   718:                         cfg_guidance,
   719:                         add_cond_kwargs,
   720:                         shape=(1024, 1024),
   721:                         callback_fn=None,
   722:                         **kwargs):
   723:         zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
   724: 
   725:         K = 2  # Skip first K steps
   726: 
   727:         pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
   728:         for step, t in enumerate(pbar):
   729:             if step < K:
   730:                 continue
   731: 
   732:             next_t = t - self.skip
   733:             at = self.scheduler.alphas_cumprod[t]
   734:             at_next = self.scheduler.alphas_cumprod[next_t]
   735: 
   736:             with torch.no_grad():
   737:                 noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
   738:                 noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
   739: 
   740:             z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
   741: 
   742:             # CFG++: use noise_uc to stay on manifold
   743:             zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_uc
   744: 
   745:             if callback_fn is not None:
   746:                 callback_kwargs = {'z0t': z0t.detach(),
   747:                                     'zt': zt.detach(),
   748:                                     'decode': self.decode}
   749:                 callback_kwargs = callback_fn(step, t, callback_kwargs)
   750:                 z0t = callback_kwargs["z0t"]
   751:                 zt = callback_kwargs["zt"]
   752: 
   753:         return z0t
   754: 
   755: @register_solver('euler_cfg++')
   756: class EulerCFGpp(SDXL):
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
