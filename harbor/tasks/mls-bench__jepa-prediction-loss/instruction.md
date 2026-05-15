# MLS-Bench: jepa-prediction-loss

# Temporal JEPA Prediction Loss Optimization

## Research Question
Design a better prediction cost function for multi-step temporal Joint Embedding Predictive Architecture (JEPA). The prediction loss measures discrepancy between predicted and target representations in the latent space, directly influencing how well the predictor learns to model temporal dynamics.

## Background
JEPA-style self-supervised models (Assran et al., I-JEPA, CVPR 2023, arXiv:2301.08243) train an encoder and a predictor jointly, with the predictor matching latent representations of context and target. In the temporal extension used here, the encoder produces a spatial feature map for each frame and the predictor operates autoregressively over time on a Moving MNIST sequence. The training loss is the sum of:
- a **prediction loss** comparing predicted and target latent feature maps (the component you redesign), and
- a **VCLoss (Variance–Covariance) regularizer** that prevents collapse, in the spirit of VICReg (Bardes, Ponce, LeCun, ICLR 2022, arXiv:2105.04906).

The current baseline uses a plain `F.mse_loss(state, predicted)`, which treats every channel and spatial location identically and ignores temporal structure.

## What You Can Modify
The `CustomPredictionLoss` class in `custom_prediction_loss.py`. You may modify the `__init__` and `forward` methods, add helper methods, and import additional modules.

## Interface
```python
class CustomPredictionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state, predicted):
        """
        Args:
            state:     [B, C, T, H, W] - target encoded representations from the encoder
            predicted: [B, C, T, H, W] - predicted representations from the predictor

        Returns:
            Scalar loss tensor (lower means predicted is closer to state)
        """
```
The loss is called during JEPA's `unroll()` method as `predcost(state, predicted_states)`, where both tensors share the same shape. The returned scalar is added to the regularization loss and backpropagated.

## Evaluation
Mean detection Average Precision (AP) across prediction timesteps on Moving MNIST. Higher is better. The model is trained for 50 epochs with the Adam optimizer (lr=1e-3) and the final mean detection AP is reported.

The prediction loss is evaluated across three model sizes to test generalization:
- **small**: henc=16, dstc=8, hpre=16
- **base**: henc=32, dstc=16, hpre=32
- **large**: henc=64, dstc=32, hpre=64

## Notes
- The encoder produces spatial feature maps (not just vectors), so spatial structure matters.
- The predictor operates autoregressively over time steps, so temporal weighting/ordering can be exploited.
- The VCLoss regularizer is computed separately and added by the trainer; you only design the prediction term.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/eb_jepa/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `eb_jepa/custom_prediction_loss.py`
- editable lines **36–54**


Other files you may **read** for context (do not modify):
- `eb_jepa/losses.py`
- `eb_jepa/jepa.py`


## Readable Context


### `eb_jepa/custom_prediction_loss.py`  [EDITABLE — lines 36–54 only]

```python
     1: """Self-contained Video JEPA training script with custom prediction loss.
     2: 
     3: Trains a JEPA model on Moving MNIST and evaluates detection Average Precision.
     4: The CustomPredictionLoss class is the editable component that the agent modifies.
     5: """
     6: import os
     7: import sys; sys.path = [p for p in sys.path if not os.path.isfile(os.path.join(p, 'logging.py'))]
     8: import collections
     9: import random
    10: 
    11: import numpy as np
    12: import torch
    13: import torch.nn as nn
    14: import torch.nn.functional as F
    15: from torch.optim import Adam
    16: from torch.utils.data import DataLoader
    17: from tqdm import tqdm
    18: 
    19: from eb_jepa.architectures import (
    20:     DetHead,
    21:     Projector,
    22:     ResNet5,
    23:     ResUNet,
    24:     StateOnlyPredictor,
    25: )
    26: from eb_jepa.datasets.moving_mnist import MovingMNISTDet
    27: from eb_jepa.image_decoder import ImageDecoder
    28: from eb_jepa.jepa import JEPA, JEPAProbe
    29: from eb_jepa.losses import VCLoss
    30: 
    31: # ==============================================================================
    32: # EDITABLE REGION START
    33: # ==============================================================================
    34: 
    35: 
    36: class CustomPredictionLoss(nn.Module):
    37:     """Prediction cost function for temporal JEPA.
    38: 
    39:     Measures discrepancy between predicted and target representations
    40:     in the latent space. Used to train the predictor network.
    41: 
    42:     Args:
    43:         state: [B, C, T, H, W] target encoded representations
    44:         predicted: [B, C, T, H, W] predicted representations
    45: 
    46:     Returns:
    47:         Scalar loss tensor
    48:     """
    49: 
    50:     def __init__(self):
    51:         super().__init__()
    52: 
    53:     def forward(self, state, predicted):
    54:         return torch.tensor(0.0, device=state.device, requires_grad=True)
    55: 
    56: 
    57: # ==============================================================================
    58: # EDITABLE REGION END
    59: # ==============================================================================
    60: # ============================================================================
    61: # FIXED REGION (do not modify below this line)
    62: # ============================================================================
    63: 
    64: 
    65: def seed_everything(seed):
    66:     os.environ["PYTHONHASHSEED"] = str(seed)
    67:     random.seed(seed)
    68:     np.random.seed(seed)
    69:     torch.manual_seed(seed)
    70:     if torch.cuda.is_available():
    71:         torch.cuda.manual_seed(seed)
    72:         torch.cuda.manual_seed_all(seed)
    73:     torch.backends.cudnn.benchmark = False
    74: 
    75: 
    76: def seed_worker(worker_id):
    77:     worker_seed = torch.initial_seed() % 2**32
    78:     random.seed(worker_seed)
    79:     np.random.seed(worker_seed)
    80: 
    81: 
    82: def make_generator(seed):
    83:     generator = torch.Generator()
    84:     generator.manual_seed(seed)
    85:     return generator
    86: 
    87: 
    88: def validation_loop(val_loader, jepa, detection_head, pixel_decoder, steps, device):
    89:     """Run validation and compute detection AP metrics."""
    90:     jepa.eval()
    91:     detection_head.eval()
    92:     pixel_decoder.eval()
    93: 
    94:     metrics = collections.defaultdict(list)
    95:     for batch in tqdm(val_loader, desc="Validation"):
    96:         batch = {k: v.to(device) for k, v in batch.items()}
    97:         x = batch["video"]
    98:         loc_map = batch["digit_location"]
    99: 
   100:         recon_loss = pixel_decoder(x, x)
   101:         det_loss = detection_head(x, loc_map)
   102: 
   103:         logs = {
   104:             "val/recon_loss": float(recon_loss.item()),
   105:             "val/det_loss": float(det_loss.item()),
   106:         }
   107:         for k, v in logs.items():
   108:             metrics[k].append(v)
   109: 
   110:         T = x.shape[2]
   111:         preds, _ = jepa.unroll(
   112:             x,
   113:             actions=None,
   114:             nsteps=T - 2,
   115:             unroll_mode="parallel",
   116:             compute_loss=False,
   117:             return_all_steps=True,
   118:         )
   119:         scores = detection_head.head.score(preds, loc_map[:, 2:])
   120:         for s, score in enumerate(scores):
   121:             metrics[f"AP_{s}"].append(float(score))
   122: 
   123:     # Aggregate results
   124:     metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
   125: 
   126:     jepa.train()
   127:     detection_head.train()
   128:     pixel_decoder.train()
   129: 
   130:     return metrics
   131: 
   132: 
   133: def main():
   134:     """Train Video JEPA with custom prediction loss on Moving MNIST."""
   135:     # Setup
   136:     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   137:     print(f"Using device: {device}", flush=True)
   138: 
   139:     seed = int(os.environ.get("SEED", "42"))
   140:     seed_everything(seed)
   141: 
   142:     # Hyperparameters
   143:     epochs = 50
   144:     lr = 1e-3
   145:     steps = 4
   146:     dobs = 1
   147: 
   148:     # Model size from environment (small / base / large)
   149:     model_size = os.environ.get("MODEL_SIZE", "base")
   150:     _MODEL_CONFIGS = {
   151:         "small": {"henc": 16, "dstc": 8,  "hpre": 16, "batch_size": 64},
   152:         "base":  {"henc": 32, "dstc": 16, "hpre": 32, "batch_size": 32},
   153:         "large": {"henc": 64, "dstc": 32, "hpre": 64, "batch_size": 16},
   154:     }
   155:     cfg = _MODEL_CONFIGS[model_size]
   156:     henc, dstc, hpre = cfg["henc"], cfg["dstc"], cfg["hpre"]
   157:     batch_size = cfg["batch_size"]
   158:     print(f"Model size: {model_size} (henc={henc}, dstc={dstc}, hpre={hpre}, bs={batch_size})", flush=True)
   159: 
   160:     # Load datasets
   161:     print("Loading Moving MNIST dataset...", flush=True)
   162:     train_set = MovingMNISTDet(split="train")
   163:     val_set = MovingMNISTDet(split="val")
   164:     train_loader = DataLoader(
   165:         train_set, batch_size=batch_size, shuffle=True, num_workers=2,
   166:         worker_init_fn=seed_worker, generator=make_generator(seed)
   167:     )
   168:     val_loader = DataLoader(
   169:         val_set, batch_size=batch_size, shuffle=False, num_workers=2,
   170:         worker_init_fn=seed_worker, generator=make_generator(seed + 1)
   171:     )
   172:     print(
   173:         f"Dataset loaded: {len(train_set)} train, {len(val_set)} val samples",
   174:         flush=True,
   175:     )
   176: 
   177:     # Initialize model components
   178:     print("Initializing model...", flush=True)
   179:     encoder = ResNet5(dobs, henc, dstc)
   180:     predictor_model = ResUNet(2 * dstc, hpre, dstc)
   181:     predictor = StateOnlyPredictor(predictor_model, context_length=2)
   182:     projector = Projector(f"{dstc}-{dstc * 4}-{dstc * 4}")
   183:     regularizer = VCLoss(std_coeff=10, cov_coeff=100, proj=projector)
   184: 
   185:     # Use CustomPredictionLoss instead of SquareLossSeq
   186:     ploss = CustomPredictionLoss()
   187:     jepa = JEPA(encoder, encoder, predictor, regularizer, ploss).to(device)
   188: 
   189:     # Initialize decoder and detection head (for evaluation only)
   190:     decoder = ImageDecoder(dstc, dobs, hidden_dim=dstc)
   191:     dethead = DetHead(dstc, hpre, dobs)
   192:     pixel_decoder = JEPAProbe(jepa, decoder, nn.MSELoss()).to(device)
   193:     detection_head = JEPAProbe(jepa, dethead, nn.BCELoss()).to(device)
   194: 
   195:     jepa.train()
   196:     detection_head.train()
   197:     pixel_decoder.train()
   198: 
   199:     optimizer = Adam(
   200:         [
   201:             {"params": jepa.parameters(), "lr": lr},
   202:             {"params": pixel_decoder.head.parameters(), "lr": lr / 10},
   203:             {"params": detection_head.head.parameters(), "lr": lr},
   204:         ]
   205:     )
   206: 
   207:     # Training loop
   208:     print(f"Starting training for {epochs} epochs...", flush=True)
   209: 
   210:     for epoch in range(epochs):
   211:         pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
   212: 
   213:         for batch in pbar:
   214:             batch = {k: v.to(device) for k, v in batch.items()}
   215:             x = batch["video"]
   216:             loc_map = batch["digit_location"]
   217: 
   218:             optimizer.zero_grad()
   219:             _, (jepa_loss, regl, _, regldict, pl) = jepa.unroll(
   220:                 x,
   221:                 actions=None,
   222:                 nsteps=steps,
   223:                 unroll_mode="parallel",
   224:                 compute_loss=True,
   225:                 return_all_steps=False,
   226:             )
   227:             recon_loss = pixel_decoder(x, x)
   228:             det_loss = detection_head(x, loc_map)
   229:             total_loss = jepa_loss + recon_loss + det_loss
   230: 
   231:             total_loss.backward()
   232:             optimizer.step()
   233: 
   234:             pbar.set_postfix(
   235:                 {
   236:                     "loss": f"{jepa_loss.item():.4f}",
   237:                     "vc": f"{regl.item():.4f}",
   238:                     "pred": f"{pl.item():.4f}",
   239:                 }
   240:             )
   241: 
   242:         # Print training metrics every epoch
   243:         print(
   244:             f"TRAIN_METRICS epoch={epoch} "
   245:             f"loss={jepa_loss.item():.6f} "
   246:             f"vc_loss={regl.item():.6f} "
   247:             f"pred_loss={pl.item():.6f}",
   248:             flush=True,
   249:         )
   250: 
   251:         # Validation every 5 epochs and at last epoch
   252:         if epoch % 5 == 0 or epoch == epochs - 1:
   253:             val_metrics = validation_loop(
   254:                 val_loader, jepa, detection_head, pixel_decoder, steps, device
   255:             )
   256: 
   257:             # Compute mean detection AP across timesteps
   258:             ap_keys = [k for k in val_metrics if k.startswith("AP_")]
   259:             if ap_keys:
   260:                 mean_ap = np.mean([val_metrics[k] for k in ap_keys])
   261:             else:
   262:                 mean_ap = 0.0
   263: 
   264:             print(
   265:                 f"Validation epoch={epoch}: "
   266:                 f"recon_loss={val_metrics.get('val/recon_loss', 0):.4f} "
   267:                 f"det_loss={val_metrics.get('val/det_loss', 0):.4f} "
   268:                 f"mean_detection_ap={mean_ap:.4f}",
   269:                 flush=True,
   270:             )
   271: 
   272:     # Final evaluation
   273:     print("\nRunning final evaluation...", flush=True)
   274:     val_metrics = validation_loop(
   275:         val_loader, jepa, detection_head, pixel_decoder, steps, device
   276:     )
   277: 
   278:     ap_keys = [k for k in val_metrics if k.startswith("AP_")]
   279:     if ap_keys:
   280:         mean_ap = np.mean([val_metrics[k] for k in ap_keys])
   281:     else:
   282:         mean_ap = 0.0
   283: 
   284:     print(f"TEST_METRICS: mean_detection_ap={mean_ap:.4f}", flush=True)
   285:     print("Training complete!", flush=True)
   286: 
   287: 
   288: if __name__ == "__main__":
   289:     main()
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **small** — wall-clock budget `3:30:00`, compute share `1.0`
- **base** — wall-clock budget `4:00:00`, compute share `1.0`
- **large** — wall-clock budget `8:00:00`, compute share `1.0`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `mse` baseline — editable region  [READ-ONLY — reference implementation]

In `eb_jepa/custom_prediction_loss.py`:

```python
Lines 36–43:
    33: # ==============================================================================
    34: 
    35: 
    36: class CustomPredictionLoss(nn.Module):
    37:     """MSE prediction loss for temporal JEPA."""
    38: 
    39:     def __init__(self):
    40:         super().__init__()
    41: 
    42:     def forward(self, state, predicted):
    43:         return F.mse_loss(state, predicted)
    44: 
    45: 
    46: # ==============================================================================
```

### `smooth_l1` baseline — editable region  [READ-ONLY — reference implementation]

In `eb_jepa/custom_prediction_loss.py`:

```python
Lines 36–43:
    33: # ==============================================================================
    34: 
    35: 
    36: class CustomPredictionLoss(nn.Module):
    37:     """Smooth L1 prediction loss for temporal JEPA."""
    38: 
    39:     def __init__(self):
    40:         super().__init__()
    41: 
    42:     def forward(self, state, predicted):
    43:         return F.smooth_l1_loss(state, predicted)
    44: 
    45: 
    46: # ==============================================================================
```

### `cosine` baseline — editable region  [READ-ONLY — reference implementation]

In `eb_jepa/custom_prediction_loss.py`:

```python
Lines 36–45:
    33: # ==============================================================================
    34: 
    35: 
    36: class CustomPredictionLoss(nn.Module):
    37:     """Cosine similarity prediction loss for temporal JEPA."""
    38: 
    39:     def __init__(self):
    40:         super().__init__()
    41: 
    42:     def forward(self, state, predicted):
    43:         s = F.normalize(state, dim=1)
    44:         p = F.normalize(predicted, dim=1)
    45:         return (1 - (s * p).sum(dim=1)).mean()
    46: 
    47: 
    48: # ==============================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
