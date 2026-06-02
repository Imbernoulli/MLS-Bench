# MLS-Bench: meta-inner-loop-optimizer

# Meta-Learning: Inner-Loop Optimization Algorithm Design

## Research Question
Design a novel inner-loop adaptation algorithm for gradient-based meta-learning. The contribution is the *adaptation rule itself* (which parameters change, how gradients are scaled or transformed, what state is carried across inner steps), not changes to the data loader, backbone, or outer-loop schedule.

## Background
Gradient-based meta-learning (MAML-style) learns a model initialization that can be quickly adapted to new tasks via a few gradient steps. The **inner loop** is the adaptation step on the support set; the **outer loop** optimizes the initialization (and any optimizer state) across tasks.

Reference baselines (provided as read-only modules in `learn2learn/algorithms/`):
- **MAML** — Finn, Abbeel, Levine, ICML 2017 ([arXiv:1703.03400](https://arxiv.org/abs/1703.03400)). Inner loop = differentiable SGD with a fixed scalar learning rate; outer loop optimizes only the initialization.
- **Meta-SGD** — Li, Zhou, Chen, Li, 2017 ([arXiv:1707.09835](https://arxiv.org/abs/1707.09835)). Per-parameter learnable inner-loop learning rates (one rate vector per parameter tensor), meta-trained jointly with the initialization.
- **ANIL (Almost No Inner Loop)** — Raghu, Raghu, Bengio, Vinyals, ICLR 2020 ([arXiv:1909.09157](https://arxiv.org/abs/1909.09157)). Adapts only the classification head in the inner loop; the feature extractor is frozen during adaptation, exploiting feature reuse.

Design axes worth considering:
- **What to adapt**: all parameters, head only, learned subset/mask.
- **How to scale gradients**: fixed LR, per-parameter LR, preconditioning matrix, learned transform.
- **Memory across inner steps**: momentum-like state, recurrent updates, second-order info.
- **Regularization**: trust-region constraints, support-set overfitting penalties.

## Implementation Contract
Modify `InnerLoopOptimizer` in `learn2learn/custom_maml.py`:

```python
class InnerLoopOptimizer:
    def __init__(self, model: nn.Module, inner_lr: float):
        # model: base model (for parameter shape inspection)
        # inner_lr: default learning rate
        # Create any learnable parameters here.
        ...

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        # model is a CLONE — safe to modify in-place.
        # MUST use differentiable ops (torch.autograd.grad), NOT torch.optim.
        # Return the adapted model.
        ...

    def meta_parameters(self) -> List[Tensor]:
        # Learnable optimizer parameters for the outer loop.
        # Return [] if optimizer has no learnable state (vanilla MAML).
        ...
```

Available reference code: `learn2learn/algorithms/maml.py`, `meta_sgd.py`, `gbml.py`.

## Fixed Pipeline
- Backbone: CNN4.
- Meta-training: 60,000 iterations, 4 tasks per meta-batch.
- Inner loop: 5 steps during training, 10 steps during evaluation.
- Evaluation metric, benchmark settings, and episode counts are fixed.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/learn2learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `learn2learn/custom_maml.py`
- editable lines **177–254**


Other files you may **read** for context (do not modify):
- `learn2learn/learn2learn/algorithms/gbml.py`


## Readable Context


### `learn2learn/custom_maml.py`  [EDITABLE — lines 177–254 only]

```python
     1: # Custom inner-loop optimizer for gradient-based meta-learning
     2: #
     3: # EDITABLE section: InnerLoopOptimizer class and helper modules.
     4: # FIXED sections: everything else (config, data loading, backbone, outer loop, evaluation).
     5: #
     6: # Research question: Design the inner-loop adaptation algorithm that determines
     7: # HOW model parameters are updated during fast adaptation to a new task.
     8: import os
     9: import sys
    10: import copy
    11: import random
    12: from statistics import mean
    13: from typing import Optional, Tuple, Dict, List
    14: 
    15: # Fix import path: exclude the learn2learn source tree so that
    16: # ``import learn2learn`` resolves to the pip-installed package,
    17: # not the source checkout at /workspace/learn2learn/learn2learn/.
    18: _script_dir = os.path.dirname(os.path.abspath(__file__))
    19: sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(_script_dir)]
    20: 
    21: import numpy as np
    22: import torch
    23: import torch.nn as nn
    24: import torch.nn.functional as F
    25: from torch import Tensor
    26: 
    27: import learn2learn as l2l
    28: from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
    29: from torchvision import transforms as tv_transforms
    30: 
    31: 
    32: # =====================================================================
    33: # FIXED: Configuration
    34: # =====================================================================
    35: SEED = int(os.environ.get("SEED", "42"))
    36: OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
    37: SETTING = os.environ.get("ENV", "mini_imagenet_5shot")
    38: 
    39: # Parse setting: dataset_Nshot
    40: _parts = SETTING.rsplit("_", 1)
    41: DATASET_NAME = _parts[0]            # e.g. "mini_imagenet" or "cifar_fs"
    42: N_SHOT = int(_parts[1].replace("shot", ""))  # e.g. 1 or 5
    43: 
    44: # Few-shot settings
    45: N_WAY = 5
    46: N_QUERY = 15
    47: IMAGE_SIZE = 84
    48: HIDDEN_SIZE = 64  # CNN4 channel width
    49: 
    50: # Training settings — 1-shot converges slower, use 4x iterations.
    51: N_META_ITERS = 60000 if N_SHOT == 1 else 15000
    52: META_BATCH_SIZE = 4        # tasks per meta-update
    53: INNER_STEPS_TRAIN = 5      # adaptation steps during training
    54: INNER_STEPS_TEST = 10      # adaptation steps during evaluation (more steps at test)
    55: META_LR = 0.003            # outer-loop learning rate
    56: INNER_LR = 0.5             # default inner-loop learning rate
    57: 
    58: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    59: 
    60: EVAL_INTERVAL = 500        # meta-iterations between evaluations
    61: N_EVAL_TASKS = 200         # tasks for validation
    62: N_TEST_TASKS = 600         # tasks for final test
    63: 
    64: 
    65: # =====================================================================
    66: # FIXED: Dataset loading via learn2learn benchmarks
    67: # =====================================================================
    68: def get_tasksets(dataset_name: str, n_way: int, n_shot: int, n_query: int,
    69:                  root: str = os.environ.get("L2L_DATA_ROOT", "/workspace/l2l_data")):
    70:     """Create train/val/test TaskDataset objects using learn2learn."""
    71:     total_samples = n_shot + n_query
    72: 
    73:     if dataset_name == "mini_imagenet":
    74:         dataset_cls = l2l.vision.datasets.MiniImagenet
    75:     elif dataset_name == "cifar_fs":
    76:         dataset_cls = l2l.vision.datasets.CIFARFS
    77:     else:
    78:         raise ValueError(f"Unknown dataset: {dataset_name}")
    79: 
    80:     # CIFAR-FS returns PIL Images (32x32) — needs ToTensor + Resize to 84x84.
    81:     # MiniImagenet already returns [3,84,84] tensors — no transform needed.
    82:     if dataset_name == "cifar_fs":
    83:         img_transform = tv_transforms.Compose([
    84:             tv_transforms.Resize((84, 84)),
    85:             tv_transforms.ToTensor(),
    86:         ])
    87:     else:
    88:         img_transform = None
    89: 
    90:     splits = {}
    91:     for mode in ["train", "validation", "test"]:
    92:         ds = dataset_cls(root=root, mode=mode, download=False,
    93:                          transform=img_transform)
    94:         meta_ds = l2l.data.MetaDataset(ds)
    95:         transforms = [
    96:             NWays(meta_ds, n_way),
    97:             KShots(meta_ds, total_samples),
    98:             LoadData(meta_ds),
    99:             RemapLabels(meta_ds),
   100:             ConsecutiveLabels(meta_ds),
   101:         ]
   102:         splits[mode] = l2l.data.TaskDataset(meta_ds, transforms, num_tasks=-1)
   103: 
   104:     return splits["train"], splits["validation"], splits["test"]
   105: 
   106: 
   107: def split_support_query(data: Tensor, labels: Tensor, n_way: int, n_shot: int):
   108:     """Split a task batch into support and query sets.
   109: 
   110:     Args:
   111:         data: images [n_way * (n_shot + n_query), C, H, W]
   112:         labels: labels [n_way * (n_shot + n_query)]
   113:         n_way: number of classes
   114:         n_shot: number of support examples per class
   115: 
   116:     Returns:
   117:         support_x, support_y, query_x, query_y
   118:     """
   119:     sort_idx = torch.sort(labels).indices
   120:     data = data[sort_idx]
   121:     labels = labels[sort_idx]
   122: 
   123:     n_query_per_class = len(labels) // n_way - n_shot
   124:     support_idx = []
   125:     query_idx = []
   126:     for cls in range(n_way):
   127:         start = cls * (n_shot + n_query_per_class)
   128:         support_idx.extend(range(start, start + n_shot))
   129:         query_idx.extend(range(start + n_shot, start + n_shot + n_query_per_class))
   130: 
   131:     return data[support_idx], labels[support_idx], data[query_idx], labels[query_idx]
   132: 
   133: 
   134: # =====================================================================
   135: # FIXED: CNN4 Backbone (shared by all methods)
   136: # =====================================================================
   137: def make_model(n_way: int, hidden_size: int = HIDDEN_SIZE) -> nn.Module:
   138:     """Create a CNN4 model for few-shot classification.
   139: 
   140:     Returns a CNN4 model with:
   141:     - 4 convolutional blocks (each: Conv2d -> BN -> ReLU -> MaxPool)
   142:     - hidden_size channels per block (default 64)
   143:     - A linear classifier head mapping features to n_way classes
   144:     - Input: [B, 3, 84, 84], Output: [B, n_way]
   145: 
   146:     The feature dimension before the head is hidden_size * 5 * 5 = 1600.
   147:     """
   148:     return l2l.vision.models.CNN4(
   149:         output_size=n_way,
   150:         hidden_size=hidden_size,
   151:         embedding_size=hidden_size * 5 * 5,
   152:     )
   153: 
   154: 
   155: FEATURE_DIM = HIDDEN_SIZE * 5 * 5  # 1600 for CNN4 with hidden_size=64
   156: 
   157: 
   158: # =====================================================================
   159: # FIXED: Utility functions
   160: # =====================================================================
   161: def accuracy(predictions: Tensor, targets: Tensor) -> float:
   162:     """Compute classification accuracy."""
   163:     return (predictions.argmax(dim=1) == targets).float().mean().item()
   164: 
   165: 
   166: def compute_loss_and_acc(model: nn.Module, data: Tensor, labels: Tensor):
   167:     """Compute cross-entropy loss and accuracy for a batch."""
   168:     logits = model(data)
   169:     loss = F.cross_entropy(logits, labels)
   170:     acc = accuracy(logits, labels)
   171:     return loss, acc
   172: 
   173: 
   174: # =====================================================================
   175: # EDITABLE: Inner-Loop Optimizer for Gradient-Based Meta-Learning
   176: # =====================================================================
   177: class InnerLoopOptimizer:
   178:     """Inner-loop adaptation algorithm for gradient-based meta-learning.
   179: 
   180:     This class defines HOW model parameters are updated during fast adaptation
   181:     to a new task. The outer loop (meta-optimizer) is fixed; only this inner
   182:     loop is editable.
   183: 
   184:     The default implementation is vanilla MAML: simple SGD with a fixed
   185:     learning rate applied to all parameters.
   186: 
   187:     You may redesign:
   188:     - Per-parameter or per-layer learning rates (Meta-SGD)
   189:     - Which parameters to adapt (full model vs. head-only, ANIL)
   190:     - Preconditioning / curvature information (Meta-Curvature)
   191:     - Momentum, second-order corrections, or learned update rules
   192:     - Any combination of the above
   193: 
   194:     Interface contract:
   195:     - __init__(model, inner_lr): initialize the optimizer, may create
   196:       learnable parameters that will be meta-learned by the outer loop
   197:     - adapt(model, support_x, support_y, n_steps): perform n_steps of
   198:       inner-loop adaptation on the support set, return adapted model
   199:     - meta_parameters(): return any learnable parameters of the optimizer
   200:       itself (e.g., per-parameter learning rates) for outer-loop optimization
   201: 
   202:     IMPORTANT:
   203:     - The model passed to adapt() is a clone (via l2l.clone_module).
   204:       You must use differentiable operations so gradients flow to the outer loop.
   205:     - Use l2l.algorithms.maml.maml_update(model, lr, grads) or manual
   206:       parameter updates. Do NOT use torch.optim optimizers (they break the
   207:       computational graph).
   208:     - meta_parameters() must return all learnable optimizer state so the
   209:       outer loop can optimize them.
   210:     """
   211: 
   212:     def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
   213:         """Initialize the inner-loop optimizer.
   214: 
   215:         Args:
   216:             model: the base model (used to inspect parameter shapes/counts).
   217:                    Do NOT store a reference to this model — a fresh clone
   218:                    is passed to adapt() each time.
   219:             inner_lr: default inner-loop learning rate.
   220:         """
   221:         self.inner_lr = inner_lr
   222: 
   223:     def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
   224:               n_steps: int) -> nn.Module:
   225:         """Perform inner-loop adaptation.
   226: 
   227:         Args:
   228:             model: a CLONED model (via l2l.clone_module) — safe to modify in-place.
   229:             support_x: support images [n_way * n_shot, C, H, W]
   230:             support_y: support labels [n_way * n_shot]
   231:             n_steps: number of inner-loop gradient steps
   232: 
   233:         Returns:
   234:             The adapted model (may be the same object, modified in-place).
   235:         """
   236:         model.train()
   237:         for _ in range(n_steps):
   238:             loss = F.cross_entropy(model(support_x), support_y)
   239:             grads = torch.autograd.grad(
   240:                 loss, model.parameters(), create_graph=True
   241:             )
   242:             # Vanilla SGD update using learn2learn's differentiable update
   243:             model = l2l.algorithms.maml.maml_update(
   244:                 model, lr=self.inner_lr, grads=grads
   245:             )
   246:         return model
   247: 
   248:     def meta_parameters(self) -> List[Tensor]:
   249:         """Return learnable parameters of the optimizer for outer-loop training.
   250: 
   251:         For vanilla MAML, the inner LR is fixed, so this returns [].
   252:         For Meta-SGD, this would return the per-parameter learning rates.
   253:         """
   254:         return []
   255: 
   256: 
   257: # =====================================================================
   258: # FIXED: Meta-Training and Evaluation Loop
   259: # =====================================================================
   260: def meta_train_step(model, inner_opt, meta_optimizer,
   261:                     taskset, n_way, n_shot, n_query, meta_batch_size,
   262:                     inner_steps, device):
   263:     """One meta-training iteration: sample tasks, adapt, compute meta-loss."""
   264:     meta_train_loss = 0.0
   265:     meta_train_acc = 0.0
   266: 
   267:     for _ in range(meta_batch_size):
   268:         # Clone model for this task
   269:         learner = l2l.clone_module(model)
   270: 
   271:         # Sample a task
   272:         task_data = taskset.sample()
   273:         data, labels = task_data
   274:         data, labels = data.to(device), labels.to(device)
   275: 
   276:         # Split into support / query
   277:         support_x, support_y, query_x, query_y = split_support_query(
   278:             data, labels, n_way, n_shot
   279:         )
   280: 
   281:         # Inner-loop adaptation (uses the shared inner_opt instance)
   282:         learner = inner_opt.adapt(learner, support_x, support_y, inner_steps)
   283: 
   284:         # Evaluate on query set (for meta-gradient)
   285:         loss, acc = compute_loss_and_acc(learner, query_x, query_y)
   286:         meta_train_loss += loss
   287:         meta_train_acc += acc
   288: 
   289:     meta_train_loss /= meta_batch_size
   290:     meta_train_acc /= meta_batch_size
   291: 
   292:     # Meta-update
   293:     meta_optimizer.zero_grad()
   294:     meta_train_loss.backward()
   295:     meta_optimizer.step()
   296: 
   297:     return meta_train_loss.item(), meta_train_acc
   298: 
   299: 
   300: def meta_evaluate(model, inner_opt, taskset,
   301:                   n_way, n_shot, n_query, n_tasks, inner_steps, device):
   302:     """Evaluate on a set of tasks."""
   303:     accs = []
   304:     for _ in range(n_tasks):
   305:         learner = l2l.clone_module(model)
   306: 
   307:         task_data = taskset.sample()
   308:         data, labels = task_data
   309:         data, labels = data.to(device), labels.to(device)
   310: 
   311:         support_x, support_y, query_x, query_y = split_support_query(
   312:             data, labels, n_way, n_shot
   313:         )
   314: 
   315:         learner = inner_opt.adapt(learner, support_x, support_y, inner_steps)
   316: 
   317:         with torch.no_grad():
   318:             _, acc = compute_loss_and_acc(learner, query_x, query_y)
   319:         accs.append(acc)
   320: 
   321:     mean_acc = np.mean(accs)
   322:     ci95 = 1.96 * np.std(accs) / np.sqrt(len(accs))
   323:     return mean_acc, ci95
   324: 
   325: 
   326: # =====================================================================
   327: # FIXED: Main Script
   328: # =====================================================================
   329: if __name__ == "__main__":
   330:     # Reproducibility
   331:     random.seed(SEED)
   332:     np.random.seed(SEED)
   333:     torch.manual_seed(SEED)
   334:     torch.cuda.manual_seed_all(SEED)
   335:     torch.backends.cudnn.deterministic = True
   336:     torch.backends.cudnn.benchmark = False
   337: 
   338:     os.makedirs(OUTPUT_DIR, exist_ok=True)
   339: 
   340:     print(f"Dataset: {DATASET_NAME}, N-way: {N_WAY}, N-shot: {N_SHOT}, Seed: {SEED}", flush=True)
   341:     print(f"Setting: {SETTING}", flush=True)
   342:     print(f"Meta-LR: {META_LR}, Inner-LR: {INNER_LR}", flush=True)
   343:     print(f"Inner steps train/test: {INNER_STEPS_TRAIN}/{INNER_STEPS_TEST}", flush=True)
   344: 
   345:     # Load tasksets
   346:     train_tasks, val_tasks, test_tasks = get_tasksets(
   347:         DATASET_NAME, N_WAY, N_SHOT, N_QUERY
   348:     )
   349: 
   350:     # Build model
   351:     model = make_model(N_WAY).to(DEVICE)
   352: 
   353:     # ── FIXED: Parameter count check ────────────────────────────────
   354:     # Budget: CNN4 model (~112K) + inner-loop optimizer learnable params
   355:     # Meta-SGD adds one scalar per parameter (~112K extra).
   356:     # Budget is 1.2x of (model params + Meta-SGD optimizer params).
   357:     _model_params = sum(p.numel() for p in model.parameters())
   358:     _optimizer_budget = _model_params  # Meta-SGD needs one LR per param
   359:     _budget = int((_model_params + _optimizer_budget) * 1.2)
   360: 
   361:     # Create inner-loop optimizer (persistent across all iterations)
   362:     inner_opt = InnerLoopOptimizer(model, INNER_LR)
   363:     _opt_params = sum(p.numel() for p in inner_opt.meta_parameters())
   364:     _total_params = _model_params + _opt_params
   365:     print(f"Model params: {_model_params:,}, Optimizer params: {_opt_params:,}, "
   366:           f"Total: {_total_params:,} (budget: {_budget:,})", flush=True)
   367:     # ────────────────────────────────────────────────────────────────
   368: 
   369:     # Collect all meta-learnable parameters: model params + optimizer params
   370:     all_meta_params = list(model.parameters()) + list(inner_opt.meta_parameters())
   371:     meta_optimizer = torch.optim.Adam(all_meta_params, lr=META_LR)
   372: 
   373:     # Meta-training loop
   374:     best_val_acc = 0.0
   375:     best_state = copy.deepcopy(model.state_dict())
   376:     best_inner_meta_state = [
   377:         p.detach().clone() for p in inner_opt.meta_parameters()
   378:     ]
   379: 
   380:     for iteration in range(1, N_META_ITERS + 1):
   381:         model.train()
   382:         train_loss, train_acc = meta_train_step(
   383:             model, inner_opt, meta_optimizer,
   384:             train_tasks, N_WAY, N_SHOT, N_QUERY, META_BATCH_SIZE,
   385:             INNER_STEPS_TRAIN, DEVICE,
   386:         )
   387: 
   388:         if iteration % EVAL_INTERVAL == 0:
   389:             model.eval()
   390:             val_acc, val_ci = meta_evaluate(
   391:                 model, inner_opt, val_tasks,
   392:                 N_WAY, N_SHOT, N_QUERY, N_EVAL_TASKS,
   393:                 INNER_STEPS_TEST, DEVICE,
   394:             )
   395:             print(
   396:                 f"TRAIN_METRICS iter={iteration} "
   397:                 f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
   398:                 f"val_acc={val_acc:.4f} val_ci95={val_ci:.4f}",
   399:                 flush=True,
   400:             )
   401:             if val_acc > best_val_acc:
   402:                 best_val_acc = val_acc
   403:                 best_state = copy.deepcopy(model.state_dict())
   404:                 best_inner_meta_state = [
   405:                     p.detach().clone() for p in inner_opt.meta_parameters()
   406:                 ]
   407:                 print(f"  New best val accuracy: {val_acc:.4f} +/- {val_ci:.4f}", flush=True)
   408: 
   409:     # Load best model and evaluate on test set
   410:     model.load_state_dict(best_state)
   411:     with torch.no_grad():
   412:         for p, saved in zip(inner_opt.meta_parameters(), best_inner_meta_state):
   413:             p.copy_(saved.to(device=p.device, dtype=p.dtype))
   414:     model.eval()
   415:     TEST_RNG_SEED = 0xBEEF
   416:     random.seed(TEST_RNG_SEED)
   417:     np.random.seed(TEST_RNG_SEED)
   418:     torch.manual_seed(TEST_RNG_SEED)
   419:     torch.cuda.manual_seed_all(TEST_RNG_SEED)
   420:     test_acc, test_ci = meta_evaluate(
   421:         model, inner_opt, test_tasks,
   422:         N_WAY, N_SHOT, N_QUERY, N_TEST_TASKS,
   423:         INNER_STEPS_TEST, DEVICE,
   424:     )
   425:     print(f"TEST_METRICS accuracy={test_acc:.4f} ci95={test_ci:.4f}", flush=True)
   426:     print(f"Test accuracy: {100 * test_acc:.2f}% +/- {100 * test_ci:.2f}%", flush=True)
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `maml` baseline — editable region  [READ-ONLY — reference implementation]

In `learn2learn/custom_maml.py`:

```python
Lines 177–208:
   174: # =====================================================================
   175: # EDITABLE: Inner-Loop Optimizer for Gradient-Based Meta-Learning
   176: # =====================================================================
   177: class InnerLoopOptimizer:
   178:     """MAML inner-loop optimizer (Finn et al., 2017).
   179: 
   180:     Vanilla SGD with a fixed learning rate applied uniformly to all
   181:     model parameters. This is the standard MAML inner loop.
   182: 
   183:     Shot-aware LR override: the global INNER_LR=0.5 destabilizes
   184:     full-network adaptation at 1-shot in the local harness. At 5-shot the larger
   185:     support set buffers gradient noise so 0.5 is fine (matches
   186:     learn2learn benchmark default). Use the common 1-shot recipe
   187:     (0.01) only when N_SHOT=1, keep 0.5 for 5-shot.
   188:     """
   189: 
   190:     def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
   191:         self.inner_lr = 0.01 if N_SHOT == 1 else 0.5
   192: 
   193:     def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
   194:               n_steps: int) -> nn.Module:
   195:         model.train()
   196:         for _ in range(n_steps):
   197:             loss = F.cross_entropy(model(support_x), support_y)
   198:             grads = torch.autograd.grad(
   199:                 loss, model.parameters(), create_graph=True
   200:             )
   201:             model = l2l.algorithms.maml.maml_update(
   202:                 model, lr=self.inner_lr, grads=grads
   203:             )
   204:         return model
   205: 
   206:     def meta_parameters(self) -> List[Tensor]:
   207:         return []
   208: 
   209: 
   210: 
   211: # =====================================================================
```

### `meta_sgd` baseline — editable region  [READ-ONLY — reference implementation]

In `learn2learn/custom_maml.py`:

```python
Lines 177–207:
   174: # =====================================================================
   175: # EDITABLE: Inner-Loop Optimizer for Gradient-Based Meta-Learning
   176: # =====================================================================
   177: class InnerLoopOptimizer:
   178:     """Meta-SGD inner-loop optimizer (Li et al., 2017).
   179: 
   180:     Learns a per-parameter learning rate vector that is meta-optimized
   181:     by the outer loop. Each model parameter gets a corresponding learnable
   182:     learning rate tensor of the same shape, initialized to inner_lr.
   183:     """
   184: 
   185:     def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
   186:         self.inner_lr = inner_lr
   187:         # Create per-parameter learnable learning rates
   188:         self.lrs = nn.ParameterList([
   189:             nn.Parameter(torch.ones_like(p) * inner_lr)
   190:             for p in model.parameters()
   191:         ])
   192: 
   193:     def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
   194:               n_steps: int) -> nn.Module:
   195:         model.train()
   196:         for _ in range(n_steps):
   197:             loss = F.cross_entropy(model(support_x), support_y)
   198:             grads = torch.autograd.grad(
   199:                 loss, model.parameters(), create_graph=True
   200:             )
   201:             updates = [-lr * g for g, lr in zip(grads, self.lrs)]
   202:             l2l.update_module(model, updates=updates)
   203:         return model
   204: 
   205:     def meta_parameters(self) -> List[Tensor]:
   206:         return list(self.lrs.parameters())
   207: 
   208: 
   209: 
   210: # =====================================================================
```

### `anil` baseline — editable region  [READ-ONLY — reference implementation]

In `learn2learn/custom_maml.py`:

```python
Lines 177–223:
   174: # =====================================================================
   175: # EDITABLE: Inner-Loop Optimizer for Gradient-Based Meta-Learning
   176: # =====================================================================
   177: class InnerLoopOptimizer:
   178:     """ANIL inner-loop optimizer (Raghu et al., 2019).
   179: 
   180:     Almost No Inner Loop: only adapts the final classification head
   181:     during inner-loop adaptation. The feature extractor backbone is
   182:     frozen, relying on feature reuse from the meta-initialization.
   183:     """
   184: 
   185:     def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
   186:         self.inner_lr = inner_lr
   187:         # Identify head parameters (the last linear layer: classifier)
   188:         # CNN4 structure: features (CNN4Backbone) -> classifier (Linear)
   189:         self._head_param_names = set()
   190:         for name, _ in model.named_parameters():
   191:             if "classifier" in name:
   192:                 self._head_param_names.add(name)
   193: 
   194:     def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
   195:               n_steps: int) -> nn.Module:
   196:         model.train()
   197:         for _ in range(n_steps):
   198:             # Re-identify head params each step because l2l.update_module
   199:             # replaces parameter objects (new ids), so stale references
   200:             # from a previous step would cause all updates to be zero.
   201:             head_params = []
   202:             head_ids = set()
   203:             for name, p in model.named_parameters():
   204:                 if name in self._head_param_names:
   205:                     head_params.append(p)
   206:                     head_ids.add(id(p))
   207: 
   208:             loss = F.cross_entropy(model(support_x), support_y)
   209:             grads = torch.autograd.grad(
   210:                 loss, head_params, create_graph=True
   211:             )
   212:             grad_map = {id(p): g for p, g in zip(head_params, grads)}
   213:             updates = [
   214:                 -self.inner_lr * grad_map[id(p)] if id(p) in head_ids
   215:                 else torch.zeros_like(p)
   216:                 for p in model.parameters()
   217:             ]
   218:             l2l.update_module(model, updates=updates)
   219:         return model
   220: 
   221:     def meta_parameters(self) -> List[Tensor]:
   222:         return []
   223: 
   224: 
   225: 
   226: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
