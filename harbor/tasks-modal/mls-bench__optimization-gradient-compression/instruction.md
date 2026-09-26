# MLS-Bench: optimization-gradient-compression

# Gradient Compression for Communication-Efficient Distributed Training

## Research Question
Design a gradient compression operator that reduces communication cost in distributed training while maintaining convergence quality (test accuracy).

## Background
In distributed data-parallel training, gradient communication is often the bottleneck. Workers compute local gradients, which must be aggregated (e.g., via all-reduce) before the optimizer step. Gradient compression reduces the volume of data communicated by applying lossy compression to gradients before transmission.

Three main families of compression exist:
- **Sparsification**: keep only a subset of gradient elements (e.g., TopK selects the largest magnitudes; Stich, Cordonnier, and Jaggi, "Sparsified SGD with Memory", NeurIPS 2018).
- **Quantization**: reduce the precision of gradient values (e.g., QSGD uses stochastic rounding to discrete levels).
- **Low-rank approximation**: approximate gradient matrices with low-rank factors (e.g., PowerSGD).

A key challenge is that naive compression introduces bias or variance that degrades convergence. Error feedback — accumulating compression residuals locally and adding them to the next gradient — is a widely used correction (Karimireddy, Rebjock, Stich, and Jaggi, "Error Feedback Fixes SignSGD and Other Gradient Compression Schemes", ICML 2019; arXiv:1901.09847).

## Task
Modify the `Compressor` class in `custom_compressor.py`. Your compressor must implement:
- `__init__(self, compress_ratio, param_numels, budget_bits)`: `param_numels` maps every parameter name to its number of entries; `budget_bits` is the communication budget of one training step.
- `compress(self, tensor, name)`: encode a gradient tensor into a packet (format below).

The compressor may maintain internal state (e.g., error feedback residuals) across calls. The `name` parameter identifies parameters for per-parameter state tracking. Decoding is fixed: the harness reconstructs each gradient from its packet alone, and only that reconstruction reaches the optimizer.

## Interface
```python
class Compressor:
    def __init__(self, compress_ratio, param_numels, budget_bits): ...
    def compress(self, tensor, name) -> packet: ...  # or a list of packets, whose decodings are summed
```
A packet is one of:
- `{"values": V, "bits": b}`: dense, one value per gradient entry;
- `{"values": V, "bits": b, "indices": I}`: sparse, `I` the distinct flat positions of the values `V`;
- `{"factors": (P, Q)}`: low rank, `P @ Q.T` viewed as `(shape[0], numel / shape[0])`.

Values are transmitted as float32. The harness charges each packet in bits: `len(V) * b`; plus 32 bits per distinct value when `b < 32` (the codebook, which may hold at most `2**b` values); plus the Elias-gamma code of the sorted gaps between sparse positions; plus 32 bits per low-rank factor entry. The helpers `packet_cost(packet, shape)` and `elias_gamma_bound(k, numel)` compute these charges.

**Budget.** One training step may transmit at most `budget_bits = compress_ratio × 32 × (total parameter entries)` bits over all parameters together, i.e. 100x less than the dense float32 gradient at `compress_ratio = 0.01`. A step over the budget, or a malformed packet (non-finite values, repeated or out-of-range indices, more distinct values than `2**b`), stops the run and the run is invalid.

## Baselines (paper-cited reference implementations, each sized to the budget)
- **topk_ef** — Top-K sparsification with error feedback (Stich et al., "Sparsified SGD with Memory", NeurIPS 2018; Karimireddy et al., "Error Feedback Fixes SignSGD and Other Gradient Compression Schemes", ICML 2019; arXiv:1901.09847). Sends the largest-magnitude entries as float32 values at Elias-gamma coded positions (about 0.65% of the entries at this budget).
- **qsgd** — Quantized SGD with stochastic uniform quantization (Alistarh, Grubic, Li, Tomioka, and Vojnovic, "QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding", NeurIPS 2017; arXiv:1610.02132), in the paper's Elias-coded sparse encoding with the largest number of levels that fits the budget.
- **signsgd** — Scaled sign compression with error feedback (Bernstein, Wang, Azizzadenesheli, and Anandkumar, "signSGD: Compressed Optimisation for Non-Convex Problems", ICML 2018; arXiv:1802.04434; Karimireddy et al. 2019). One bit for every entry is only 32x, so it sends the signs of the largest-magnitude entries the budget can carry (about 2.5%), scaled by their mean magnitude.

A reference low-rank method (Vogels, Karimireddy, and Jaggi, "PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization", NeurIPS 2019; arXiv:1905.13727) is a useful design point even though it is not run as a baseline here.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-vision/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `pytorch-vision/custom_compressor.py`
- editable lines **182–224**




## Readable Context


### `pytorch-vision/custom_compressor.py`  [EDITABLE — lines 182–224 only]

```python
     1: """Gradient Compression for Communication-Efficient Distributed Training.
     2: 
     3: Self-contained benchmark: trains standard vision models on CIFAR datasets
     4: using data-parallel SGD with a pluggable gradient compressor.
     5: 
     6: The script simulates distributed training on a single node by:
     7: 1. Computing gradients normally
     8: 2. Encoding each gradient with the editable Compressor into a packet whose
     9:    size fixed code charges in bits against a per-step budget
    10: 3. Decoding the packet in fixed code and using it for the optimizer step
    11: 
    12: This measures the effect of gradient compression on convergence quality at
    13: a fixed communication budget, without requiring multi-node infrastructure.
    14: """
    15: 
    16: import argparse
    17: import math
    18: import os
    19: import time
    20: 
    21: import torch
    22: import torch.nn as nn
    23: import torch.nn.functional as F
    24: import torch.optim as optim
    25: from torch.utils.data import DataLoader
    26: from torchvision import datasets, transforms
    27: 
    28: # ============================================================================
    29: # Model Definitions (FIXED)
    30: # ============================================================================
    31: 
    32: 
    33: def conv3x3(in_planes, out_planes, stride=1):
    34:     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    35:                      padding=1, bias=False)
    36: 
    37: 
    38: class BasicBlock(nn.Module):
    39:     expansion = 1
    40: 
    41:     def __init__(self, in_planes, planes, stride=1):
    42:         super().__init__()
    43:         self.conv1 = conv3x3(in_planes, planes, stride)
    44:         self.bn1 = nn.BatchNorm2d(planes)
    45:         self.conv2 = conv3x3(planes, planes)
    46:         self.bn2 = nn.BatchNorm2d(planes)
    47:         self.shortcut = nn.Sequential()
    48:         if stride != 1 or in_planes != planes * self.expansion:
    49:             self.shortcut = nn.Sequential(
    50:                 nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
    51:                           stride=stride, bias=False),
    52:                 nn.BatchNorm2d(planes * self.expansion),
    53:             )
    54: 
    55:     def forward(self, x):
    56:         out = F.relu(self.bn1(self.conv1(x)))
    57:         out = self.bn2(self.conv2(out))
    58:         out += self.shortcut(x)
    59:         return F.relu(out)
    60: 
    61: 
    62: class ResNet(nn.Module):
    63:     def __init__(self, block, num_blocks, num_classes=10):
    64:         super().__init__()
    65:         self.in_planes = 16
    66:         self.conv1 = conv3x3(3, 16)
    67:         self.bn1 = nn.BatchNorm2d(16)
    68:         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
    69:         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
    70:         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
    71:         self.linear = nn.Linear(64 * block.expansion, num_classes)
    72: 
    73:     def _make_layer(self, block, planes, num_blocks, stride):
    74:         strides = [stride] + [1] * (num_blocks - 1)
    75:         layers = []
    76:         for s in strides:
    77:             layers.append(block(self.in_planes, planes, s))
    78:             self.in_planes = planes * block.expansion
    79:         return nn.Sequential(*layers)
    80: 
    81:     def forward(self, x):
    82:         out = F.relu(self.bn1(self.conv1(x)))
    83:         out = self.layer1(out)
    84:         out = self.layer2(out)
    85:         out = self.layer3(out)
    86:         out = F.adaptive_avg_pool2d(out, 1)
    87:         out = out.view(out.size(0), -1)
    88:         return self.linear(out)
    89: 
    90: 
    91: class VGG(nn.Module):
    92:     """VGG-11 with batch normalization."""
    93: 
    94:     def __init__(self, num_classes=100):
    95:         super().__init__()
    96:         cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    97:         layers = []
    98:         in_channels = 3
    99:         for v in cfg:
   100:             if v == 'M':
   101:                 layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
   102:             else:
   103:                 layers.extend([
   104:                     nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
   105:                     nn.BatchNorm2d(v),
   106:                     nn.ReLU(inplace=True),
   107:                 ])
   108:                 in_channels = v
   109:         self.features = nn.Sequential(*layers)
   110:         self.classifier = nn.Sequential(
   111:             nn.Linear(512, 512),
   112:             nn.ReLU(True),
   113:             nn.Dropout(),
   114:             nn.Linear(512, 512),
   115:             nn.ReLU(True),
   116:             nn.Dropout(),
   117:             nn.Linear(512, num_classes),
   118:         )
   119: 
   120:     def forward(self, x):
   121:         x = self.features(x)
   122:         x = F.adaptive_avg_pool2d(x, 1)
   123:         x = x.view(x.size(0), -1)
   124:         return self.classifier(x)
   125: 
   126: 
   127: def build_model(model_name, num_classes, device):
   128:     if model_name == 'resnet20':
   129:         model = ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)
   130:     elif model_name == 'resnet56':
   131:         model = ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)
   132:     elif model_name == 'vgg11':
   133:         model = VGG(num_classes=num_classes)
   134:     else:
   135:         raise ValueError(f"Unknown model: {model_name}")
   136:     return model.to(device)
   137: 
   138: 
   139: # ============================================================================
   140: # Data Loading (FIXED)
   141: # ============================================================================
   142: 
   143: def get_dataloaders(dataset_name, batch_size, num_workers=2):
   144:     if dataset_name == 'cifar10':
   145:         mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
   146:         num_classes = 10
   147:         Dataset = datasets.CIFAR10
   148:     elif dataset_name == 'cifar100':
   149:         mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
   150:         num_classes = 100
   151:         Dataset = datasets.CIFAR100
   152:     else:
   153:         raise ValueError(f"Unknown dataset: {dataset_name}")
   154: 
   155:     train_transform = transforms.Compose([
   156:         transforms.RandomCrop(32, padding=4),
   157:         transforms.RandomHorizontalFlip(),
   158:         transforms.ToTensor(),
   159:         transforms.Normalize(mean, std),
   160:     ])
   161:     test_transform = transforms.Compose([
   162:         transforms.ToTensor(),
   163:         transforms.Normalize(mean, std),
   164:     ])
   165: 
   166:     _data_root = os.environ.get("DATA_ROOT", "/data")
   167:     train_set = Dataset(_data_root + '/cifar', train=True, download=False,
   168:                         transform=train_transform)
   169:     test_set = Dataset(_data_root + '/cifar', train=False, download=False,
   170:                        transform=test_transform)
   171: 
   172:     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
   173:                               num_workers=num_workers, pin_memory=True)
   174:     test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
   175:                              num_workers=num_workers, pin_memory=True)
   176:     return train_loader, test_loader, num_classes
   177: 
   178: 
   179: # ============================================================================
   180: # EDITABLE SECTION — Gradient Compressor (lines 182-224)
   181: # ============================================================================
   182: 
   183: class Compressor:
   184:     """Gradient compressor. Default: plain Top-K sparsification (no error
   185:     feedback), each tensor's k sized to its share of the step budget.
   186: 
   187:     Interface contract (enforced by the fixed code below this class):
   188:     - __init__(compress_ratio, param_numels, budget_bits): `param_numels`
   189:       maps every parameter name to its number of entries; `budget_bits` is
   190:       the most one training step may transmit for all parameters together
   191:       (compress_ratio x 32 bits x total entries, i.e. 100x at 0.01).
   192:     - compress(tensor, name) -> packet, or a list of packets whose decodings
   193:       are summed. Fixed code decodes the packet; only its contents reach
   194:       the optimizer. A packet is one of
   195:         {"values": V, "bits": b}                dense: V has one entry per
   196:                                                 gradient entry
   197:         {"values": V, "bits": b, "indices": I}  sparse: I = distinct flat
   198:                                                 positions of the entries V
   199:         {"factors": (P, Q)}                     low rank: P @ Q.T viewed as
   200:                                                 (shape[0], numel/shape[0])
   201:       Values are sent as float32. Cost in bits (fixed `packet_cost`):
   202:       len(V) * b, plus 32 per distinct value when b < 32 (the codebook;
   203:       at most 2**b distinct values), plus Elias-gamma coded gaps of I,
   204:       plus 32 per factor entry. A step over budget_bits stops the run.
   205:     - The compressor may keep state across calls (e.g. error feedback).
   206:       `name` identifies the parameter.
   207:     """
   208: 
   209:     def __init__(self, compress_ratio, param_numels, budget_bits):
   210:         self.compress_ratio = compress_ratio
   211:         total = sum(param_numels.values())
   212:         self.k = {}
   213:         for name, numel in param_numels.items():
   214:             share = budget_bits * numel // total
   215:             k = 0
   216:             while (k < numel and
   217:                    32 * (k + 1) + elias_gamma_bound(k + 1, numel) <= share):
   218:                 k += 1
   219:             self.k[name] = k
   220: 
   221:     def compress(self, tensor, name):
   222:         flat = tensor.flatten()
   223:         _, indices = torch.topk(flat.abs(), self.k[name], sorted=False)
   224:         return {"values": flat[indices], "bits": 32, "indices": indices}
   225: 
   226: 
   227: # ============================================================================
   228: # FIXED SECTION — Transmission Format, Bit Accounting and Decoding
   229: # ============================================================================
   230: 
   231: class CompressionContractError(RuntimeError):
   232:     """A packet is malformed, or a step transmits more than its budget."""
   233: 
   234: 
   235: def step_budget_bits(compress_ratio, param_numels):
   236:     """Bits one step may transmit: compress_ratio x the dense float32 payload."""
   237:     return int(compress_ratio * 32 * sum(param_numels.values()))
   238: 
   239: 
   240: def elias_gamma_bound(k, numel):
   241:     """Most Elias-gamma bits any k distinct positions in [0, numel) can cost
   242:     (equal gaps are the worst case)."""
   243:     if k <= 0:
   244:         return 0
   245:     return k + int(math.floor(2 * k * math.log2(numel / k) + 1e-9))
   246: 
   247: 
   248: def _to_device(values, dtype, device):
   249:     """A host list as a device tensor without waiting on the device queue."""
   250:     t = torch.tensor(values, dtype=dtype)
   251:     if device.type == "cuda":
   252:         return t.pin_memory().to(device, non_blocking=True)
   253:     return t.to(device)
   254: 
   255: 
   256: def _parts(packet, shape, device):
   257:     """Check a packet's structure and copy its tensors off the compressor.
   258:     Returns a list of ("dense", v, b) / ("sparse", v, b, idx) /
   259:     ("lowrank", P, Q) with float32 values and int64 indices."""
   260:     numel = shape.numel()
   261:     if isinstance(packet, dict):
   262:         packet = [packet]
   263:     if not (isinstance(packet, (list, tuple)) and packet
   264:             and all(isinstance(p, dict) for p in packet)):
   265:         raise CompressionContractError(
   266:             "compress must return a packet dict or a non-empty list of them")
   267:     out = []
   268:     for p in packet:
   269:         if "factors" in p:
   270:             if set(p) != {"factors"} or len(p["factors"]) != 2:
   271:                 raise CompressionContractError(
   272:                     "a low-rank packet is {'factors': (P, Q)}")
   273:             P, Q = p["factors"]
   274:             rows = shape[0] if len(shape) >= 1 else 1
   275:             cols = numel // rows
   276:             if not (torch.is_tensor(P) and torch.is_tensor(Q)
   277:                     and P.dim() == 2 and Q.dim() == 2
   278:                     and P.shape[0] == rows and Q.shape[0] == cols
   279:                     and P.shape[1] == Q.shape[1] and P.shape[1] >= 1):
   280:                 raise CompressionContractError(
   281:                     f"factors must be P ({rows} x r) and Q ({cols} x r)")
   282:             out.append(("lowrank",
   283:                         P.detach().to(device=device, dtype=torch.float32) + 0.0,
   284:                         Q.detach().to(device=device, dtype=torch.float32) + 0.0))
   285:             continue
   286:         if not (set(p) <= {"values", "bits", "indices"}
   287:                 and "values" in p and "bits" in p):
   288:             raise CompressionContractError(
   289:                 "a packet is {'values', 'bits'[, 'indices']} or {'factors'}")
   290:         b = p["bits"]
   291:         if isinstance(b, bool) or not isinstance(b, int) or not 1 <= b <= 32:
   292:             raise CompressionContractError("'bits' must be an int in [1, 32]")
   293:         v = p["values"]
   294:         if not torch.is_tensor(v) or v.is_complex() or v.dtype == torch.bool:
   295:             raise CompressionContractError("'values' must be a real tensor")
   296:         v = v.detach().reshape(-1).to(device=device, dtype=torch.float32) + 0.0
   297:         idx = p.get("indices")
   298:         if idx is None:
   299:             if v.numel() != numel:
   300:                 raise CompressionContractError(
   301:                     f"a dense packet needs {numel} values, got {v.numel()}")
   302:             out.append(("dense", v, b))
   303:             continue
   304:         if (not torch.is_tensor(idx) or idx.is_floating_point()
   305:                 or idx.is_complex() or idx.dtype == torch.bool):
   306:             raise CompressionContractError("'indices' must be an integer tensor")
   307:         idx = idx.detach().reshape(-1).to(device=device,
   308:                                           dtype=torch.int64).clone()
   309:         if idx.numel() != v.numel():
   310:             raise CompressionContractError(
   311:                 "'indices' and 'values' differ in length")
   312:         out.append(("sparse", v, b, idx))
   313:     return out
   314: 
   315: 
   316: def _account(tensors, device, *, _to_device=_to_device):
   317:     """Charge and decode the packets of one step, all on the device.
   318: 
   319:     `tensors` is a list of (numel, parts), one per gradient tensor. Cost:
   320:     len(values) * bits, plus 32 bits per distinct value when bits < 32
   321:     (the codebook, at most 2**bits entries), plus the Elias-gamma code of
   322:     each sparse packet's sorted position gaps, plus 32 bits per low-rank
   323:     factor entry. Returns (bits, ok, flat): ok = [finite, indices valid,
   324:     codebook valid], flat = the decoded gradients, concatenated."""
   325:     t_true = torch.ones((), dtype=torch.bool, device=device)
   326:     total = sum(numel for numel, _ in tensors)
   327:     flat = torch.zeros(total, dtype=torch.float32, device=device)
   328:     static = 0
   329:     finite, cb_vals, cb_caps = [], [], []
   330:     sp_idx, sp_val, sp_meta = [], [], []  # meta: len, numel, off, part off
   331:     off = part_off = 0
   332:     for numel, parts in tensors:
   333:         for part in parts:
   334:             if part[0] == "lowrank":
   335:                 P, Q = part[1], part[2]
   336:                 static += 32 * (P.numel() + Q.numel())
   337:                 finite += [P.reshape(-1), Q.reshape(-1)]
   338:                 flat[off:off + numel].add_((P @ Q.t()).reshape(-1))
   339:                 continue
   340:             v, b = part[1], part[2]
   341:             static += v.numel() * b
   342:             finite.append(v)
   343:             if b < 32:
   344:                 cb_vals.append(v)
   345:                 cb_caps.append(2 ** b)
   346:             if part[0] == "dense":
   347:                 flat[off:off + numel].add_(v)
   348:             else:
   349:                 sp_idx.append(part[3])
   350:                 sp_val.append(v)
   351:                 sp_meta.append((v.numel(), numel, off, part_off))
   352:                 part_off += numel
   353:         off += numel
   354:     bits = torch.full((), float(static), dtype=torch.float64, device=device)
   355:     fin_ok = torch.isfinite(torch.cat(finite)).all() if finite else t_true
   356:     cb_ok, idx_ok = t_true, t_true
   357:     if cb_vals:
   358:         lens = [v.numel() for v in cb_vals]
   359:         v = torch.cat(cb_vals)
   360:         meta = _to_device([lens, cb_caps], torch.int64, device)
   361:         seg = torch.repeat_interleave(
   362:             torch.arange(len(lens), device=device), meta[0],
   363:             output_size=sum(lens))
   364:         order = torch.sort(v, stable=True).indices
   365:         order = order[torch.sort(seg[order], stable=True).indices]
   366:         vs, ss = v[order], seg[order]
   367:         start = torch.ones_like(vs, dtype=torch.bool)
   368:         start[1:] = (vs[1:] != vs[:-1]) | (ss[1:] != ss[:-1])
   369:         levels = torch.zeros(len(lens), dtype=torch.int64, device=device)
   370:         levels.index_add_(0, ss, start.long())
   371:         cb_ok = (levels <= meta[1]).all()
   372:         bits = bits + 32 * levels.sum()
   373:     if sp_idx:
   374:         lens = [m[0] for m in sp_meta]
   375:         meta = _to_device([list(col) for col in zip(*sp_meta)],
   376:                           torch.int64, device)
   377:         seg = torch.repeat_interleave(
   378:             torch.arange(len(lens), device=device), meta[0],
   379:             output_size=sum(lens))
   380:         idx = torch.cat(sp_idx)
   381:         numels = meta[1][seg]
   382:         in_range = ((idx >= 0) & (idx < numels)).all()
   383:         # Invalid indices fail the check; clamp so that, until the run
   384:         # stops, the scatter stays inside the packet's own tensor.
   385:         safe = torch.minimum(idx.clamp(min=0), numels - 1)
   386:         flat.index_add_(0, safe + meta[2][seg], torch.cat(sp_val))
   387:         order = torch.sort(safe + meta[3][seg]).indices
   388:         local, ss = idx[order], seg[order]
   389:         first = torch.ones_like(local, dtype=torch.bool)
   390:         first[1:] = ss[1:] != ss[:-1]
   391:         gap = torch.where(first, local + 1, local - torch.roll(local, 1))
   392:         idx_ok = in_range & (gap > 0).all()
   393:         bits = bits + (2 * torch.floor(torch.log2(
   394:             gap.clamp(min=1).double())) + 1).sum()
   395:     return bits, torch.stack([fin_ok, idx_ok, cb_ok]), flat
   396: 
   397: 
   398: def _check(ok):
   399:     if not ok[0]:
   400:         raise CompressionContractError("a packet carries non-finite values")
   401:     if not ok[1]:
   402:         raise CompressionContractError(
   403:             "packet indices must be distinct and inside the tensor")
   404:     if not ok[2]:
   405:         raise CompressionContractError(
   406:             "a packet has more distinct values than 2**bits")
   407: 
   408: 
   409: def packet_cost(packet, shape, *, _parts=_parts, _account=_account,
   410:                 _check=_check):
   411:     """Bits `packet` (a packet or a list of them) costs for a tensor of
   412:     `shape`, exactly as the per-step budget check charges it."""
   413:     shape = torch.Size(shape)
   414:     first = packet if isinstance(packet, dict) else (packet or [{}])[0]
   415:     t = first.get("values") if isinstance(first, dict) else None
   416:     if t is None and isinstance(first, dict) and first.get("factors"):
   417:         t = first["factors"][0]
   418:     device = t.device if torch.is_tensor(t) else torch.device("cpu")
   419:     bits, ok, _ = _account([(shape.numel(), _parts(packet, shape, device))],
   420:                            device)
   421:     _check(ok.tolist())
   422:     return int(bits.item())
   423: 
   424: 
   425: def apply_gradient_compression(model, compressor, check_bindings, *,
   426:                                _parts=_parts, _account=_account):
   427:     """Encode every gradient with the compressor and replace it with the
   428:     decoding of its packet: decoding is fixed, so the optimizer sees only
   429:     what was transmitted. Returns the step's receipt (bits, then the three
   430:     validity checks), filled asynchronously; `settle_step` reads it after
   431:     the training loop's next host sync and stops the run on a violation.
   432:     `check_bindings` runs after every compress call (see `bindings_check`)."""
   433:     staged = []
   434:     for name, param in model.named_parameters():
   435:         if param.grad is None:
   436:             continue
   437:         packet = compressor.compress(param.grad.detach(), name)
   438:         check_bindings()
   439:         staged.append((param, _parts(packet, param.shape, param.device)))
   440:     if not staged:
   441:         return torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float64), None
   442:     device = staged[0][0].device
   443:     bits, ok, flat = _account([(p.numel(), parts) for p, parts in staged],
   444:                               device)
   445:     receipt, done = torch.cat([bits.reshape(1), ok.double()]), None
   446:     if device.type == "cuda":
   447:         host = torch.empty(4, dtype=torch.float64, pin_memory=True)
   448:         receipt = host.copy_(receipt, non_blocking=True)
   449:         done = torch.cuda.Event()
   450:         done.record()
   451:     off = 0
   452:     for param, _ in staged:
   453:         n = param.numel()
   454:         param.grad = flat[off:off + n].view(param.shape).to(param.dtype)
   455:         off += n
   456:     return receipt, done
   457: 
   458: 
   459: def settle_step(receipt, budget_bits, *, _check=_check):
   460:     """Check a step's receipt against the contract and the budget; returns
   461:     the bits the step transmitted. Cheap after a host sync (loss.item())."""
   462:     receipt, done = receipt
   463:     if done is not None:
   464:         done.synchronize()
   465:     result = receipt.tolist()
   466:     _check([bool(x) for x in result[1:]])
   467:     if result[0] > budget_bits:
   468:         raise CompressionContractError(
   469:             f"a step transmitted {result[0]:.0f} bits, over the budget of "
   470:             f"{budget_bits} bits (compress_ratio x 32 bits x parameter entries)")
   471:     return result[0]
   472: 
   473: 
   474: # ============================================================================
   475: # FIXED SECTION — Training Loop
   476: # ============================================================================
   477: 
   478: def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, base_lr, min_lr=0.0):
   479:     """Cosine learning rate schedule with linear warmup."""
   480:     if epoch < warmup_epochs:
   481:         lr = base_lr * (epoch + 1) / (warmup_epochs + 1)
   482:     else:
   483:         progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
   484:         lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
   485:     for param_group in optimizer.param_groups:
   486:         param_group['lr'] = lr
   487:     return lr
   488: 
   489: 
   490: def evaluate(model, test_loader, device):
   491:     model.eval()
   492:     correct = 0
   493:     total = 0
   494:     total_loss = 0.0
   495:     with torch.no_grad():
   496:         for images, labels in test_loader:
   497:             images, labels = images.to(device), labels.to(device)
   498:             outputs = model(images)
   499:             loss = F.cross_entropy(outputs, labels, reduction='sum')
   500:             total_loss += loss.item()

[truncated: showing at most 500 lines / 60000 bytes from pytorch-vision/custom_compressor.py]
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `topk_ef` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_compressor.py`:

```python
Lines 182–224:
   179: # ============================================================================
   180: # EDITABLE SECTION — Gradient Compressor (lines 182-224)
   181: # ============================================================================
   182: class Compressor:
   183:     """TopK sparsification with error feedback (EF-TopK).
   184: 
   185:     Keeps the K largest-magnitude entries of each (error-corrected)
   186:     gradient, sent as float32 values with Elias-gamma coded positions.
   187:     K is the largest count whose worst-case packet fits the tensor's share
   188:     of the step budget (one entry per tensor first, the rest of the budget
   189:     split in proportion to tensor size). Error feedback accumulates what
   190:     was not sent and adds it to the next gradient before compression.
   191:     """
   192: 
   193:     def __init__(self, compress_ratio, param_numels, budget_bits):
   194:         self.compress_ratio = compress_ratio
   195:         self.residuals = {}
   196:         self.k = self._plan(param_numels, budget_bits,
   197:                             lambda k, n: 32 * k + elias_gamma_bound(k, n))
   198: 
   199:     @staticmethod
   200:     def _plan(param_numels, budget_bits, cost):
   201:         total = sum(param_numels.values())
   202:         spare = budget_bits - sum(cost(1, n) for n in param_numels.values())
   203:         plan = {}
   204:         for name, n in param_numels.items():
   205:             share = cost(1, n) + spare * n // total
   206:             lo, hi = 1, n
   207:             while lo < hi:
   208:                 mid = (lo + hi + 1) // 2
   209:                 if cost(mid, n) <= share:
   210:                     lo = mid
   211:                 else:
   212:                     hi = mid - 1
   213:             plan[name] = lo
   214:         return plan
   215: 
   216:     def compress(self, tensor, name):
   217:         if name in self.residuals:
   218:             tensor = tensor + self.residuals[name]
   219:         flat = tensor.flatten()
   220:         _, indices = torch.topk(flat.abs(), self.k[name], sorted=False)
   221:         values = flat[indices]
   222:         sent = torch.zeros_like(flat).scatter_(0, indices, values)
   223:         self.residuals[name] = (flat - sent).view(tensor.shape)
   224:         return {"values": values, "bits": 32, "indices": indices}
   225: 
   226: 
   227: # ============================================================================
```

### `qsgd` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_compressor.py`:

```python
Lines 182–255:
   179: # ============================================================================
   180: # EDITABLE SECTION — Gradient Compressor (lines 182-224)
   181: # ============================================================================
   182: class Compressor:
   183:     """QSGD with Elias coding, at the step budget.
   184: 
   185:     Unbiased stochastic quantization of |g_i| / ||g|| to s levels
   186:     (Alistarh et al., NeurIPS 2017), sent in the paper's sparse encoding:
   187:     sign-and-level codes (a codebook of the values +-l * ||g|| / s and 0)
   188:     at Elias-gamma coded positions. Each tensor uses the largest
   189:     power-of-two s (up to 256) whose expected packet, measured on its
   190:     previous gradient, fits its share of the step budget; the packet
   191:     carries at most K entries, K the most that fit that share (should a
   192:     gradient have more nonzero levels, the largest are kept). No error
   193:     feedback (QSGD is unbiased). Per-tensor gradient clipping keeps
   194:     quantization noise from diverging.
   195:     """
   196: 
   197:     def __init__(self, compress_ratio, param_numels, budget_bits):
   198:         self.compress_ratio = compress_ratio
   199:         self.clip_norm = 1.0
   200:         self.levels = [2 ** j for j in range(9)]  # s = 1 .. 256
   201:         total = sum(param_numels.values())
   202:         floor = {n: self._cost(1, 1, n) for n in set(param_numels.values())}
   203:         spare = budget_bits - sum(floor[n] for n in param_numels.values())
   204:         self.share = {name: floor[n] + spare * n // total
   205:                       for name, n in param_numels.items()}
   206:         self.expected = {}  # name -> (host tensor of E[nnz] per s, event)
   207: 
   208:     @staticmethod
   209:     def _cost(k, s, n):
   210:         bits = math.ceil(math.log2(2 * s + 1))
   211:         return k * bits + 32 * (2 * s + 1) + elias_gamma_bound(k, n)
   212: 
   213:     def _largest_k(self, s, n, share):
   214:         lo, hi = 0, n
   215:         while lo < hi:
   216:             mid = (lo + hi + 1) // 2
   217:             if self._cost(mid, s, n) <= share:
   218:                 lo = mid
   219:             else:
   220:                 hi = mid - 1
   221:         return lo
   222: 
   223:     def compress(self, tensor, name):
   224:         flat = tensor.flatten()
   225:         n = flat.numel()
   226:         share = self.share[name]
   227:         s = 1
   228:         if name in self.expected:
   229:             host, done = self.expected[name]
   230:             if done is not None:
   231:                 done.synchronize()
   232:             for s_try, e in zip(self.levels, host.tolist()):
   233:                 if self._cost(min(n, math.ceil(1.1 * e) + 8), s_try, n) <= share:
   234:                     s = s_try
   235:         k = min(n, self._largest_k(s, n, share))
   236:         flat = flat * (self.clip_norm / (flat.norm() + 1e-6)).clamp(max=1.0)
   237:         norm = flat.norm()
   238:         unit = flat.abs() / norm.clamp(min=1e-30)
   239:         # Expected nonzeros at each s, read at this tensor's next call.
   240:         s_vec = torch.tensor(self.levels, dtype=unit.dtype).to(unit.device)
   241:         e = (s_vec.unsqueeze(1) * unit.unsqueeze(0)).clamp(max=1.0).sum(1)
   242:         done = None
   243:         if e.is_cuda:
   244:             e = torch.empty(e.shape, dtype=e.dtype,
   245:                             pin_memory=True).copy_(e, non_blocking=True)
   246:             done = torch.cuda.Event()
   247:             done.record()
   248:         self.expected[name] = (e, done)
   249:         level = s * unit
   250:         level = level.floor() + (torch.rand_like(level) < level - level.floor()).float()
   251:         idx = torch.topk((level > 0).float() * (1.0 + unit), k,
   252:                          sorted=False).indices
   253:         values = flat[idx].sign() * level[idx] * (norm / s)
   254:         return {"values": values, "bits": math.ceil(math.log2(2 * s + 1)),
   255:                 "indices": idx}
   256: 
   257: 
   258: # ============================================================================
```

### `signsgd` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_compressor.py`:

```python
Lines 182–227:
   179: # ============================================================================
   180: # EDITABLE SECTION — Gradient Compressor (lines 182-224)
   181: # ============================================================================
   182: class Compressor:
   183:     """Scaled signSGD with error feedback, at the step budget.
   184: 
   185:     Sends one sign bit per entry, scaled by the mean magnitude of the sent
   186:     entries (a two-value codebook), with error feedback carrying what the
   187:     sign loses into the next gradient. One bit for every entry would be
   188:     32x, not the 100x budget, so each tensor sends the signs of its K
   189:     largest-magnitude error-corrected entries, K being the largest count
   190:     whose worst-case packet (sign bits, codebook, Elias-gamma coded
   191:     positions) fits the tensor's share of the step budget.
   192:     """
   193: 
   194:     def __init__(self, compress_ratio, param_numels, budget_bits):
   195:         self.compress_ratio = compress_ratio
   196:         self.residuals = {}
   197:         self.k = self._plan(param_numels, budget_bits,
   198:                             lambda k, n: k + 64 + elias_gamma_bound(k, n))
   199: 
   200:     @staticmethod
   201:     def _plan(param_numels, budget_bits, cost):
   202:         total = sum(param_numels.values())
   203:         spare = budget_bits - sum(cost(1, n) for n in param_numels.values())
   204:         plan = {}
   205:         for name, n in param_numels.items():
   206:             share = cost(1, n) + spare * n // total
   207:             lo, hi = 1, n
   208:             while lo < hi:
   209:                 mid = (lo + hi + 1) // 2
   210:                 if cost(mid, n) <= share:
   211:                     lo = mid
   212:                 else:
   213:                     hi = mid - 1
   214:             plan[name] = lo
   215:         return plan
   216: 
   217:     def compress(self, tensor, name):
   218:         if name in self.residuals:
   219:             tensor = tensor + self.residuals[name]
   220:         flat = tensor.flatten()
   221:         _, indices = torch.topk(flat.abs(), self.k[name], sorted=False)
   222:         picked = flat[indices]
   223:         scale = picked.abs().mean()
   224:         values = torch.where(picked >= 0, scale, -scale)
   225:         sent = torch.zeros_like(flat).scatter_(0, indices, values)
   226:         self.residuals[name] = (flat - sent).view(tensor.shape)
   227:         return {"values": values, "bits": 1, "indices": indices}
   228: 
   229: 
   230: # ============================================================================
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
