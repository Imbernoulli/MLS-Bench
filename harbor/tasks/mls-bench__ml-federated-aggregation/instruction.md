# MLS-Bench: ml-federated-aggregation

# Federated Learning Aggregation Strategy Design

## Research Question
Design a server-side aggregation strategy for federated learning that converges faster and to a better-performing global model under heterogeneous (non-IID) client data. The contribution is the *aggregation rule* (and optionally the client-selection / client-side correction exposed by this interface), not changes to the local optimizer or simulation harness.

## Background
Federated Learning (FL) trains a shared global model across many clients without centralizing data. Under non-IID client data, naive averaging suffers from "client drift" — local updates diverge, slowing or destabilizing convergence.

Reference baselines:
- **FedAvg** — McMahan, Moore, Ramage, Hampson, Agüera y Arcas, AISTATS 2017 ([arXiv:1602.05629](https://arxiv.org/abs/1602.05629)). Server averages client model parameters weighted by `n_k / sum(n_k)` (number of samples per client). No server-side state.
- **FedProx** — Li, Sahu, Zaheer, Sanjabi, Talwalkar, Smith, MLSys 2020 ([arXiv:1812.06127](https://arxiv.org/abs/1812.06127)). Same server aggregation as FedAvg, but each client adds a proximal term `(mu/2) * ||w - w_global||^2` to its local objective; default `mu = 0.01`.
- **SCAFFOLD** — Karimireddy, Kale, Mohri, Reddi, Stich, Suresh, ICML 2020 ([arXiv:1910.06378](https://arxiv.org/abs/1910.06378)). Maintains server- and client-side control variates `c, c_i` to correct client drift. Local update: `w <- w - eta * (g_i - c_i + c)`. Server updates `c` after each round from received deltas.

## Implementation Contract
Modify `ServerAggregator` in `flower/custom_fl_aggregation.py`:

```python
class ServerAggregator:
    def __init__(self, global_model, args):
        # Initialize aggregation state (momentum buffers, control variates, ...).
        ...

    def aggregate(self, global_state_dict, client_updates, round_num):
        # global_state_dict: OrderedDict of current global model parameters
        # client_updates: list of (state_dict, num_samples, avg_loss) tuples
        # round_num: current communication round (0-indexed)
        # Returns: OrderedDict of updated global model parameters.
        ...

    def select_clients(self, num_available, num_to_select, round_num):
        # Returns list of client indices to participate this round.
        ...
```

## Fixed Pipeline
The federated simulation pipeline (number of communication rounds, client
population and per-round participation, local training schedule, optimizer,
datasets, non-IID partitioning, and evaluation) is fixed by the harness and not
editable. Your contribution must be confined to the strategy in the editable
region.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/flower/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `flower/custom_fl_aggregation.py`
- editable lines **340–420**




## Readable Context


### `flower/custom_fl_aggregation.py`  [EDITABLE — lines 340–420 only]

```python
     1: # Custom federated learning aggregation strategy for MLS-Bench
     2: #
     3: # EDITABLE section: ServerAggregator class (aggregate method + helpers).
     4: # FIXED sections: everything else (config, data partitioning, client training,
     5: #                 FL simulation loop, evaluation).
     6: import argparse
     7: import copy
     8: import json
     9: import os
    10: import random
    11: import time
    12: from collections import OrderedDict
    13: from pathlib import Path
    14: 
    15: import numpy as np
    16: import torch
    17: import torch.nn as nn
    18: import torch.nn.functional as F
    19: import torch.optim as optim
    20: from torch.utils.data import DataLoader, Dataset, Subset
    21: 
    22: 
    23: # =====================================================================
    24: # FIXED: Configuration
    25: # =====================================================================
    26: def parse_args():
    27:     parser = argparse.ArgumentParser(description="Federated Learning Simulation")
    28:     parser.add_argument("--dataset", type=str, default="cifar10",
    29:                         choices=["cifar10", "femnist", "shakespeare"])
    30:     parser.add_argument("--data-dir", type=str, default="/data")
    31:     parser.add_argument("--num-clients", type=int, default=100,
    32:                         help="Total number of clients")
    33:     parser.add_argument("--clients-per-round", type=int, default=10,
    34:                         help="Number of clients sampled per round")
    35:     parser.add_argument("--num-rounds", type=int, default=200,
    36:                         help="Number of communication rounds")
    37:     parser.add_argument("--local-epochs", type=int, default=5,
    38:                         help="Number of local training epochs per round")
    39:     parser.add_argument("--local-lr", type=float, default=0.01,
    40:                         help="Local SGD learning rate")
    41:     parser.add_argument("--local-batch-size", type=int, default=64,
    42:                         help="Local training batch size")
    43:     parser.add_argument("--dirichlet-alpha", type=float, default=0.1,
    44:                         help="Dirichlet concentration for non-IID split (CIFAR-10)")
    45:     parser.add_argument("--seed", type=int, default=42)
    46:     parser.add_argument("--output-dir", type=str, default="./output")
    47:     parser.add_argument("--eval-every", type=int, default=10,
    48:                         help="Evaluate global model every N rounds")
    49:     return parser.parse_args()
    50: 
    51: 
    52: # =====================================================================
    53: # FIXED: Models
    54: # =====================================================================
    55: class CifarCNN(nn.Module):
    56:     """Simple CNN for CIFAR-10 (used in FedAvg/FedProx literature)."""
    57: 
    58:     def __init__(self):
    59:         super().__init__()
    60:         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    61:         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    62:         self.pool = nn.MaxPool2d(2, 2)
    63:         self.fc1 = nn.Linear(64 * 8 * 8, 512)
    64:         self.fc2 = nn.Linear(512, 10)
    65: 
    66:     def forward(self, x):
    67:         x = self.pool(F.relu(self.conv1(x)))
    68:         x = self.pool(F.relu(self.conv2(x)))
    69:         x = x.view(x.size(0), -1)
    70:         x = F.relu(self.fc1(x))
    71:         return self.fc2(x)
    72: 
    73: 
    74: class FemnistCNN(nn.Module):
    75:     """CNN for FEMNIST (62 classes: digits + upper + lower)."""
    76: 
    77:     def __init__(self):
    78:         super().__init__()
    79:         self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
    80:         self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
    81:         self.pool = nn.MaxPool2d(2, 2)
    82:         self.fc1 = nn.Linear(64 * 7 * 7, 2048)
    83:         self.fc2 = nn.Linear(2048, 62)
    84: 
    85:     def forward(self, x):
    86:         x = self.pool(F.relu(self.conv1(x)))
    87:         x = self.pool(F.relu(self.conv2(x)))
    88:         x = x.view(x.size(0), -1)
    89:         x = F.relu(self.fc1(x))
    90:         return self.fc2(x)
    91: 
    92: 
    93: class CharLSTM(nn.Module):
    94:     """Character-level LSTM for Shakespeare next-char prediction."""
    95: 
    96:     def __init__(self, vocab_size=80, embed_dim=8, hidden_dim=256, num_layers=2):
    97:         super().__init__()
    98:         self.embed = nn.Embedding(vocab_size, embed_dim)
    99:         self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
   100:                             batch_first=True)
   101:         self.fc = nn.Linear(hidden_dim, vocab_size)
   102:         self.vocab_size = vocab_size
   103: 
   104:     def forward(self, x):
   105:         # x: (batch, seq_len) of char indices
   106:         emb = self.embed(x)
   107:         out, _ = self.lstm(emb)
   108:         logits = self.fc(out)  # (batch, seq_len, vocab_size)
   109:         return logits
   110: 
   111: 
   112: # =====================================================================
   113: # FIXED: Dataset loading and non-IID partitioning
   114: # =====================================================================
   115: class ShakespeareCharDataset(Dataset):
   116:     """Character-level Shakespeare dataset for next-char prediction."""
   117: 
   118:     def __init__(self, text, seq_len=80, char2idx=None):
   119:         self.seq_len = seq_len
   120:         if char2idx is not None:
   121:             self.char2idx = char2idx
   122:         else:
   123:             chars = sorted(set(text))
   124:             self.char2idx = {c: i for i, c in enumerate(chars)}
   125:         self.idx2char = {i: c for c, i in self.char2idx.items()}
   126:         self.vocab_size = len(self.char2idx)
   127:         self.data = [self.char2idx[c] for c in text if c in self.char2idx]
   128: 
   129:     def __len__(self):
   130:         return max(0, len(self.data) - self.seq_len - 1)
   131: 
   132:     def __getitem__(self, idx):
   133:         x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
   134:         y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
   135:         return x, y
   136: 
   137: 
   138: def load_cifar10(data_dir):
   139:     """Load CIFAR-10 using torchvision."""
   140:     import torchvision
   141:     import torchvision.transforms as T
   142:     transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465),
   143:                                                       (0.2023, 0.1994, 0.2010))])
   144:     train_set = torchvision.datasets.CIFAR10(
   145:         os.path.join(data_dir, "cifar10"), train=True, transform=transform)
   146:     test_set = torchvision.datasets.CIFAR10(
   147:         os.path.join(data_dir, "cifar10"), train=False, transform=transform)
   148:     return train_set, test_set
   149: 
   150: 
   151: def load_femnist(data_dir):
   152:     """Load EMNIST ByClass split (simulates FEMNIST)."""
   153:     import torchvision
   154:     import torchvision.transforms as T
   155:     transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
   156:     train_set = torchvision.datasets.EMNIST(
   157:         os.path.join(data_dir, "emnist"), split="byclass",
   158:         train=True, transform=transform)
   159:     test_set = torchvision.datasets.EMNIST(
   160:         os.path.join(data_dir, "emnist"), split="byclass",
   161:         train=False, transform=transform)
   162:     return train_set, test_set
   163: 
   164: 
   165: def load_shakespeare(data_dir):
   166:     """Load Shakespeare text and split by character/speaker."""
   167:     text_path = os.path.join(data_dir, "shakespeare", "input.txt")
   168:     with open(text_path, "r") as f:
   169:         text = f.read()
   170:     return text
   171: 
   172: 
   173: def dirichlet_partition(targets, num_clients, alpha, seed=42):
   174:     """Partition dataset indices using Dirichlet distribution for non-IID split."""
   175:     rng = np.random.default_rng(seed)
   176:     targets = np.array(targets)
   177:     num_classes = len(np.unique(targets))
   178:     client_indices = [[] for _ in range(num_clients)]
   179: 
   180:     for c in range(num_classes):
   181:         class_indices = np.where(targets == c)[0]
   182:         rng.shuffle(class_indices)
   183:         proportions = rng.dirichlet(np.repeat(alpha, num_clients))
   184:         proportions = proportions / proportions.sum()
   185:         split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)
   186:         splits = np.split(class_indices, split_points[:-1])
   187:         for i, split in enumerate(splits):
   188:             client_indices[i].extend(split.tolist())
   189: 
   190:     # Shuffle each client's data
   191:     for i in range(num_clients):
   192:         rng.shuffle(client_indices[i])
   193: 
   194:     return client_indices
   195: 
   196: 
   197: def shakespeare_partition(text, num_clients, seed=42):
   198:     """Partition Shakespeare text by speaker (naturally non-IID).
   199: 
   200:     Falls back to chunk-based partitioning if parsing fails.
   201:     """
   202:     rng = np.random.default_rng(seed)
   203:     # Split by speaker blocks (lines starting with all-caps name followed by colon)
   204:     import re
   205:     blocks = re.split(r'\n(?=[A-Z][A-Z ]+:)', text)
   206:     blocks = [b for b in blocks if len(b.strip()) > 100]
   207: 
   208:     if len(blocks) < num_clients:
   209:         # Fallback: chunk-based
   210:         chunk_size = len(text) // num_clients
   211:         return [text[i * chunk_size:(i + 1) * chunk_size] for i in range(num_clients)]
   212: 
   213:     rng.shuffle(blocks)
   214:     client_texts = [""] * num_clients
   215:     for i, block in enumerate(blocks):
   216:         client_texts[i % num_clients] += block
   217: 
   218:     return client_texts
   219: 
   220: 
   221: def prepare_data(args):
   222:     """Prepare dataset, partition among clients, return client datasets + test set."""
   223:     if args.dataset == "cifar10":
   224:         train_set, test_set = load_cifar10(args.data_dir)
   225:         targets = train_set.targets
   226:         client_indices = dirichlet_partition(
   227:             targets, args.num_clients, args.dirichlet_alpha, args.seed)
   228:         client_datasets = [Subset(train_set, idx) for idx in client_indices]
   229:         model_fn = CifarCNN
   230:         loss_fn = nn.CrossEntropyLoss()
   231:         return client_datasets, test_set, model_fn, loss_fn
   232: 
   233:     elif args.dataset == "femnist":
   234:         train_set, test_set = load_femnist(args.data_dir)
   235:         targets = train_set.targets.numpy()
   236:         client_indices = dirichlet_partition(
   237:             targets, args.num_clients, args.dirichlet_alpha, args.seed)
   238:         client_datasets = [Subset(train_set, idx) for idx in client_indices]
   239:         model_fn = FemnistCNN
   240:         loss_fn = nn.CrossEntropyLoss()
   241:         return client_datasets, test_set, model_fn, loss_fn
   242: 
   243:     elif args.dataset == "shakespeare":
   244:         text = load_shakespeare(args.data_dir)
   245:         client_texts = shakespeare_partition(text, args.num_clients, args.seed)
   246:         # Create per-client datasets
   247:         full_ds = ShakespeareCharDataset(text)
   248:         vocab_size = full_ds.vocab_size
   249:         char2idx = full_ds.char2idx
   250:         client_datasets = [ShakespeareCharDataset(t, char2idx=char2idx)
   251:                            for t in client_texts if len(t) > 100]
   252:         # Pad to num_clients if needed
   253:         while len(client_datasets) < args.num_clients:
   254:             client_datasets.append(client_datasets[-1])
   255:         # Test set: last 10% of full text
   256:         split_pt = int(len(text) * 0.9)
   257:         test_ds = ShakespeareCharDataset(text[split_pt:], char2idx=char2idx)
   258:         model_fn = lambda: CharLSTM(vocab_size=vocab_size)
   259:         loss_fn = nn.CrossEntropyLoss()
   260:         return client_datasets, test_ds, model_fn, loss_fn
   261: 
   262:     else:
   263:         raise ValueError(f"Unknown dataset: {args.dataset}")
   264: 
   265: 
   266: # =====================================================================
   267: # FIXED: default helpers shared by every FL Strategy
   268: # =====================================================================
   269: def _default_client_sgd(model, loader, loss_fn, local_epochs, local_lr, device,
   270:                         loss_aug=None):
   271:     """Plain SGD loop used by the default Strategy. ``loss_aug`` optionally
   272:     adds a term to each mini-batch loss (e.g. FedProx's proximal term)."""
   273:     optimizer = optim.SGD(model.parameters(), lr=local_lr)
   274:     total_loss, total_samples = 0.0, 0
   275:     for _ in range(local_epochs):
   276:         for batch_data in loader:
   277:             if len(batch_data) != 2:
   278:                 continue
   279:             inputs, targets = batch_data
   280:             inputs, targets = inputs.to(device), targets.to(device)
   281:             optimizer.zero_grad()
   282:             outputs = model(inputs)
   283:             if outputs.dim() == 3:
   284:                 outputs = outputs.view(-1, outputs.size(-1))
   285:                 targets = targets.view(-1)
   286:             loss = loss_fn(outputs, targets)
   287:             if loss_aug is not None:
   288:                 loss = loss + loss_aug(model)
   289:             loss.backward()
   290:             optimizer.step()
   291:             total_loss += loss.item() * inputs.size(0)
   292:             total_samples += inputs.size(0)
   293:     return total_loss / max(total_samples, 1), total_samples
   294: 
   295: 
   296: # =====================================================================
   297: # FIXED: Evaluation
   298: # =====================================================================
   299: @torch.no_grad()
   300: def evaluate_global_model(model, test_set, loss_fn, device, batch_size=256):
   301:     """Evaluate the global model on the test set; returns (loss, accuracy)."""
   302:     model.eval()
   303:     model.to(device)
   304:     loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
   305: 
   306:     total_loss = 0.0
   307:     total_correct = 0
   308:     total_samples = 0
   309: 
   310:     for batch_data in loader:
   311:         inputs, targets = batch_data
   312:         inputs, targets = inputs.to(device), targets.to(device)
   313:         outputs = model(inputs)
   314: 
   315:         if outputs.dim() == 3:
   316:             # Shakespeare: flatten for loss and accuracy
   317:             outputs_flat = outputs.view(-1, outputs.size(-1))
   318:             targets_flat = targets.view(-1)
   319:             loss = loss_fn(outputs_flat, targets_flat)
   320:             preds = outputs_flat.argmax(dim=-1)
   321:             total_correct += (preds == targets_flat).sum().item()
   322:             total_samples += targets_flat.numel()
   323:         else:
   324:             loss = loss_fn(outputs, targets)
   325:             preds = outputs.argmax(dim=-1)
   326:             total_correct += (preds == targets).sum().item()
   327:             total_samples += targets.size(0)
   328: 
   329:         total_loss += loss.item() * inputs.size(0)
   330: 
   331:     avg_loss = total_loss / max(total_samples, 1)
   332:     accuracy = total_correct / max(total_samples, 1)
   333:     model.cpu()
   334:     return avg_loss, accuracy
   335: 
   336: 
   337: # =====================================================================
   338: # EDITABLE: FL Strategy — owns BOTH client-side and server-side logic
   339: # =====================================================================
   340: class Strategy:
   341:     """End-to-end FL strategy.
   342: 
   343:     Unlike a pure ServerAggregator, this class is responsible for the FULL
   344:     FL recipe: how each client trains locally AND how the server aggregates.
   345:     This matches the scope of real FL methods (FedProx / SCAFFOLD / FedDyn /
   346:     MOON / ...) whose innovations live inside the local loop — a server-only
   347:     API can only approximate them.
   348: 
   349:     The run_fl_simulation loop calls:
   350:         strategy = Strategy(global_model, args)
   351:         selected = strategy.select_clients(N, K, round_num)
   352:         # for each client i:
   353:         state_i, n_i, loss_i = strategy.client_local_train(
   354:             global_state, client_dataset, model_fn, loss_fn,
   355:             local_epochs, local_lr, local_batch_size, device, client_idx)
   356:         global_state = strategy.aggregate(global_state, [(state_i, n_i, loss_i)], round_num)
   357: 
   358:     You SHOULD override at least ``client_local_train`` and/or ``aggregate``
   359:     to implement your method. Sensible defaults (plain SGD + weighted
   360:     average) are provided so trivial subclasses still run.
   361: 
   362:     Innovation space:
   363:         * client-side: proximal regularization (FedProx), per-step
   364:           control-variate correction (SCAFFOLD), dynamic regularizer
   365:           (FedDyn), contrastive loss (MOON), adaptive local LR, ...
   366:         * server-side: sample-weighted / robust / learning-rate-adapted
   367:           aggregation, server momentum, drift estimation, ...
   368:     """
   369: 
   370:     def __init__(self, global_model, args):
   371:         """Initialize strategy state.
   372: 
   373:         Args:
   374:             global_model: the initial global nn.Module.
   375:             args: parsed CLI args (num_clients, clients_per_round, ...).
   376:         """
   377:         self.args = args
   378: 
   379:     def client_local_train(self, global_state_dict, client_dataset, model_fn,
   380:                            loss_fn, local_epochs, local_lr, local_batch_size,
   381:                            device, client_idx):
   382:         """Train one client locally.
   383: 
   384:         Must return ``(state_dict, num_samples, avg_loss)``. The default
   385:         implementation does plain SGD (no momentum). Override to add
   386:         proximal / correction / regularizer terms to the local objective
   387:         or to modify the optimizer / gradient flow.
   388:         """
   389:         model = model_fn()
   390:         model.load_state_dict(global_state_dict)
   391:         model.to(device)
   392:         model.train()
   393:         loader = DataLoader(client_dataset, batch_size=local_batch_size,
   394:                             shuffle=True, drop_last=False, num_workers=0)
   395:         avg_loss, _ = _default_client_sgd(
   396:             model, loader, loss_fn, local_epochs, local_lr, device)
   397:         return model.cpu().state_dict(), len(client_dataset), avg_loss
   398: 
   399:     def aggregate(self, global_state_dict, client_updates, round_num):
   400:         """Aggregate client updates into a new global state_dict.
   401: 
   402:         Default is sample-count-weighted average (FedAvg). Override for
   403:         server-side innovations (momentum, robust median, control-variate
   404:         server update, ...).
   405:         """
   406:         total_samples = sum(max(upd[1], 1) for upd in client_updates)
   407:         new_state = OrderedDict()
   408:         for key, ref in global_state_dict.items():
   409:             if not ref.is_floating_point():
   410:                 new_state[key] = client_updates[0][0][key].detach().clone()
   411:                 continue
   412:             acc = torch.zeros_like(ref, device="cpu", dtype=torch.float32)
   413:             for st, n, _ in client_updates:
   414:                 acc += st[key].detach().cpu().float() * (max(n, 1) / total_samples)
   415:             new_state[key] = acc.to(ref.dtype)
   416:         return new_state
   417: 
   418:     def select_clients(self, num_available, num_to_select, round_num):
   419:         """Pick client indices for this round. Default: uniform random."""
   420:         return random.sample(range(num_available), min(num_to_select, num_available))
   421: 
   422: 
   423: # =====================================================================
   424: # FIXED: FL Simulation Loop
   425: # =====================================================================
   426: def run_fl_simulation(args):
   427:     """Main federated learning simulation."""
   428:     # Seed everything
   429:     random.seed(args.seed)
   430:     np.random.seed(args.seed)
   431:     torch.manual_seed(args.seed)
   432:     torch.backends.cudnn.deterministic = True
   433: 
   434:     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   435:     print(f"Device: {device}", flush=True)
   436: 
   437:     # Prepare data
   438:     print(f"Loading dataset: {args.dataset}", flush=True)
   439:     client_datasets, test_set, model_fn, loss_fn = prepare_data(args)
   440:     print(f"Number of clients: {len(client_datasets)}", flush=True)
   441: 
   442:     # Initialize global model
   443:     global_model = model_fn()
   444:     global_state = copy.deepcopy(global_model.state_dict())
   445: 
   446:     # Initialize FL strategy (owns both client-side and server-side logic)
   447:     strategy = Strategy(global_model, args)
   448: 
   449:     best_accuracy = 0.0
   450:     start_time = time.time()
   451: 
   452:     for round_num in range(args.num_rounds):
   453:         round_start = time.time()
   454: 
   455:         # Client selection
   456:         selected = strategy.select_clients(
   457:             len(client_datasets), args.clients_per_round, round_num)
   458: 
   459:         # Local training (simulated sequentially)
   460:         client_updates = []
   461:         round_loss = 0.0
   462:         for client_idx in selected:
   463:             updated_state, n_samples, avg_loss = strategy.client_local_train(
   464:                 global_state, client_datasets[client_idx],
   465:                 model_fn, loss_fn,
   466:                 args.local_epochs, args.local_lr,
   467:                 args.local_batch_size, device,
   468:                 client_idx)
   469:             client_updates.append((updated_state, n_samples, avg_loss))
   470:             round_loss += avg_loss
   471: 
   472:         avg_round_loss = round_loss / len(selected)
   473: 
   474:         # Server aggregation
   475:         global_state = strategy.aggregate(global_state, client_updates, round_num)
   476: 
   477:         # Log training metrics
   478:         round_time = time.time() - round_start
   479:         if (round_num + 1) % 5 == 0 or round_num == 0:
   480:             print(f"TRAIN_METRICS round={round_num+1} avg_loss={avg_round_loss:.4f} "
   481:                   f"round_time={round_time:.1f}s", flush=True)
   482: 
   483:         # Periodic evaluation
   484:         if (round_num + 1) % args.eval_every == 0 or round_num == args.num_rounds - 1:
   485:             global_model.load_state_dict(global_state)
   486:             test_loss, test_acc = evaluate_global_model(
   487:                 global_model, test_set, loss_fn, device)
   488:             elapsed = time.time() - start_time
   489:             print(f"EVAL round={round_num+1} test_loss={test_loss:.4f} "
   490:                   f"test_accuracy={test_acc:.4f} elapsed={elapsed:.0f}s", flush=True)
   491:             if test_acc > best_accuracy:
   492:                 best_accuracy = test_acc
   493: 
   494:     # Final evaluation
   495:     global_model.load_state_dict(global_state)
   496:     test_loss, test_acc = evaluate_global_model(global_model, test_set, loss_fn, device)
   497:     print(f"TEST_METRICS test_accuracy={test_acc:.4f} test_loss={test_loss:.4f} "
   498:           f"best_accuracy={best_accuracy:.4f}", flush=True)
   499: 
   500:     # Save results

[truncated: showing at most 500 lines / 60000 bytes from flower/custom_fl_aggregation.py]
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `fedavg` baseline — editable region  [READ-ONLY — reference implementation]

In `flower/custom_fl_aggregation.py`:

```python
Lines 340–373:
   337: # =====================================================================
   338: # EDITABLE: FL Strategy — owns BOTH client-side and server-side logic
   339: # =====================================================================
   340: class Strategy:
   341:     """FedAvg — plain SGD + weighted average of client state dicts."""
   342: 
   343:     def __init__(self, global_model, args):
   344:         self.args = args
   345: 
   346:     def client_local_train(self, global_state_dict, client_dataset, model_fn,
   347:                            loss_fn, local_epochs, local_lr, local_batch_size,
   348:                            device, client_idx):
   349:         model = model_fn()
   350:         model.load_state_dict(global_state_dict)
   351:         model.to(device)
   352:         model.train()
   353:         loader = DataLoader(client_dataset, batch_size=local_batch_size,
   354:                             shuffle=True, drop_last=False, num_workers=0)
   355:         avg_loss, _ = _default_client_sgd(
   356:             model, loader, loss_fn, local_epochs, local_lr, device)
   357:         return model.cpu().state_dict(), len(client_dataset), avg_loss
   358: 
   359:     def aggregate(self, global_state_dict, client_updates, round_num):
   360:         total_samples = sum(max(upd[1], 1) for upd in client_updates)
   361:         new_state = OrderedDict()
   362:         for key, ref in global_state_dict.items():
   363:             if not ref.is_floating_point():
   364:                 new_state[key] = client_updates[0][0][key].detach().clone()
   365:                 continue
   366:             acc = torch.zeros_like(ref, device="cpu", dtype=torch.float32)
   367:             for st, n, _ in client_updates:
   368:                 acc += st[key].detach().cpu().float() * (max(n, 1) / total_samples)
   369:             new_state[key] = acc.to(ref.dtype)
   370:         return new_state
   371: 
   372:     def select_clients(self, num_available, num_to_select, round_num):
   373:         return random.sample(range(num_available), min(num_to_select, num_available))
   374: 
   375: 
   376: # =====================================================================
```

### `fedprox` baseline — editable region  [READ-ONLY — reference implementation]

In `flower/custom_fl_aggregation.py`:

```python
Lines 340–391:
   337: # =====================================================================
   338: # EDITABLE: FL Strategy — owns BOTH client-side and server-side logic
   339: # =====================================================================
   340: class Strategy:
   341:     """FedProx — plain SGD + proximal term in the local objective."""
   342: 
   343:     def __init__(self, global_model, args):
   344:         self.args = args
   345:         self.mu = 0.01  # Li et al. 2020 suggested range 0.001-0.1 on CIFAR/FEMNIST.
   346: 
   347:     def client_local_train(self, global_state_dict, client_dataset, model_fn,
   348:                            loss_fn, local_epochs, local_lr, local_batch_size,
   349:                            device, client_idx):
   350:         model = model_fn()
   351:         model.load_state_dict(global_state_dict)
   352:         model.to(device)
   353:         model.train()
   354:         # Freeze copies of the global parameters for the prox term.
   355:         global_params = [
   356:             p.detach().clone() for p in model.parameters() if p.requires_grad
   357:         ]
   358:         mu_half = 0.5 * self.mu
   359: 
   360:         def prox_loss(m):
   361:             prox = 0.0
   362:             for w, w0 in zip(
   363:                 [p for p in m.parameters() if p.requires_grad],
   364:                 global_params,
   365:             ):
   366:                 prox = prox + (w - w0).pow(2).sum()
   367:             return mu_half * prox
   368: 
   369:         loader = DataLoader(client_dataset, batch_size=local_batch_size,
   370:                             shuffle=True, drop_last=False, num_workers=0)
   371:         avg_loss, _ = _default_client_sgd(
   372:             model, loader, loss_fn, local_epochs, local_lr, device,
   373:             loss_aug=prox_loss,
   374:         )
   375:         return model.cpu().state_dict(), len(client_dataset), avg_loss
   376: 
   377:     def aggregate(self, global_state_dict, client_updates, round_num):
   378:         total_samples = sum(max(upd[1], 1) for upd in client_updates)
   379:         new_state = OrderedDict()
   380:         for key, ref in global_state_dict.items():
   381:             if not ref.is_floating_point():
   382:                 new_state[key] = client_updates[0][0][key].detach().clone()
   383:                 continue
   384:             acc = torch.zeros_like(ref, device="cpu", dtype=torch.float32)
   385:             for st, n, _ in client_updates:
   386:                 acc += st[key].detach().cpu().float() * (max(n, 1) / total_samples)
   387:             new_state[key] = acc.to(ref.dtype)
   388:         return new_state
   389: 
   390:     def select_clients(self, num_available, num_to_select, round_num):
   391:         return random.sample(range(num_available), min(num_to_select, num_available))
   392: 
   393: 
   394: # =====================================================================
```

### `scaffold` baseline — editable region  [READ-ONLY — reference implementation]

In `flower/custom_fl_aggregation.py`:

```python
Lines 340–509:
   337: # =====================================================================
   338: # EDITABLE: FL Strategy — owns BOTH client-side and server-side logic
   339: # =====================================================================
   340: class Strategy:
   341:     """SCAFFOLD — Alg 1 with Option-II control-variate update."""
   342: 
   343:     def __init__(self, global_model, args):
   344:         self.args = args
   345:         self.num_clients = args.num_clients
   346:         self.global_control = OrderedDict(
   347:             (k, torch.zeros_like(v, device="cpu"))
   348:             for k, v in global_model.state_dict().items()
   349:         )
   350:         self.client_controls = {}   # client_idx -> OrderedDict (CPU)
   351: 
   352:     def _zero_like_state(self, state_dict):
   353:         return OrderedDict(
   354:             (k, torch.zeros_like(v, device="cpu")) for k, v in state_dict.items()
   355:         )
   356: 
   357:     def _get_client_control(self, client_idx, reference_state):
   358:         c_i = self.client_controls.get(client_idx)
   359:         if c_i is None:
   360:             c_i = self._zero_like_state(reference_state)
   361:             self.client_controls[client_idx] = c_i
   362:         return c_i
   363: 
   364:     def _ensure_global_control_on(self, device, reference_state):
   365:         # Lazily move global_control to the model's device once and keep it there.
   366:         if (not hasattr(self, "_gc_dev")) or self._gc_dev_id != id(device):
   367:             self.global_control = OrderedDict(
   368:                 (k, v.to(device) if v.is_floating_point() else v)
   369:                 for k, v in self.global_control.items()
   370:             )
   371:             self._gc_dev = device
   372:             self._gc_dev_id = id(device)
   373: 
   374:     def _get_client_control_on(self, client_idx, reference_state, device):
   375:         c_i = self.client_controls.get(client_idx)
   376:         if c_i is None:
   377:             c_i = OrderedDict(
   378:                 (k, torch.zeros_like(v, device=device) if v.is_floating_point()
   379:                  else torch.zeros_like(v, device="cpu"))
   380:                 for k, v in reference_state.items()
   381:             )
   382:             self.client_controls[client_idx] = c_i
   383:         elif any(v.device != device for v in c_i.values() if v.is_floating_point()):
   384:             c_i = OrderedDict(
   385:                 (k, v.to(device) if v.is_floating_point() else v)
   386:                 for k, v in c_i.items()
   387:             )
   388:             self.client_controls[client_idx] = c_i
   389:         return c_i
   390: 
   391:     def client_local_train(self, global_state_dict, client_dataset, model_fn,
   392:                            loss_fn, local_epochs, local_lr, local_batch_size,
   393:                            device, client_idx):
   394:         model = model_fn()
   395:         model.load_state_dict(global_state_dict)
   396:         model.to(device)
   397:         model.train()
   398: 
   399:         # Move global_control + c_i to device once; keep them resident.
   400:         self._ensure_global_control_on(device, model.state_dict())
   401:         c_i = self._get_client_control_on(client_idx, model.state_dict(), device)
   402: 
   403:         # Snapshot global params x ON DEVICE for Option-II later.
   404:         x_dev = OrderedDict(
   405:             (n, p.detach().clone())
   406:             for n, p in model.named_parameters()
   407:             if n in self.global_control
   408:         )
   409: 
   410:         # Pre-compute (c - c_i) on device once per client.
   411:         correction_dev = {}
   412:         for name, p in model.named_parameters():
   413:             if name in self.global_control:
   414:                 correction_dev[id(p)] = self.global_control[name] - c_i[name]
   415: 
   416:         optimizer = optim.SGD(model.parameters(), lr=local_lr)  # plain SGD
   417:         loader = DataLoader(client_dataset, batch_size=local_batch_size,
   418:                             shuffle=True, drop_last=False, num_workers=0)
   419: 
   420:         total_loss, total_samples, local_steps = 0.0, 0, 0
   421:         for _ in range(local_epochs):
   422:             for batch_data in loader:
   423:                 if len(batch_data) != 2:
   424:                     continue
   425:                 inputs, targets = batch_data
   426:                 inputs, targets = inputs.to(device), targets.to(device)
   427:                 optimizer.zero_grad()
   428:                 outputs = model(inputs)
   429:                 if outputs.dim() == 3:
   430:                     outputs = outputs.view(-1, outputs.size(-1))
   431:                     targets = targets.view(-1)
   432:                 loss = loss_fn(outputs, targets)
   433:                 loss.backward()
   434:                 # Corrected gradient: g + (c - c_i). Pure in-place add on device.
   435:                 for p in model.parameters():
   436:                     if p.grad is None:
   437:                         continue
   438:                     corr = correction_dev.get(id(p))
   439:                     if corr is not None:
   440:                         p.grad.add_(corr)
   441:                 optimizer.step()
   442:                 local_steps += 1
   443:                 total_loss += loss.item() * inputs.size(0)
   444:                 total_samples += inputs.size(0)
   445: 
   446:         # Option-II update — stay on device.
   447:         if local_steps > 0 and local_lr > 0.0:
   448:             denom = local_steps * local_lr
   449:             new_ci = OrderedDict()
   450:             delta_c = OrderedDict()
   451:             for name, p in model.named_parameters():
   452:                 if name not in self.global_control:
   453:                     continue
   454:                 # c_i+ = c_i - c + (x - y) / (K * eta)
   455:                 update = c_i[name] - self.global_control[name] + (x_dev[name] - p.detach()) / denom
   456:                 delta_c[name] = (update - c_i[name]).clone()
   457:                 new_ci[name] = update
   458:             # Carry over non-FP buffer keys unchanged from existing c_i.
   459:             for k, v in c_i.items():
   460:                 if k not in new_ci:
   461:                     new_ci[k] = v
   462:                     delta_c[k] = torch.zeros_like(v)
   463:             self._pending_delta_c = getattr(self, "_pending_delta_c", {})
   464:             self._pending_delta_c[client_idx] = delta_c
   465:             self.client_controls[client_idx] = new_ci
   466: 
   467:         # Single GPU→CPU transfer for the returned state_dict (server aggregates on CPU).
   468:         final_state = OrderedDict(
   469:             (k, v.detach().cpu()) for k, v in model.state_dict().items()
   470:         )
   471:         avg_loss = total_loss / max(total_samples, 1)
   472:         return final_state, len(client_dataset), avg_loss
   473: 
   474:     def aggregate(self, global_state_dict, client_updates, round_num):
   475:         # FedAvg-style weighted model average.
   476:         total_samples = sum(max(upd[1], 1) for upd in client_updates)
   477:         new_state = OrderedDict()
   478:         for key, ref in global_state_dict.items():
   479:             if not ref.is_floating_point():
   480:                 new_state[key] = client_updates[0][0][key].detach().clone()
   481:                 continue
   482:             acc = torch.zeros_like(ref, device="cpu", dtype=torch.float32)
   483:             for st, n, _ in client_updates:
   484:                 acc += st[key].detach().cpu().float() * (max(n, 1) / total_samples)
   485:             new_state[key] = acc.to(ref.dtype)
   486: 
   487:         # Server-side global c update: c <- c + (|S|/N) * mean_i Δc_i.
   488:         # δc_i are on the same device as global_control — stay there.
   489:         deltas = getattr(self, "_pending_delta_c", {})
   490:         if deltas:
   491:             weight = len(client_updates) / max(self.num_clients, 1)
   492:             n_updates = len(deltas)
   493:             for key in self.global_control:
   494:                 if not self.global_control[key].is_floating_point():
   495:                     continue
   496:                 acc = None
   497:                 for dc in deltas.values():
   498:                     if key in dc and dc[key].is_floating_point():
   499:                         contrib = dc[key].to(self.global_control[key].device)
   500:                         acc = contrib.clone() if acc is None else acc + contrib
   501:                 if acc is not None:
   502:                     self.global_control[key] = (
   503:                         self.global_control[key] + (weight / n_updates) * acc
   504:                     )
   505:             self._pending_delta_c = {}
   506:         return new_state
   507: 
   508:     def select_clients(self, num_available, num_to_select, round_num):
   509:         return random.sample(range(num_available), min(num_to_select, num_available))
   510: 
   511: 
   512: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
