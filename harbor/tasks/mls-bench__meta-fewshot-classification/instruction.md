# MLS-Bench: meta-fewshot-classification

# Meta-Learning: Few-Shot Image Classification

## Research Question
Design a novel few-shot image classifier that, given a small support set of N classes with K labeled examples each, generalizes to query examples of those classes. The contribution should be a reusable algorithmic component (a way of summarizing the support set, comparing query to support, or doing task-level adaptation), not a dataset-specific trick.

## Background
Few-shot classification recognizes new classes from a handful of labeled examples. Episodic evaluation samples N-way K-shot tasks: K support examples per class, then unlabeled query images to classify into one of the N classes. Common design axes:
- **Feature comparison**: Euclidean distance, cosine similarity, learned metric.
- **Support encoding**: per-class prototypes, attention, graph neural networks.
- **Query adaptation**: cross-attention, transductive inference, LSTM context.

Reference baselines (provided as read-only modules):
- **Prototypical Networks** — Snell, Swersky, Zemel, NeurIPS 2017 ([arXiv:1703.05175](https://arxiv.org/abs/1703.05175)). Class prototype = mean embedding of support; query classified by negative squared Euclidean distance.
- **Matching Networks** — Vinyals, Blundell, Lillicrap, Kavukcuoglu, Wierstra, NeurIPS 2016 ([arXiv:1606.04080](https://arxiv.org/abs/1606.04080)). Cosine attention over support embeddings; weighted-sum label prediction (no fine-tuning at test time).
- **Relation Networks** — Sung, Yang, Zhang, Xiang, Torr, Hospedales, CVPR 2018 ([arXiv:1711.06025](https://arxiv.org/abs/1711.06025)). A learned MLP scores the relation between query feature and class prototype, replacing fixed metrics.

## Model Interface
Implement `CustomFewShotMethod` in `custom_fewshot.py`:
```python
class CustomFewShotMethod(FewShotClassifier):
    def __init__(self):
        backbone = make_backbone(use_pooling=True)  # ResNet-12, 640-dim features
        super().__init__(backbone=backbone)

    def process_support_set(self, support_images, support_labels):
        # Extract and store support set information for forward()
        ...

    def forward(self, query_images) -> Tensor:
        # Return classification scores of shape (n_query, n_way)
        ...

    def compute_loss(self, scores, labels) -> Tensor:
        # Default: cross-entropy
        ...
```

## Available Utilities
- `self.compute_features(images)` — pass through `self.backbone`.
- `self.l2_distance_to_prototypes(features)` — negative Euclidean distance to `self.prototypes`.
- `self.cosine_distance_to_prototypes(features)` — cosine similarity to `self.prototypes`.
- `compute_prototypes(features, labels)` — mean feature per class.
- `self.compute_prototypes_and_store_support_set(images, labels)` — convenience method.
- `make_backbone(use_pooling=True/False)` — ResNet-12 with 640-dim feature vector or feature maps.

## Fixed Training & Evaluation Pipeline
- Backbone: ResNet-12 (640-dim).
- Episodic training: 500 tasks/epoch for 200 epochs, 5-way 5-shot tasks.
- Evaluation: episodic classification accuracy on held-out test splits across multiple few-shot image recognition benchmarks.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/easy-few-shot-learning/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `easy-few-shot-learning/custom_fewshot.py`
- editable lines **225–286**


Other files you may **read** for context (do not modify):
- `easy-few-shot-learning/easyfsl/methods/few_shot_classifier.py`
- `easy-few-shot-learning/easyfsl/methods/utils.py`


## Readable Context


### `easy-few-shot-learning/custom_fewshot.py`  [EDITABLE — lines 225–286 only]

```python
     1: # Custom few-shot classification method for MLS-Bench
     2: #
     3: # EDITABLE section: CustomFewShotMethod class and helper modules.
     4: # FIXED sections: everything else (config, data loading, training loop, evaluation).
     5: import os
     6: import copy
     7: import random
     8: import json
     9: from pathlib import Path
    10: from statistics import mean
    11: from typing import List, Tuple, Optional
    12: 
    13: import numpy as np
    14: import torch
    15: import torch.nn as nn
    16: import torch.nn.functional as F
    17: from torch import Tensor
    18: from torch.utils.data import DataLoader
    19: from torch.optim import SGD, Adam
    20: from torch.optim.lr_scheduler import MultiStepLR
    21: from torchvision import transforms
    22: from tqdm import tqdm
    23: 
    24: from easyfsl.datasets import FewShotDataset
    25: from easyfsl.samplers import TaskSampler
    26: from easyfsl.methods import FewShotClassifier
    27: from easyfsl.methods.utils import compute_prototypes
    28: 
    29: 
    30: # =====================================================================
    31: # FIXED: Configuration
    32: # =====================================================================
    33: SEED = int(os.environ.get("SEED", "42"))
    34: OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
    35: DATASET_NAME = os.environ.get("ENV", "mini_imagenet")
    36: 
    37: # Few-shot settings
    38: N_WAY = 5
    39: N_SHOT = 5
    40: N_QUERY = 15
    41: IMAGE_SIZE = 84
    42: 
    43: # Training settings
    44: N_EPOCHS = 200
    45: N_TASKS_PER_EPOCH = 500
    46: N_VALIDATION_TASKS = 200
    47: N_TEST_TASKS = 600
    48: LEARNING_RATE = 1e-2
    49: SCHEDULER_MILESTONES = [120, 160]
    50: SCHEDULER_GAMMA = 0.1
    51: WEIGHT_DECAY = 5e-4
    52: GRAD_CLIP_NORM = 5.0  # Vinyals et al. 2016 Sec 3.1 (Matching Networks)
    53: 
    54: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    55: N_WORKERS = 4
    56: 
    57: IMAGENET_NORMALIZATION = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    58: 
    59: 
    60: # =====================================================================
    61: # FIXED: Dataset loading
    62: # =====================================================================
    63: class ImageFolderFewShot(FewShotDataset):
    64:     """A general-purpose few-shot dataset that loads images from class-organized directories."""
    65: 
    66:     def __init__(
    67:         self,
    68:         specs_file: str,
    69:         image_size: int = 84,
    70:         training: bool = False,
    71:     ):
    72:         specs = self._load_specs(specs_file)
    73:         self.images: List[str] = []
    74:         self.labels: List[int] = []
    75: 
    76:         supported_formats = {".bmp", ".png", ".jpeg", ".jpg", ".JPEG"}
    77:         for class_id, class_root in enumerate(specs["class_roots"]):
    78:             class_images = sorted(
    79:                 str(p) for p in Path(class_root).glob("*")
    80:                 if p.is_file() and p.suffix in supported_formats
    81:             )
    82:             self.images.extend(class_images)
    83:             self.labels.extend([class_id] * len(class_images))
    84: 
    85:         self.transform = (
    86:             transforms.Compose([
    87:                 transforms.RandomResizedCrop(image_size),
    88:                 transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    89:                 transforms.RandomHorizontalFlip(),
    90:                 transforms.ToTensor(),
    91:                 transforms.Normalize(**IMAGENET_NORMALIZATION),
    92:             ])
    93:             if training
    94:             else transforms.Compose([
    95:                 transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
    96:                 transforms.CenterCrop(image_size),
    97:                 transforms.ToTensor(),
    98:                 transforms.Normalize(**IMAGENET_NORMALIZATION),
    99:             ])
   100:         )
   101: 
   102:     @staticmethod
   103:     def _load_specs(specs_file: str) -> dict:
   104:         ext = Path(specs_file).suffix
   105:         if ext == ".json":
   106:             with open(specs_file, "r") as f:
   107:                 return json.load(f)
   108:         elif ext == ".csv":
   109:             # miniImageNet CSV format: class_name,image_name
   110:             # Images are expected at: <specs_dir>/../images/<class_name>/ via symlink
   111:             import pandas as pd
   112:             df = pd.read_csv(specs_file)
   113:             specs_dir = str(Path(specs_file).parent)  # e.g. .../data/mini_imagenet
   114:             class_names = df["class_name"].unique().tolist()
   115:             class_roots = [os.path.join(specs_dir, "images", cn) for cn in class_names]
   116:             return {"class_names": class_names, "class_roots": class_roots}
   117:         else:
   118:             raise ValueError(f"Unsupported spec file format: {ext}")
   119: 
   120:     def __getitem__(self, item):
   121:         from PIL import Image
   122:         img = self.transform(Image.open(self.images[item]).convert("RGB"))
   123:         return img, self.labels[item]
   124: 
   125:     def __len__(self):
   126:         return len(self.labels)
   127: 
   128:     def get_labels(self) -> List[int]:
   129:         return self.labels
   130: 
   131: 
   132: def get_dataset_specs(dataset_name: str):
   133:     """Return (train_spec, val_spec, test_spec) file paths for the given dataset."""
   134:     _pkg_dir = os.environ.get("MLSBENCH_PKG_DIR", "/workspace/easy-few-shot-learning")
   135:     specs_dir = f"{_pkg_dir}/data/{dataset_name}"
   136: 
   137:     if dataset_name == "mini_imagenet":
   138:         # miniImageNet uses CSV spec files
   139:         return (
   140:             f"{specs_dir}/train.csv",
   141:             f"{specs_dir}/val.csv",
   142:             f"{specs_dir}/test.csv",
   143:         )
   144:     else:
   145:         # CUB, cifar_fs, tiered_imagenet use JSON spec files
   146:         return (
   147:             f"{specs_dir}/train.json",
   148:             f"{specs_dir}/val.json",
   149:             f"{specs_dir}/test.json",
   150:         )
   151: 
   152: 
   153: def make_data_loader(specs_file: str, n_way: int, n_shot: int, n_query: int,
   154:                      n_tasks: int, training: bool) -> DataLoader:
   155:     dataset = ImageFolderFewShot(specs_file, image_size=IMAGE_SIZE, training=training)
   156:     sampler = TaskSampler(dataset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks)
   157:     return DataLoader(
   158:         dataset,
   159:         batch_sampler=sampler,
   160:         num_workers=N_WORKERS,
   161:         pin_memory=True,
   162:         collate_fn=sampler.episodic_collate_fn,
   163:     )
   164: 
   165: 
   166: # =====================================================================
   167: # FIXED: ResNet backbone (shared by all methods)
   168: # =====================================================================
   169: from easyfsl.modules import resnet12
   170: 
   171: 
   172: def make_backbone(use_pooling: bool = True) -> nn.Module:
   173:     """Create a ResNet-12 backbone.
   174: 
   175:     Args:
   176:         use_pooling: if True, output is [B, 640] feature vectors.
   177:                      if False, output is [B, 640, H, W] feature maps.
   178:     """
   179:     return resnet12(use_pooling=use_pooling)
   180: 
   181: 
   182: FEATURE_DIMENSION = 640  # ResNet12 output dimension
   183: 
   184: 
   185: # =====================================================================
   186: # FIXED: Training utilities
   187: # =====================================================================
   188: def training_epoch(model, data_loader, optimizer):
   189:     all_loss = []
   190:     model.train()
   191:     for support_images, support_labels, query_images, query_labels, _ in data_loader:
   192:         optimizer.zero_grad()
   193:         model.process_support_set(
   194:             support_images.to(DEVICE), support_labels.to(DEVICE)
   195:         )
   196:         classification_scores = model(query_images.to(DEVICE))
   197:         loss = model.compute_loss(classification_scores, query_labels.to(DEVICE))
   198:         loss.backward()
   199:         # Vinyals et al. 2016 (Matching Networks) Sec 3.1: dampen gradients
   200:         # with norm > 5. Generally beneficial for LSTM/recurrent components.
   201:         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
   202:         optimizer.step()
   203:         all_loss.append(loss.item())
   204:     return mean(all_loss)
   205: 
   206: 
   207: def evaluate(model, data_loader):
   208:     total_predictions = 0
   209:     correct_predictions = 0
   210:     model.eval()
   211:     with torch.no_grad():
   212:         for support_images, support_labels, query_images, query_labels, _ in data_loader:
   213:             model.process_support_set(
   214:                 support_images.to(DEVICE), support_labels.to(DEVICE)
   215:             )
   216:             predictions = model(query_images.to(DEVICE)).detach().data
   217:             correct = int((torch.max(predictions, 1)[1] == query_labels.to(DEVICE)).sum().item())
   218:             correct_predictions += correct
   219:             total_predictions += len(query_labels)
   220:     return correct_predictions / total_predictions if total_predictions > 0 else 0.0
   221: 
   222: 
   223: # =====================================================================
   224: # EDITABLE: Custom Few-Shot Classification Method
   225: # =====================================================================
   226: class CustomFewShotMethod(FewShotClassifier):
   227:     """Custom few-shot classification method.
   228: 
   229:     This class defines how to classify query images given a support set.
   230:     You MUST implement:
   231:         - __init__: create the backbone and any learnable modules
   232:         - process_support_set(support_images, support_labels): extract and store
   233:           information from the support set for later query classification
   234:         - forward(query_images) -> Tensor of shape (n_query, n_way): predict
   235:           classification scores for query images
   236: 
   237:     Available utilities (from easyfsl):
   238:         - self.compute_features(images): pass images through self.backbone
   239:         - self.l2_distance_to_prototypes(features): compute negative L2 distance to self.prototypes
   240:         - self.cosine_distance_to_prototypes(features): compute cosine similarity to self.prototypes
   241:         - self.softmax_if_specified(scores): apply softmax if self.use_softmax is set
   242:         - compute_prototypes(features, labels): compute class prototypes (mean features)
   243: 
   244:     The backbone should be set as self.backbone (an nn.Module).
   245:     Feature dimension of the ResNet12 backbone is 640.
   246: 
   247:     The training loop calls model.compute_loss(scores, labels) for flexibility.
   248:     Override compute_loss if your method needs a different loss (e.g., MSE for RelationNet).
   249:     """
   250: 
   251:     def __init__(self):
   252:         backbone = make_backbone(use_pooling=True)
   253:         super().__init__(backbone=backbone, use_softmax=False)
   254: 
   255:     def process_support_set(self, support_images: Tensor, support_labels: Tensor):
   256:         """Extract and store support set information."""
   257:         self.compute_prototypes_and_store_support_set(support_images, support_labels)
   258: 
   259:     def forward(self, query_images: Tensor) -> Tensor:
   260:         """Predict classification scores for query images.
   261: 
   262:         Args:
   263:             query_images: images of shape (n_query, 3, 84, 84)
   264: 
   265:         Returns:
   266:             scores of shape (n_query, n_way) — higher means more likely
   267:         """
   268:         query_features = self.compute_features(query_images)
   269:         scores = self.l2_distance_to_prototypes(query_features)
   270:         return self.softmax_if_specified(scores)
   271: 
   272:     @staticmethod
   273:     def is_transductive() -> bool:
   274:         return False
   275: 
   276:     def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
   277:         """Compute the training loss. Override for custom loss functions.
   278: 
   279:         Args:
   280:             scores: classification scores of shape (n_query, n_way)
   281:             labels: ground truth labels of shape (n_query,), integers in [0, n_way)
   282: 
   283:         Returns:
   284:             scalar loss tensor
   285:         """
   286:         return F.cross_entropy(scores, labels)
   287: 
   288: 
   289: # =====================================================================
   290: # FIXED: Main training and evaluation script
   291: # =====================================================================
   292: if __name__ == "__main__":
   293:     # Reproducibility
   294:     random.seed(SEED)
   295:     np.random.seed(SEED)
   296:     torch.manual_seed(SEED)
   297:     torch.cuda.manual_seed_all(SEED)
   298:     torch.backends.cudnn.deterministic = True
   299:     torch.backends.cudnn.benchmark = False
   300: 
   301:     os.makedirs(OUTPUT_DIR, exist_ok=True)
   302: 
   303:     print(f"Dataset: {DATASET_NAME}, Seed: {SEED}", flush=True)
   304:     print(f"Few-shot setting: {N_WAY}-way {N_SHOT}-shot {N_QUERY}-query", flush=True)
   305: 
   306:     # Load data
   307:     train_spec, val_spec, test_spec = get_dataset_specs(DATASET_NAME)
   308:     train_loader = make_data_loader(train_spec, N_WAY, N_SHOT, N_QUERY, N_TASKS_PER_EPOCH, training=True)
   309:     val_loader = make_data_loader(val_spec, N_WAY, N_SHOT, N_QUERY, N_VALIDATION_TASKS, training=False)
   310:     test_loader = make_data_loader(test_spec, N_WAY, N_SHOT, N_QUERY, N_TEST_TASKS, training=False)
   311: 
   312:     # Build model
   313:     model = CustomFewShotMethod().to(DEVICE)
   314: 
   315:     # ── FIXED: Parameter count check ────────────────────────────────
   316:     # Budget based on 1.05x largest baseline (RelationNet).
   317:     # RelationNet: ResNet12 backbone (no pooling) + RelationModule
   318:     # ResNet12 backbone ~ 8.7M params (standard for few-shot)
   319:     # RelationModule: Conv2d(2*640,640,3)+BN + Conv2d(640,640,3)+BN + Linear(640,8) + Linear(8,1)
   320:     _backbone_params = 8_700_000  # ResNet12 backbone approximate
   321:     _relation_module = 2 * FEATURE_DIMENSION * FEATURE_DIMENSION * 9 + FEATURE_DIMENSION  # conv1
   322:     _relation_module += FEATURE_DIMENSION * 2  # BN1
   323:     _relation_module += FEATURE_DIMENSION * FEATURE_DIMENSION * 9 + FEATURE_DIMENSION  # conv2
   324:     _relation_module += FEATURE_DIMENSION * 2  # BN2
   325:     _relation_module += FEATURE_DIMENSION * 8 + 8 + 8 * 1 + 1  # FC layers
   326:     _budget = int((_backbone_params + _relation_module + 5000) * 1.05)
   327:     _total_params = sum(p.numel() for p in model.parameters())
   328:     print(f"Total params: {_total_params:,} (budget: {_budget:,})")
   329: 
   330:     # Optimizer and scheduler — methods may override the LR via class attr LR_OVERRIDE.
   331:     # TODO: Chen et al. 2019 (https://arxiv.org/abs/1904.04232, Sec. 5)
   332:     # trains meta-learning methods with Adam@1e-3 and scales MatchingNet cosine
   333:     # similarities. Validate Adam/scalar ablations before comparing to that order.
   334:     _lr = getattr(model, "LR_OVERRIDE", LEARNING_RATE)
   335:     if _lr != LEARNING_RATE:
   336:         print(f"  LR override active: {_lr} (default {LEARNING_RATE})", flush=True)
   337:     optimizer = SGD(model.parameters(), lr=_lr, momentum=0.9, weight_decay=WEIGHT_DECAY)
   338:     scheduler = MultiStepLR(optimizer, milestones=SCHEDULER_MILESTONES, gamma=SCHEDULER_GAMMA)
   339: 
   340:     # Training loop
   341:     best_state = model.state_dict()
   342:     best_val_acc = 0.0
   343: 
   344:     for epoch in range(N_EPOCHS):
   345:         avg_loss = training_epoch(model, train_loader, optimizer)
   346:         val_acc = evaluate(model, val_loader)
   347:         scheduler.step()
   348: 
   349:         print(f"TRAIN_METRICS epoch={epoch} train_loss={avg_loss:.5f} val_acc={val_acc:.4f}", flush=True)
   350: 
   351:         if val_acc > best_val_acc:
   352:             best_val_acc = val_acc
   353:             best_state = copy.deepcopy(model.state_dict())
   354:             print(f"  New best val accuracy: {val_acc:.4f}", flush=True)
   355: 
   356:     # Load best model and evaluate on test set
   357:     model.load_state_dict(best_state)
   358:     test_acc = evaluate(model, test_loader)
   359:     print(f"TEST_METRICS accuracy={test_acc:.4f}", flush=True)
   360:     print(f"Test accuracy: {100 * test_acc:.2f}%", flush=True)
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `protonet` baseline — editable region  [READ-ONLY — reference implementation]

In `easy-few-shot-learning/custom_fewshot.py`:

```python
Lines 225–250:
   222: 
   223: # =====================================================================
   224: # EDITABLE: Custom Few-Shot Classification Method
   225: class CustomFewShotMethod(FewShotClassifier):
   226:     """Prototypical Networks (Snell et al., 2017).
   227: 
   228:     Compute class prototypes as the mean feature vector of support examples,
   229:     then classify queries by negative Euclidean distance to prototypes.
   230:     """
   231: 
   232:     def __init__(self):
   233:         backbone = make_backbone(use_pooling=True)
   234:         super().__init__(backbone=backbone, use_softmax=False)
   235: 
   236:     def process_support_set(self, support_images: Tensor, support_labels: Tensor):
   237:         self.compute_prototypes_and_store_support_set(support_images, support_labels)
   238: 
   239:     def forward(self, query_images: Tensor) -> Tensor:
   240:         query_features = self.compute_features(query_images)
   241:         scores = self.l2_distance_to_prototypes(query_features)
   242:         return self.softmax_if_specified(scores)
   243: 
   244:     @staticmethod
   245:     def is_transductive() -> bool:
   246:         return False
   247: 
   248:     def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
   249:         return F.cross_entropy(scores, labels)
   250: 
   251: 
   252: 
   253: # =====================================================================
```

### `matchingnet` baseline — editable region  [READ-ONLY — reference implementation]

In `easy-few-shot-learning/custom_fewshot.py`:

```python
Lines 225–315:
   222: 
   223: # =====================================================================
   224: # EDITABLE: Custom Few-Shot Classification Method
   225: class CustomFewShotMethod(FewShotClassifier):
   226:     """Matching Networks (Vinyals et al., 2016).
   227: 
   228:     Contextualizes support and query features using LSTMs, then classifies
   229:     queries via cosine-similarity-weighted voting over support labels.
   230:     Uses NLLLoss since output is log-probabilities.
   231:     """
   232: 
   233:     # Vinyals et al. 2016 trains MatchingNet with Adam@1e-3 and gradient
   234:     # clipping at 5; SGD@1e-2 (the global default for ProtoNet/RelationNet)
   235:     # destabilizes the bidirectional LSTM and the 25-step query encoder loop,
   236:     # collapsing softmax to uniform output. The framework's training loop
   237:     # honours LR_OVERRIDE to keep this baseline trainable.
   238:     LR_OVERRIDE = 1e-3
   239: 
   240:     def __init__(self):
   241:         backbone = make_backbone(use_pooling=True)
   242:         super().__init__(backbone=backbone, use_softmax=False)
   243:         self.feature_dimension = FEATURE_DIMENSION
   244: 
   245:         # Bidirectional LSTM to contextualize support features
   246:         self.support_features_encoder = nn.LSTM(
   247:             input_size=self.feature_dimension,
   248:             hidden_size=self.feature_dimension,
   249:             num_layers=1,
   250:             batch_first=True,
   251:             bidirectional=True,
   252:         )
   253:         # LSTM cell for attention-based query encoding
   254:         self.query_features_encoding_cell = nn.LSTMCell(
   255:             self.feature_dimension * 2, self.feature_dimension
   256:         )
   257:         self.softmax = nn.Softmax(dim=1)
   258: 
   259:         self.contextualized_support_features = torch.tensor(())
   260:         self.one_hot_support_labels = torch.tensor(())
   261: 
   262:     def process_support_set(self, support_images: Tensor, support_labels: Tensor):
   263:         support_features = self.compute_features(support_images)
   264:         self.contextualized_support_features = self._encode_support(support_features)
   265:         self.one_hot_support_labels = F.one_hot(support_labels).float()
   266: 
   267:     def forward(self, query_images: Tensor) -> Tensor:
   268:         query_features = self.compute_features(query_images)
   269:         contextualized_query_features = self._encode_query(query_features)
   270: 
   271:         similarity_matrix = self.softmax(
   272:             contextualized_query_features.mm(
   273:                 F.normalize(self.contextualized_support_features, dim=1).T
   274:             )
   275:         )
   276:         log_probabilities = (
   277:             similarity_matrix.mm(self.one_hot_support_labels) + 1e-6
   278:         ).log()
   279:         return self.softmax_if_specified(log_probabilities)
   280: 
   281:     def _encode_support(self, support_features: Tensor) -> Tensor:
   282:         hidden_state = self.support_features_encoder(
   283:             support_features.unsqueeze(0)
   284:         )[0].squeeze(0)
   285:         contextualized = (
   286:             support_features
   287:             + hidden_state[:, : self.feature_dimension]
   288:             + hidden_state[:, self.feature_dimension :]
   289:         )
   290:         return contextualized
   291: 
   292:     def _encode_query(self, query_features: Tensor) -> Tensor:
   293:         hidden_state = query_features
   294:         cell_state = torch.zeros_like(query_features)
   295: 
   296:         for _ in range(len(self.contextualized_support_features)):
   297:             attention = self.softmax(
   298:                 hidden_state.mm(self.contextualized_support_features.T)
   299:             )
   300:             read_out = attention.mm(self.contextualized_support_features)
   301:             lstm_input = torch.cat((query_features, read_out), 1)
   302:             hidden_state, cell_state = self.query_features_encoding_cell(
   303:                 lstm_input, (hidden_state, cell_state)
   304:             )
   305:             hidden_state = hidden_state + query_features
   306: 
   307:         return hidden_state
   308: 
   309:     @staticmethod
   310:     def is_transductive() -> bool:
   311:         return False
   312: 
   313:     def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
   314:         return F.nll_loss(scores, labels)
   315: 
   316: 
   317: 
   318: # =====================================================================
```

### `relationnet` baseline — editable region  [READ-ONLY — reference implementation]

In `easy-few-shot-learning/custom_fewshot.py`:

```python
Lines 225–306:
   222: 
   223: # =====================================================================
   224: # EDITABLE: Custom Few-Shot Classification Method
   225: class _RelationModule(nn.Module):
   226:     """CNN relation module from Sung et al. (2018)."""
   227: 
   228:     def __init__(self, feature_dimension: int, inner_channels: int = 8):
   229:         super().__init__()
   230:         self.module = nn.Sequential(
   231:             nn.Sequential(
   232:                 nn.Conv2d(feature_dimension * 2, feature_dimension, kernel_size=3, padding=1),
   233:                 nn.BatchNorm2d(feature_dimension, momentum=1, affine=True),
   234:                 nn.ReLU(),
   235:                 nn.AdaptiveMaxPool2d((5, 5)),
   236:             ),
   237:             nn.Sequential(
   238:                 nn.Conv2d(feature_dimension, feature_dimension, kernel_size=3, padding=0),
   239:                 nn.BatchNorm2d(feature_dimension, momentum=1, affine=True),
   240:                 nn.ReLU(),
   241:                 nn.AdaptiveMaxPool2d((1, 1)),
   242:             ),
   243:             nn.Flatten(),
   244:             nn.Linear(feature_dimension, inner_channels),
   245:             nn.ReLU(),
   246:             nn.Linear(inner_channels, 1),
   247:             nn.Sigmoid(),
   248:         )
   249: 
   250:     def forward(self, x):
   251:         return self.module(x)
   252: 
   253: 
   254: class CustomFewShotMethod(FewShotClassifier):
   255:     """Relation Networks (Sung et al., 2018).
   256: 
   257:     Extracts feature maps (not pooled vectors) from support and query images.
   258:     Computes class prototypes as mean feature maps, concatenates each query-prototype
   259:     pair, and feeds them through a learned relation module to get relation scores.
   260:     Uses MSE loss since output represents relation scores in [0, 1].
   261:     """
   262: 
   263:     def __init__(self):
   264:         backbone = make_backbone(use_pooling=False)  # Need feature maps, not vectors
   265:         super().__init__(backbone=backbone, use_softmax=False)
   266:         self.feature_dimension = FEATURE_DIMENSION
   267:         self.relation_module = _RelationModule(self.feature_dimension)
   268: 
   269:     def process_support_set(self, support_images: Tensor, support_labels: Tensor):
   270:         support_features = self.compute_features(support_images)
   271:         n_way = len(torch.unique(support_labels))
   272:         self.prototypes = torch.cat(
   273:             [
   274:                 support_features[support_labels == label].mean(0, keepdim=True)
   275:                 for label in range(n_way)
   276:             ]
   277:         )
   278: 
   279:     def forward(self, query_images: Tensor) -> Tensor:
   280:         query_features = self.compute_features(query_images)
   281:         n_queries = query_features.shape[0]
   282:         n_prototypes = self.prototypes.shape[0]
   283: 
   284:         # Build pairs: [n_queries * n_prototypes, 2 * C, H, W]
   285:         query_prototype_pairs = torch.cat(
   286:             (
   287:                 self.prototypes.unsqueeze(0).expand(n_queries, -1, -1, -1, -1),
   288:                 query_features.unsqueeze(1).expand(-1, n_prototypes, -1, -1, -1),
   289:             ),
   290:             dim=2,
   291:         ).view(-1, 2 * self.feature_dimension, *query_features.shape[2:])
   292: 
   293:         relation_scores = self.relation_module(query_prototype_pairs).view(
   294:             n_queries, n_prototypes
   295:         )
   296:         return self.softmax_if_specified(relation_scores)
   297: 
   298:     @staticmethod
   299:     def is_transductive() -> bool:
   300:         return False
   301: 
   302:     def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
   303:         # RelationNet uses MSE with one-hot labels
   304:         one_hot = F.one_hot(labels, num_classes=scores.shape[1]).float()
   305:         return F.mse_loss(scores, one_hot)
   306: 
   307: 
   308: 
   309: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
