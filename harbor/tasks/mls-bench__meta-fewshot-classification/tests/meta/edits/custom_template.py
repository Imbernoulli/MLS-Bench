# Custom few-shot classification method for MLS-Bench
#
# EDITABLE section: CustomFewShotMethod class and helper modules.
# FIXED sections: everything else (config, data loading, training loop, evaluation).
import os
import copy
import random
import json
from pathlib import Path
from statistics import mean
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from tqdm import tqdm

from easyfsl.datasets import FewShotDataset
from easyfsl.samplers import TaskSampler
from easyfsl.methods import FewShotClassifier
from easyfsl.methods.utils import compute_prototypes


# =====================================================================
# FIXED: Configuration
# =====================================================================
SEED = int(os.environ.get("SEED", "42"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
DATASET_NAME = os.environ.get("ENV", "mini_imagenet")

# Few-shot settings
N_WAY = 5
N_SHOT = 5
N_QUERY = 15
IMAGE_SIZE = 84

# Training settings
N_EPOCHS = 200
N_TASKS_PER_EPOCH = 500
N_VALIDATION_TASKS = 200
N_TEST_TASKS = 600
LEARNING_RATE = 1e-2
SCHEDULER_MILESTONES = [120, 160]
SCHEDULER_GAMMA = 0.1
WEIGHT_DECAY = 5e-4
GRAD_CLIP_NORM = 5.0  # Vinyals et al. 2016 Sec 3.1 (Matching Networks)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_WORKERS = 4

IMAGENET_NORMALIZATION = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


# =====================================================================
# FIXED: Dataset loading
# =====================================================================
class ImageFolderFewShot(FewShotDataset):
    """A general-purpose few-shot dataset that loads images from class-organized directories."""

    def __init__(
        self,
        specs_file: str,
        image_size: int = 84,
        training: bool = False,
    ):
        specs = self._load_specs(specs_file)
        self.images: List[str] = []
        self.labels: List[int] = []

        supported_formats = {".bmp", ".png", ".jpeg", ".jpg", ".JPEG"}
        for class_id, class_root in enumerate(specs["class_roots"]):
            class_images = sorted(
                str(p) for p in Path(class_root).glob("*")
                if p.is_file() and p.suffix in supported_formats
            )
            self.images.extend(class_images)
            self.labels.extend([class_id] * len(class_images))

        self.transform = (
            transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**IMAGENET_NORMALIZATION),
            ])
            if training
            else transforms.Compose([
                transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(**IMAGENET_NORMALIZATION),
            ])
        )

    @staticmethod
    def _load_specs(specs_file: str) -> dict:
        ext = Path(specs_file).suffix
        if ext == ".json":
            with open(specs_file, "r") as f:
                return json.load(f)
        elif ext == ".csv":
            # miniImageNet CSV format: class_name,image_name
            # Images are expected at: <specs_dir>/../images/<class_name>/ via symlink
            import pandas as pd
            df = pd.read_csv(specs_file)
            specs_dir = str(Path(specs_file).parent)  # e.g. .../data/mini_imagenet
            class_names = df["class_name"].unique().tolist()
            class_roots = [os.path.join(specs_dir, "images", cn) for cn in class_names]
            return {"class_names": class_names, "class_roots": class_roots}
        else:
            raise ValueError(f"Unsupported spec file format: {ext}")

    def __getitem__(self, item):
        from PIL import Image
        img = self.transform(Image.open(self.images[item]).convert("RGB"))
        return img, self.labels[item]

    def __len__(self):
        return len(self.labels)

    def get_labels(self) -> List[int]:
        return self.labels


def get_dataset_specs(dataset_name: str):
    """Return (train_spec, val_spec, test_spec) file paths for the given dataset."""
    _pkg_dir = os.environ.get("MLSBENCH_PKG_DIR", "/workspace/easy-few-shot-learning")
    specs_dir = f"{_pkg_dir}/data/{dataset_name}"

    if dataset_name == "mini_imagenet":
        # miniImageNet uses CSV spec files
        return (
            f"{specs_dir}/train.csv",
            f"{specs_dir}/val.csv",
            f"{specs_dir}/test.csv",
        )
    else:
        # CUB, cifar_fs, tiered_imagenet use JSON spec files
        return (
            f"{specs_dir}/train.json",
            f"{specs_dir}/val.json",
            f"{specs_dir}/test.json",
        )


def make_data_loader(specs_file: str, n_way: int, n_shot: int, n_query: int,
                     n_tasks: int, training: bool) -> DataLoader:
    dataset = ImageFolderFewShot(specs_file, image_size=IMAGE_SIZE, training=training)
    sampler = TaskSampler(dataset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=N_WORKERS,
        pin_memory=True,
        collate_fn=sampler.episodic_collate_fn,
    )


# =====================================================================
# FIXED: ResNet backbone (shared by all methods)
# =====================================================================
from easyfsl.modules import resnet12


def make_backbone(use_pooling: bool = True) -> nn.Module:
    """Create a ResNet-12 backbone.

    Args:
        use_pooling: if True, output is [B, 640] feature vectors.
                     if False, output is [B, 640, H, W] feature maps.
    """
    return resnet12(use_pooling=use_pooling)


FEATURE_DIMENSION = 640  # ResNet12 output dimension


# =====================================================================
# FIXED: Training utilities
# =====================================================================
def training_epoch(model, data_loader, optimizer):
    all_loss = []
    model.train()
    for support_images, support_labels, query_images, query_labels, _ in data_loader:
        optimizer.zero_grad()
        model.process_support_set(
            support_images.to(DEVICE), support_labels.to(DEVICE)
        )
        classification_scores = model(query_images.to(DEVICE))
        loss = model.compute_loss(classification_scores, query_labels.to(DEVICE))
        loss.backward()
        # Vinyals et al. 2016 (Matching Networks) Sec 3.1: dampen gradients
        # with norm > 5. Generally beneficial for LSTM/recurrent components.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()
        all_loss.append(loss.item())
    return mean(all_loss)


def evaluate(model, data_loader):
    total_predictions = 0
    correct_predictions = 0
    model.eval()
    with torch.no_grad():
        for support_images, support_labels, query_images, query_labels, _ in data_loader:
            model.process_support_set(
                support_images.to(DEVICE), support_labels.to(DEVICE)
            )
            predictions = model(query_images.to(DEVICE)).detach().data
            correct = int((torch.max(predictions, 1)[1] == query_labels.to(DEVICE)).sum().item())
            correct_predictions += correct
            total_predictions += len(query_labels)
    return correct_predictions / total_predictions if total_predictions > 0 else 0.0


# =====================================================================
# EDITABLE: Custom Few-Shot Classification Method
# =====================================================================
class CustomFewShotMethod(FewShotClassifier):
    """Custom few-shot classification method.

    This class defines how to classify query images given a support set.
    You MUST implement:
        - __init__: create the backbone and any learnable modules
        - process_support_set(support_images, support_labels): extract and store
          information from the support set for later query classification
        - forward(query_images) -> Tensor of shape (n_query, n_way): predict
          classification scores for query images

    Available utilities (from easyfsl):
        - self.compute_features(images): pass images through self.backbone
        - self.l2_distance_to_prototypes(features): compute negative L2 distance to self.prototypes
        - self.cosine_distance_to_prototypes(features): compute cosine similarity to self.prototypes
        - self.softmax_if_specified(scores): apply softmax if self.use_softmax is set
        - compute_prototypes(features, labels): compute class prototypes (mean features)

    The backbone should be set as self.backbone (an nn.Module).
    Feature dimension of the ResNet12 backbone is 640.

    The training loop calls model.compute_loss(scores, labels) for flexibility.
    Override compute_loss if your method needs a different loss (e.g., MSE for RelationNet).
    """

    def __init__(self):
        backbone = make_backbone(use_pooling=True)
        super().__init__(backbone=backbone, use_softmax=False)

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        """Extract and store support set information."""
        self.compute_prototypes_and_store_support_set(support_images, support_labels)

    def forward(self, query_images: Tensor) -> Tensor:
        """Predict classification scores for query images.

        Args:
            query_images: images of shape (n_query, 3, 84, 84)

        Returns:
            scores of shape (n_query, n_way) — higher means more likely
        """
        query_features = self.compute_features(query_images)
        scores = self.l2_distance_to_prototypes(query_features)
        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() -> bool:
        return False

    def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
        """Compute the training loss. Override for custom loss functions.

        Args:
            scores: classification scores of shape (n_query, n_way)
            labels: ground truth labels of shape (n_query,), integers in [0, n_way)

        Returns:
            scalar loss tensor
        """
        return F.cross_entropy(scores, labels)


# =====================================================================
# FIXED: Main training and evaluation script
# =====================================================================
if __name__ == "__main__":
    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Dataset: {DATASET_NAME}, Seed: {SEED}", flush=True)
    print(f"Few-shot setting: {N_WAY}-way {N_SHOT}-shot {N_QUERY}-query", flush=True)

    # Load data
    train_spec, val_spec, test_spec = get_dataset_specs(DATASET_NAME)
    train_loader = make_data_loader(train_spec, N_WAY, N_SHOT, N_QUERY, N_TASKS_PER_EPOCH, training=True)
    val_loader = make_data_loader(val_spec, N_WAY, N_SHOT, N_QUERY, N_VALIDATION_TASKS, training=False)
    test_loader = make_data_loader(test_spec, N_WAY, N_SHOT, N_QUERY, N_TEST_TASKS, training=False)

    # Build model
    model = CustomFewShotMethod().to(DEVICE)

    # ── FIXED: Parameter count check ────────────────────────────────
    # Budget based on 1.05x largest baseline (RelationNet).
    # RelationNet: ResNet12 backbone (no pooling) + RelationModule
    # ResNet12 backbone ~ 8.7M params (standard for few-shot)
    # RelationModule: Conv2d(2*640,640,3)+BN + Conv2d(640,640,3)+BN + Linear(640,8) + Linear(8,1)
    _backbone_params = 8_700_000  # ResNet12 backbone approximate
    _relation_module = 2 * FEATURE_DIMENSION * FEATURE_DIMENSION * 9 + FEATURE_DIMENSION  # conv1
    _relation_module += FEATURE_DIMENSION * 2  # BN1
    _relation_module += FEATURE_DIMENSION * FEATURE_DIMENSION * 9 + FEATURE_DIMENSION  # conv2
    _relation_module += FEATURE_DIMENSION * 2  # BN2
    _relation_module += FEATURE_DIMENSION * 8 + 8 + 8 * 1 + 1  # FC layers
    _budget = int((_backbone_params + _relation_module + 5000) * 1.05)
    _total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {_total_params:,} (budget: {_budget:,})")

    # Optimizer and scheduler — methods may override the LR via class attr LR_OVERRIDE.
    # TODO: Chen et al. 2019 (https://arxiv.org/abs/1904.04232, Sec. 5)
    # trains meta-learning methods with Adam@1e-3 and scales MatchingNet cosine
    # similarities. Validate Adam/scalar ablations before comparing to that order.
    _lr = getattr(model, "LR_OVERRIDE", LEARNING_RATE)
    if _lr != LEARNING_RATE:
        print(f"  LR override active: {_lr} (default {LEARNING_RATE})", flush=True)
    optimizer = SGD(model.parameters(), lr=_lr, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = MultiStepLR(optimizer, milestones=SCHEDULER_MILESTONES, gamma=SCHEDULER_GAMMA)

    # Training loop
    best_state = model.state_dict()
    best_val_acc = 0.0

    for epoch in range(N_EPOCHS):
        avg_loss = training_epoch(model, train_loader, optimizer)
        val_acc = evaluate(model, val_loader)
        scheduler.step()

        print(f"TRAIN_METRICS epoch={epoch} train_loss={avg_loss:.5f} val_acc={val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            print(f"  New best val accuracy: {val_acc:.4f}", flush=True)

    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader)
    print(f"TEST_METRICS accuracy={test_acc:.4f}", flush=True)
    print(f"Test accuracy: {100 * test_acc:.2f}%", flush=True)
