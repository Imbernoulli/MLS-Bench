# Custom graph signal propagation filter for MLS-Bench
#
# EDITABLE section: CustomProp (propagation layer) + CustomFilter (full model).
# FIXED sections: everything else (config, data loading, training loop, evaluation).

import os
import math
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import GCNConv, APPNP as PyGAPPNP
from torch_geometric.utils import get_laplacian, add_self_loops
from scipy.special import comb

# =====================================================================
# FIXED: Configuration
# =====================================================================
SEED = int(os.environ.get("SEED", "42"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
DATASET_NAME = os.environ.get("ENV", "cora")

# Training settings
EPOCHS = 1000
EARLY_STOPPING = 200
HIDDEN = 64
K = 10          # polynomial order / propagation steps
ALPHA = 0.1     # teleport probability (for PPR-style init)
DROPOUT = 0.5
DPRATE = 0.0    # no propagation dropout (hurts spectral filters on heterophilic data)
LR = 0.05
WEIGHT_DECAY = 0.0
PROP_LR = 0.01       # lower lr for propagation/filter params (stable learning)
PROP_WD = 0.0        # no weight decay for filter coefficients
TRAIN_RATE = 0.6
VAL_RATE = 0.2
RUNS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 10 BernNet reference seeds; MLS-Bench SEED offsets them across runs
FIXED_SEEDS = [
    1941488137, 4198936517, 983997847, 4023022221, 4019585660,
    2108550661, 1648766618, 629014539, 3212139042, 2424918363,
]


# =====================================================================
# FIXED: Dataset loading (from ChebNetII codebase)
# =====================================================================
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected
import os.path as osp
import pickle


class WebKB(InMemoryDataset):
    """WebKB dataset (Texas, Cornell, Wisconsin, Washington)."""
    url = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data"

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ["cornell", "texas", "washington", "wisconsin"]
        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return ["out1_node_feature_label.txt", "out1_graph_edges.txt"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        for name in self.raw_file_names:
            download_url(f"{self.url}/{self.name}/{name}", self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], "r") as f:
            data = f.read().split("\n")[1:-1]
        x = [[float(v) for v in r.split("\t")[1].split(",")] for r in data]
        x = torch.tensor(x, dtype=torch.float)
        y = [int(r.split("\t")[2]) for r in data]
        y = torch.tensor(y, dtype=torch.long)
        with open(self.raw_paths[1], "r") as f:
            data = f.read().split("\n")[1:-1]
        data = [[int(v) for v in r.split("\t")] for r in data]
        edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])


def load_dataset(name):
    """Load a graph dataset by name."""
    name_lower = name.lower()
    data_root = os.environ.get("MLSBENCH_PKG_DIR", "/workspace/ChebNetII") + "/main/data"
    if name_lower in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(osp.join(data_root, name_lower), name_lower,
                            transform=T.NormalizeFeatures())
    elif name_lower in ["texas", "cornell"]:
        dataset = WebKB(root=data_root, name=name_lower,
                        transform=T.NormalizeFeatures())
    else:
        raise ValueError(f"Dataset {name} not supported")
    return dataset


# =====================================================================
# FIXED: Data splitting utilities
# =====================================================================
def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def random_splits(data, num_classes, percls_trn, val_lb, seed=42):
    """Create random train/val/test splits."""
    index = list(range(data.y.shape[0]))
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]
    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)
    return data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cheby(i, x):
    """Evaluate Chebyshev polynomial T_i(x)."""
    if i == 0:
        return 1
    elif i == 1:
        return x
    else:
        T0, T1 = 1, x
        for _ in range(2, i + 1):
            T2 = 2 * x * T1 - T0
            T0, T1 = T1, T2
        return T2


# =====================================================================
# FIXED: Training and evaluation functions
# =====================================================================
def train_step(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)[data.train_mask]
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, data):
    model.eval()
    logits = model(data)
    accs, losses = [], []
    for mask_name in ["train_mask", "val_mask", "test_mask"]:
        mask = getattr(data, mask_name)
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        accs.append(acc)
        losses.append(loss)
    return accs, losses


# =====================================================================
# EDITABLE: Custom Graph Signal Propagation Filter
# =====================================================================
class CustomProp(MessagePassing):
    """Custom graph signal propagation layer.

    This layer defines how node features are propagated (filtered) across
    the graph structure. It operates on the graph Laplacian spectrum.

    Design a novel spectral or spatial graph filter here. The filter should:
    1. Accept node features x and edge_index as input
    2. Apply graph-based propagation/filtering
    3. Return filtered node features

    Available graph operators (from PyG):
    - get_laplacian(edge_index, normalization='sym') -> (edge_index, norm)
      Returns the symmetric normalized Laplacian L = I - D^{-1/2}AD^{-1/2}
    - add_self_loops(edge_index, edge_weight, fill_value) -> (edge_index, weight)
    - gcn_norm(edge_index) -> (edge_index, norm)
      Returns D^{-1/2}AD^{-1/2} normalization
    - self.propagate(edge_index, x=x, norm=norm) for message passing

    Config available: K (polynomial order), ALPHA (teleport probability).

    Args:
        K: number of propagation steps / polynomial order
        alpha: teleport probability (for PPR-like initialization)
    """

    def __init__(self, K, alpha=0.1, **kwargs):
        super(CustomProp, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.alpha = alpha
        # Learnable polynomial coefficients
        self.temp = Parameter(torch.Tensor(K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize with PPR-like coefficients
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        # Compute GCN-normalized adjacency: D^{-1/2}AD^{-1/2}
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype
        )
        # Weighted sum of K-hop propagations (monomial basis)
        hidden = x * self.temp[0]
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            hidden = hidden + self.temp[k + 1] * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class CustomFilter(nn.Module):
    """Full graph filter model: MLP encoder + CustomProp + softmax.

    Architecture: input -> dropout -> Linear -> ReLU -> dropout -> Linear
                  -> (optional dprate dropout) -> CustomProp -> log_softmax

    Args:
        num_features: input feature dimension
        num_classes: number of output classes
        hidden: hidden layer dimension
        K: polynomial order for propagation
        alpha: teleport probability
        dropout: dropout rate for MLP layers
        dprate: dropout rate for propagation layer
    """

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super(CustomFilter, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop = CustomProp(K, alpha)
        self.dropout = dropout
        self.dprate = dprate

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.dprate == 0.0:
            x = self.prop(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)


# =====================================================================
# FIXED: Main training and evaluation script
# =====================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Dataset: {DATASET_NAME}, Seed: {SEED}", flush=True)
    print(f"Config: hidden={HIDDEN}, K={K}, alpha={ALPHA}, "
          f"dropout={DROPOUT}, dprate={DPRATE}, lr={LR}", flush=True)

    # Load dataset
    dataset = load_dataset(DATASET_NAME)
    data = dataset[0]
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    # Compute split sizes
    percls_trn = int(round(TRAIN_RATE * len(data.y) / num_classes))
    val_lb = int(round(VAL_RATE * len(data.y)))

    results = []
    for run_idx in range(RUNS):
        run_seed = (FIXED_SEEDS[run_idx] + SEED - 42) & 0xFFFFFFFF
        set_seed(run_seed)

        # Create data split
        data_split = random_splits(data, num_classes, percls_trn, val_lb, seed=run_seed)

        # Build model
        model = CustomFilter(
            num_features=num_features,
            num_classes=num_classes,
            hidden=HIDDEN,
            K=K,
            alpha=ALPHA,
            dropout=DROPOUT,
            dprate=DPRATE,
        ).to(DEVICE)

        data_split = data_split.to(DEVICE)

        # Allow model to override training hyperparameters via attributes
        lr = getattr(model, 'custom_lr', LR)
        wd = getattr(model, 'custom_wd', WEIGHT_DECAY)
        prop_lr = getattr(model, 'custom_prop_lr', PROP_LR)
        prop_wd = getattr(model, 'custom_prop_wd', PROP_WD)

        # Check if model has separate propagation parameters
        prop_params = []
        other_params = []
        for name, param in model.named_parameters():
            if "prop" in name:
                prop_params.append(param)
            else:
                other_params.append(param)

        if prop_params:
            optimizer = torch.optim.Adam([
                {"params": other_params, "lr": lr, "weight_decay": wd},
                {"params": prop_params, "lr": prop_lr, "weight_decay": prop_wd},
            ])
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=wd
            )

        # Training loop with early stopping
        best_val_loss = float("inf")
        best_test_acc = 0.0
        val_loss_history = []

        for epoch in range(EPOCHS):
            train_loss = train_step(model, optimizer, data_split)
            accs, losses = evaluate(model, data_split)
            train_acc, val_acc, test_acc = accs
            _, val_loss, _ = losses

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_acc = test_acc

            if epoch % 100 == 0:
                print(
                    f"TRAIN_METRICS run={run_idx} epoch={epoch} "
                    f"train_loss={train_loss:.4f} val_acc={val_acc:.4f} "
                    f"test_acc={test_acc:.4f}",
                    flush=True,
                )

            val_loss_history.append(val_loss)
            if EARLY_STOPPING > 0 and epoch > EARLY_STOPPING:
                recent = torch.tensor(val_loss_history[-(EARLY_STOPPING + 1):-1])
                if val_loss > recent.mean().item():
                    break

        results.append(best_test_acc)
        print(
            f"TRAIN_METRICS run={run_idx} final best_test_acc={best_test_acc:.4f}",
            flush=True,
        )

    # Aggregate results across runs
    mean_acc = np.mean(results)
    std_acc = np.std(results)
    print(f"TEST_METRICS accuracy={mean_acc:.4f} std={std_acc:.4f}", flush=True)
    print(
        f"Result: {DATASET_NAME} accuracy = {100*mean_acc:.2f} +/- {100*std_acc:.2f}%",
        flush=True,
    )
