# Custom GNN message passing mechanism for node classification — MLS-Bench
#
# EDITABLE section: CustomMessagePassingLayer + CustomGNN classes (lines 48-157).
# FIXED sections: everything else (config, data loading, training loop, evaluation).

import os
import copy
import random
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.typing import Adj, OptTensor
import torch_geometric.transforms as T
from sklearn.metrics import f1_score


# =====================================================================
# FIXED: Configuration
# =====================================================================
SEED = int(os.environ.get("SEED", "42"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
DATASET_NAME = os.environ.get("ENV", "Cora")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training hyperparameters
HIDDEN_CHANNELS = 64
NUM_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 200


# =====================================================================
# EDITABLE: Custom Message Passing Layer and GNN Model (lines 48-157)
# =====================================================================
class CustomMessagePassingLayer(MessagePassing):
    """Custom message passing layer for node classification.

    This layer defines how messages are constructed, aggregated, and used
    to update node representations. You should implement a novel message
    passing mechanism here.

    The PyG MessagePassing base class provides:
        - self.propagate(edge_index, ...): orchestrates message passing
        - Override message(): define how messages are computed per edge
        - Override aggregate(): define how messages are combined (default: 'add')
        - Override update(): define how node embeddings are updated

    Args:
        in_channels: input feature dimension
        out_channels: output feature dimension
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """Forward pass.

        Args:
            x: node feature matrix [num_nodes, in_channels]
            edge_index: graph connectivity [2, num_edges]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Transform features
        x = self.lin(x)

        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute normalization (symmetric like GCN)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Propagate messages
        out = self.propagate(edge_index, x=x, norm=norm)
        out = out + self.bias
        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        """Construct messages from source nodes.

        Args:
            x_j: source node features [num_edges, out_channels]
            norm: normalization coefficients [num_edges]

        Returns:
            Messages [num_edges, out_channels]
        """
        return norm.view(-1, 1) * x_j


class CustomGNN(nn.Module):
    """GNN model using CustomMessagePassingLayer for node classification.

    Must implement __init__ and forward.
    The model receives the full graph and returns logits for each node.

    Args:
        in_channels: number of input features per node
        hidden_channels: hidden layer dimension
        out_channels: number of output classes
        num_layers: number of message passing layers
        dropout: dropout probability
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(CustomMessagePassingLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(CustomMessagePassingLayer(hidden_channels, hidden_channels))
        self.convs.append(CustomMessagePassingLayer(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """Forward pass through the GNN.

        Args:
            x: node feature matrix [num_nodes, in_channels]
            edge_index: graph connectivity [2, num_edges]

        Returns:
            Node classification logits [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# =====================================================================
# FIXED: Data loading
# =====================================================================
def load_dataset(name: str):
    """Load a Planetoid citation network dataset.

    Args:
        name: one of 'Cora', 'CiteSeer', 'PubMed'

    Returns:
        dataset, data (moved to device)
    """
    dataset = Planetoid(root=os.environ.get("DATA_ROOT", "/data") + "/Planetoid", name=name,
                        transform=T.NormalizeFeatures())
    data = dataset[0].to(DEVICE)
    return dataset, data


# =====================================================================
# FIXED: Training and evaluation
# =====================================================================
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)

    results = {}
    for split, mask in [("train", data.train_mask),
                        ("val", data.val_mask),
                        ("test", data.test_mask)]:
        correct = int((pred[mask] == data.y[mask]).sum())
        total = int(mask.sum())
        acc = correct / total
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        f1 = f1_score(y_true, y_pred, average="macro")
        results[f"{split}_acc"] = acc
        results[f"{split}_f1"] = f1
    return results


# =====================================================================
# FIXED: Main
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

    # Load data
    dataset, data = load_dataset(DATASET_NAME)
    in_channels = dataset.num_node_features
    out_channels = dataset.num_classes

    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, "
          f"Features: {in_channels}, Classes: {out_channels}", flush=True)

    # Build model
    model = CustomGNN(
        in_channels=in_channels,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=out_channels,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params}", flush=True)

    # ── Parameter Budget Check ──
    # Budget = 1.05x the largest baseline across all baselines.
    # GraphSAGE uses 2 linear projections per layer (lin_self + lin_neigh),
    # which doubles the input projection cost on high-dim datasets (Cora 1433,
    # CiteSeer 3703), making it the largest baseline on those datasets.
    # GPS uses MultiheadAttention + FFN + 3 LayerNorms per layer.
    # NAGphormer uses Transformer encoder layers.
    # We compute all three and take the max.
    _H = HIDDEN_CHANNELS
    # GraphSAGE: 2 layers, each with lin_self(in,out)+bias + lin_neigh(in,out) no bias
    # Layer 1: 2*in*H + H, Layer 2: 2*H*out + out
    _graphsage_params = (2 * in_channels * _H + _H) + (2 * _H * out_channels + out_channels)
    # GPS: 2 layers + classifier(H*out+out)
    # Layer 1 (in->H): lin_in(in*H+H) + lin_msg(H*H) + lin_update(H*H+H)
    #   + MultiheadAttention(4*H*H+4*H) + FFN(4*H*H+3*H) + 3*LayerNorm(6*H)
    #   = in*H + 10*H*H + 15*H
    # Layer 2 (H->H): no lin_in + same = 10*H*H + 14*H
    _gps_params = (
        in_channels * _H + 10 * _H * _H + 15 * _H  # layer 1
        + 10 * _H * _H + 14 * _H                     # layer 2
        + _H * out_channels + out_channels             # classifier
    )
    # NAGphormer: tokenizer + hop_embed + norm + transformer + attn_vec + classifier
    _nagphormer_params = (
        in_channels * _H + _H            # tokenizer.lin
        + 6 * _H                          # hop_embedding
        + 2 * _H                          # input_norm
        + 16 * _H * _H + 24 * _H         # TransformerEncoder (2 layers + final norm)
        + _H + 1                          # attn_vec
        + _H * _H + _H + _H * out_channels + out_channels  # classifier
    )
    _max_baseline = max(_graphsage_params, _gps_params, _nagphormer_params)
    _param_budget = int(_max_baseline * 1.05)
    print(f"Parameter budget: {num_params:,} / {_param_budget:,} (1.05x largest baseline)", flush=True)

    # Allow model to override training hyperparams via attributes
    lr = getattr(model, 'custom_lr', LEARNING_RATE)
    wd = getattr(model, 'custom_wd', WEIGHT_DECAY)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Training loop
    best_val_acc = 0.0
    best_state = None
    patience = 50
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        loss = train(model, data, optimizer)
        results = evaluate(model, data)

        if epoch % 10 == 0 or epoch == 1:
            print(f"TRAIN_METRICS epoch={epoch} loss={loss:.4f} "
                  f"train_acc={results['train_acc']:.4f} "
                  f"val_acc={results['val_acc']:.4f} "
                  f"test_acc={results['test_acc']:.4f}", flush=True)

        if results["val_acc"] > best_val_acc:
            best_val_acc = results["val_acc"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    # Load best model and final evaluation
    model.load_state_dict(best_state)
    final = evaluate(model, data)

    print(f"TEST_METRICS accuracy={final['test_acc']:.4f} "
          f"macro_f1={final['test_f1']:.4f}", flush=True)
    print(f"Final test accuracy: {100 * final['test_acc']:.2f}%", flush=True)
    print(f"Final test macro F1: {100 * final['test_f1']:.2f}%", flush=True)
