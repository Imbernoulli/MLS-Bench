"""
Graph Link Prediction — Self-contained template.
Predicts missing links in graphs using learned node representations and a
link scoring function. Evaluated on citation networks (Cora, CiteSeer) and
a collaboration network (ogbl-collab).

Structure:
  Lines 1-126:   FIXED — Imports, data loading, negative sampling, evaluation
  Lines 127-210: EDITABLE — LinkPredictor class (model + scoring)
  Lines 211+:    FIXED — Training loop, metric computation, CLI
"""
import os
import sys
import math
import argparse
import warnings
import numpy as np
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import (
    negative_sampling,
    to_undirected,
    add_self_loops,
    degree,
    coalesce,
)
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GINConv, GraphConv,
    MessagePassing, global_mean_pool, global_add_pool,
)
from torch_geometric.transforms import RandomLinkSplit

from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# Data loading utilities
# =====================================================================

def load_planetoid(name: str, data_dir: str) -> dict:
    """Load Cora or CiteSeer with random 85/5/10 link split."""
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root=data_dir, name=name)
    data = dataset[0]

    transform = RandomLinkSplit(
        num_val=0.05, num_test=0.10,
        is_undirected=True,
        add_negative_train_samples=True,
        split_labels=True,
    )
    train_data, val_data, test_data = transform(data)
    return {
        "train": train_data, "val": val_data, "test": test_data,
        "num_nodes": data.num_nodes,
        "num_features": data.num_node_features,
        "dataset_type": "planetoid",
    }


def load_ogbl_collab(data_dir: str) -> dict:
    """Load ogbl-collab with official split."""
    from ogb.linkproppred import PygLinkPropPredDataset
    dataset = PygLinkPropPredDataset(name="ogbl-collab", root=data_dir)
    data = dataset[0]
    split_edge = dataset.get_edge_split()

    # Build training graph
    row, col = data.edge_index
    train_edge = split_edge["train"]["edge"]
    train_ei = torch.cat([train_edge, train_edge.flip(1)], dim=0).t()
    train_ei = coalesce(train_ei)
    train_data = Data(x=data.x, edge_index=train_ei, num_nodes=data.num_nodes)

    return {
        "train_data": train_data,
        "split_edge": split_edge,
        "num_nodes": data.num_nodes,
        "num_features": data.x.size(1) if data.x is not None else 128,
        "dataset_type": "ogbl",
    }


def compute_mrr(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """Compute MRR: for each positive, rank among all negatives."""
    # pos_scores: [num_pos], neg_scores: [num_pos, num_neg] or [num_neg]
    if neg_scores.dim() == 1:
        neg_scores = neg_scores.unsqueeze(0).expand(pos_scores.size(0), -1)
    # rank = 1 + number of negatives scored higher
    ranks = (neg_scores >= pos_scores.unsqueeze(1)).sum(dim=1) + 1
    return (1.0 / ranks.float()).mean().item()


def compute_hits_at_k(pos_scores: torch.Tensor, neg_scores: torch.Tensor,
                       k: int = 50) -> float:
    """Compute Hits@K."""
    if neg_scores.dim() == 1:
        neg_scores = neg_scores.unsqueeze(0).expand(pos_scores.size(0), -1)
    kth_neg, _ = neg_scores.kthvalue(max(neg_scores.size(1) - k + 1, 1), dim=1)
    return (pos_scores >= kth_neg).float().mean().item()


# =====================================================================
# EDITABLE SECTION START — Lines 127-210
# Implement your link prediction model below.
# You MUST define a class named `LinkPredictor` with the following interface:
#   __init__(self, in_channels, hidden_channels, num_layers, dropout)
#   encode(self, x, edge_index) -> node embeddings [N, hidden_channels]
#   decode(self, edge_label_index, z, edge_index=None, num_nodes=None)
#       -> scores [num_edges]
#   forward(self, x, edge_index, edge_label_index) -> scores [num_edges]
#
# Note: `decode` receives the original `edge_label_index` (shape [2, E])
# plus the full node embedding table `z`, so structural features can be
# computed directly from the true node indices (no index recovery needed).
# =====================================================================

class LinkPredictor(nn.Module):
    """
    Link prediction model.

    Default: 2-layer GCN encoder + dot-product decoder (simple baseline).
    The agent should replace this with a better approach.

    Args:
        in_channels: Input feature dimension per node.
        hidden_channels: Hidden dimension.
        num_layers: Number of GNN layers.
        dropout: Dropout rate.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 256,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # BN only on intermediate layers — not on the last layer
        # to preserve embedding magnitude for dot-product scoring
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)
        ])

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode nodes into embeddings.

        Args:
            x: Node features [N, in_channels].
            edge_index: Graph connectivity [2, E].

        Returns:
            Node embeddings [N, hidden_channels].
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def decode(self, edge_label_index: torch.Tensor, z: torch.Tensor,
               edge_index: Optional[torch.Tensor] = None,
               num_nodes: Optional[int] = None) -> torch.Tensor:
        """Score candidate edges.

        Args:
            edge_label_index: Candidate edges [2, num_edges] (original node
                indices into `z`).
            z: Full node embedding table [N, hidden_channels].
            edge_index: Optional training graph connectivity [2, E], available
                so structure-aware decoders can compute CN/AA/RA etc.
            num_nodes: Optional number of nodes in the graph.

        Returns:
            Edge scores [num_edges].
        """
        z_src = z[edge_label_index[0]]
        z_dst = z[edge_label_index[1]]
        return (z_src * z_dst).sum(dim=-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_label_index: torch.Tensor) -> torch.Tensor:
        """Full forward: encode all nodes, then decode candidate edges.

        Args:
            x: Node features [N, in_channels].
            edge_index: Training graph connectivity [2, E_train].
            edge_label_index: Candidate edges to score [2, num_candidates].

        Returns:
            Edge scores [num_candidates].
        """
        z = self.encode(x, edge_index)
        return self.decode(edge_label_index, z, edge_index=edge_index,
                           num_nodes=x.size(0))

# Helper functions may be defined here as needed.

# =====================================================================
# EDITABLE SECTION END
# =====================================================================

# =====================================================================
# FIXED — Training loop, evaluation, CLI
# =====================================================================

def train_planetoid(model, data_bundle, args, device):
    """Train and evaluate on Planetoid (Cora/CiteSeer)."""
    train_data = data_bundle["train"].to(device)
    val_data = data_bundle["val"].to(device)
    test_data = data_bundle["test"].to(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Positive edges from split; resample negatives each epoch
        pos_ei = train_data.pos_edge_label_index
        neg_ei = negative_sampling(
            train_data.edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=pos_ei.size(1),
        )

        pos_scores = model(train_data.x, train_data.edge_index, pos_ei)
        neg_scores = model(train_data.x, train_data.edge_index, neg_ei)

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validation
        if epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_pos = model(val_data.x, train_data.edge_index,
                                val_data.pos_edge_label_index)
                val_neg = model(val_data.x, train_data.edge_index,
                                val_data.neg_edge_label_index)
            val_scores = torch.cat([val_pos, val_neg]).sigmoid().cpu().numpy()
            val_labels = np.concatenate([
                np.ones(val_pos.size(0)), np.zeros(val_neg.size(0))
            ])
            val_auc = roc_auc_score(val_labels, val_scores) * 100

            print(f"TRAIN_METRICS epoch={epoch} loss={loss.item():.4f} "
                  f"val_auc={val_auc:.2f}", flush=True)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}.", flush=True)
                    break

    # Test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pos = model(test_data.x, train_data.edge_index,
                         test_data.pos_edge_label_index)
        test_neg = model(test_data.x, train_data.edge_index,
                         test_data.neg_edge_label_index)

    # AUC
    scores = torch.cat([test_pos, test_neg]).sigmoid().cpu().numpy()
    labels = np.concatenate([
        np.ones(test_pos.size(0)), np.zeros(test_neg.size(0))
    ])
    auc = roc_auc_score(labels, scores) * 100

    # MRR
    mrr = compute_mrr(test_pos.cpu(), test_neg.cpu()) * 100

    # Hits@20
    hits20 = compute_hits_at_k(test_pos.cpu(), test_neg.cpu(), k=20) * 100

    print(f"TEST_METRICS AUC={auc:.2f} MRR={mrr:.2f} Hits@20={hits20:.2f}",
          flush=True)


def train_ogbl(model, data_bundle, args, device):
    """Train and evaluate on ogbl-collab."""
    from ogb.linkproppred import Evaluator
    evaluator = Evaluator(name="ogbl-collab")

    train_data = data_bundle["train_data"].to(device)
    split_edge = data_bundle["split_edge"]

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    best_val_hits = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Sample positive and negative training edges
        pos_train = split_edge["train"]["edge"].to(device)
        # Subsample for efficiency
        n_pos = min(pos_train.size(0), args.batch_size)
        idx = torch.randperm(pos_train.size(0))[:n_pos]
        pos_ei = pos_train[idx].t()  # [2, n_pos]

        neg_ei = negative_sampling(
            train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=n_pos,
        )

        x = train_data.x
        if x is None:
            x = torch.ones(train_data.num_nodes, 1, device=device)

        pos_scores = model(x, train_data.edge_index, pos_ei)
        neg_scores = model(x, train_data.edge_index, neg_ei)

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validation
        if epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                z = model.encode(x, train_data.edge_index)

                val_pos = split_edge["valid"]["edge"].to(device)
                val_neg = split_edge["valid"]["edge_neg"].to(device)

                _N = train_data.num_nodes
                pos_eli = val_pos.t().contiguous()  # [2, P]
                val_pos_scores = model.decode(
                    pos_eli, z,
                    edge_index=train_data.edge_index, num_nodes=_N)
                if val_neg.dim() == 3:
                    vn = val_neg.reshape(-1, 2)
                    neg_eli = vn.t().contiguous()
                    val_neg_scores = model.decode(
                        neg_eli, z,
                        edge_index=train_data.edge_index, num_nodes=_N)
                    val_neg_scores = val_neg_scores.view(val_neg.size(0), val_neg.size(1))
                elif val_neg.dim() == 2 and val_neg.size(1) == 2:
                    neg_eli = val_neg.t().contiguous()
                    val_neg_scores = model.decode(
                        neg_eli, z,
                        edge_index=train_data.edge_index, num_nodes=_N)
                else:
                    # [num_pos, K] format: destinations only, source = val_pos source
                    src_rep = val_pos[:, 0].unsqueeze(1).expand_as(val_neg).reshape(-1)
                    dst_rep = val_neg.reshape(-1)
                    neg_eli = torch.stack([src_rep, dst_rep], dim=0)
                    val_neg_scores = model.decode(
                        neg_eli, z,
                        edge_index=train_data.edge_index, num_nodes=_N)
                    val_neg_scores = val_neg_scores.view(val_neg.size(0), val_neg.size(1))

            val_hits = compute_hits_at_k(val_pos_scores.cpu(), val_neg_scores.cpu(), k=50) * 100

            print(f"TRAIN_METRICS epoch={epoch} loss={loss.item():.4f} "
                  f"val_hits50={val_hits:.2f}", flush=True)

            if val_hits > best_val_hits:
                best_val_hits = val_hits
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}.", flush=True)
                    break

    # Test evaluation
    # OGB standard: include validation edges in the adjacency at test time
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        x = train_data.x
        if x is None:
            x = torch.ones(train_data.num_nodes, 1, device=device)

        # Build test-time adjacency: train + validation edges
        val_edge = split_edge["valid"]["edge"].to(device)
        val_ei = torch.cat([val_edge, val_edge.flip(1)], dim=0).t()
        test_edge_index = coalesce(
            torch.cat([train_data.edge_index, val_ei], dim=1))

        z = model.encode(x, test_edge_index)

        test_pos = split_edge["test"]["edge"].to(device)
        test_neg = split_edge["test"]["edge_neg"].to(device)

        _N = train_data.num_nodes
        pos_eli = test_pos.t().contiguous()  # [2, P]
        pos_scores = model.decode(
            pos_eli, z, edge_index=test_edge_index, num_nodes=_N)
        if test_neg.dim() == 3:
            tn = test_neg.reshape(-1, 2)
            neg_eli = tn.t().contiguous()
            neg_scores = model.decode(
                neg_eli, z, edge_index=test_edge_index, num_nodes=_N)
            neg_scores = neg_scores.view(test_neg.size(0), test_neg.size(1))
        elif test_neg.dim() == 2 and test_neg.size(1) == 2:
            neg_eli = test_neg.t().contiguous()
            neg_scores = model.decode(
                neg_eli, z, edge_index=test_edge_index, num_nodes=_N)
        else:
            src_rep = test_pos[:, 0].unsqueeze(1).expand_as(test_neg).reshape(-1)
            dst_rep = test_neg.reshape(-1)
            neg_eli = torch.stack([src_rep, dst_rep], dim=0)
            neg_scores = model.decode(
                neg_eli, z, edge_index=test_edge_index, num_nodes=_N)
            neg_scores = neg_scores.view(test_neg.size(0), test_neg.size(1))

    hits50 = compute_hits_at_k(pos_scores.cpu(), neg_scores.cpu(), k=50) * 100
    mrr = compute_mrr(pos_scores.cpu(), neg_scores.cpu()) * 100

    print(f"TEST_METRICS Hits@50={hits50:.2f} MRR={mrr:.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Graph Link Prediction")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["Cora", "CiteSeer", "ogbl-collab"])
    parser.add_argument("--data-dir", type=str, default="/data")
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./output")
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Parameter Budget Check ──
    # Budget = 1.05x the largest baseline.
    # Neo-GNN is the largest GNN-based baseline, but node2vec uses
    # nn.Embedding(max_num_nodes, H) which can be 50000*H = 12.8M+ params,
    # fundamentally different from GNN methods (~500K params).
    # We take max(gnn_budget, embedding_budget).
    def _check_param_budget(model, in_ch, H):
        # Neo-GNN: GCN encoder(2 layers + 2 BN) + 3 struct_layers + hop_weights + decoder
        _neo_gnn_params = (
            in_ch * H + H * H + 6 * H     # GCN encoder (2 layers + BN)
            + 3 * H * H + 9 * H            # struct_layers (3 NeoGNNLayers)
            + 3                             # hop_weights
            + 2 * H * H + 3 * H + 1        # decoder MLP
        )
        # node2vec: Embedding(50000, H) + feat_proj(in*H+H + H*H+H)
        #   + decoder(3*H*H+H + H*H+H + H+1)
        _max_num_nodes = 50000
        _node2vec_params = (
            _max_num_nodes * H             # node_emb
            + in_ch * H + H + H * H + H    # feat_proj
            + 3 * H * H + H + H * H + H + H + 1  # decoder MLP
        )
        _max_baseline = max(_neo_gnn_params, _node2vec_params)
        _budget = int(_max_baseline * 1.05)
        _n = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {_n:,} (budget: {_budget:,})", flush=True)

    # Load data
    if args.dataset in ("Cora", "CiteSeer"):
        data_bundle = load_planetoid(
            args.dataset, os.path.join(args.data_dir, "Planetoid"))
        in_channels = data_bundle["num_features"]
        model = LinkPredictor(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
        _check_param_budget(model, in_channels, args.hidden_channels)
        train_planetoid(model, data_bundle, args, device)

    elif args.dataset == "ogbl-collab":
        data_bundle = load_ogbl_collab(os.path.join(args.data_dir, "OGB"))
        in_channels = data_bundle["num_features"]
        model = LinkPredictor(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
        _check_param_budget(model, in_channels, args.hidden_channels)
        train_ogbl(model, data_bundle, args, device)


if __name__ == "__main__":
    main()
