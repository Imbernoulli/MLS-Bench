"""
Molecular Property Prediction — Self-contained template.
Predicts molecular properties (classification: ROC-AUC, regression: RMSE)
on MoleculeNet benchmarks (BBBP, BACE, Tox21, ESOL, FreeSolv, Lipophilicity).

Uses official Uni-Mol pre-split LMDB data with train/valid/test splits
and pre-computed multi-conformer 3D coordinates.  Data pipeline mirrors
Uni-Mol: LMDB -> conformer sample/enumerate -> remove polar H -> normalize
coordinates -> Uni-Mol vocabulary tokenization -> distance matrix + edge types.

Structure:
  Lines 1-114:   FIXED — Imports, constants, atom/bond featurization
  Lines 115-207: EDITABLE — MoleculeModel class (starter: simple GIN)
  Lines 208+:    FIXED — Data loading, training loop, evaluation, TTA
"""
import os
import sys
import math
import copy
import json
import lmdb
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from scipy.spatial import distance_matrix as scipy_distance_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# Atom and bond featurization constants (used by GNN-based models)
# =====================================================================

ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'num_hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ],
}

ATOM_DIM = len(ATOM_FEATURES['atomic_num']) + len(ATOM_FEATURES['degree']) + \
           len(ATOM_FEATURES['formal_charge']) + len(ATOM_FEATURES['num_hs']) + \
           len(ATOM_FEATURES['hybridization']) + 2  # +2 for aromatic, in_ring

EDGE_DIM = len(BOND_FEATURES['bond_type']) + len(BOND_FEATURES['stereo']) + 2  # +2 for conjugated, in_ring


def one_hot(val, allowable_set):
    """One-hot encode a value. Unknown values map to all-zeros."""
    encoding = [0] * len(allowable_set)
    if val in allowable_set:
        encoding[allowable_set.index(val)] = 1
    return encoding


def atom_features(atom):
    """Compute atom feature vector."""
    features = []
    features += one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    features += one_hot(atom.GetDegree(), ATOM_FEATURES['degree'])
    features += one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    features += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs'])
    features += one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    features += [int(atom.GetIsAromatic())]
    features += [int(atom.IsInRing())]
    return features


def bond_features(bond):
    """Compute bond feature vector."""
    features = []
    features += one_hot(bond.GetBondType(), BOND_FEATURES['bond_type'])
    features += one_hot(bond.GetStereo(), BOND_FEATURES['stereo'])
    features += [int(bond.GetIsConjugated())]
    features += [int(bond.IsInRing())]
    return features


# =====================================================================
# EDITABLE SECTION START — MoleculeModel + helper modules
# =====================================================================

class GINConv(nn.Module):
    """Graph Isomorphism Network convolution layer."""

    def __init__(self, in_dim, out_dim, edge_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.edge_proj = nn.Linear(edge_dim, in_dim)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_attr, batch_idx):
        """
        x: [total_atoms, in_dim]
        edge_index: [2, total_edges]
        edge_attr: [total_edges, edge_dim]
        batch_idx: [total_atoms]
        """
        src, dst = edge_index
        edge_msg = self.edge_proj(edge_attr)
        msg = x[src] + edge_msg

        # Aggregate messages to destination nodes
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)

        out = self.mlp((1 + self.eps) * x + agg)
        return out


class MoleculeModel(nn.Module):
    """Starter model: Graph Isomorphism Network (GIN) with mean pooling.

    Simple but effective baseline for molecular property prediction.
    Uses message passing on the molecular graph with learned edge features.
    """

    def __init__(self, atom_dim: int, edge_dim: int, num_tasks: int, task_type: str):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_type = task_type
        hidden_dim = 256
        num_layers = 4

        self.atom_embed = nn.Linear(atom_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GINConv(hidden_dim, hidden_dim, edge_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.1)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, batch):
        """
        Args:
            batch: MolBatch with sparse graph data.
        Returns:
            predictions: [B, num_tasks]
        """
        x = self.atom_embed(batch.x)

        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, batch.edge_index, batch.edge_attr, batch.batch_idx)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x = x + self.dropout(x_new)  # residual

        # Mean pooling per graph
        num_graphs = batch.batch_idx.max().item() + 1
        graph_embed = torch.zeros(num_graphs, x.size(-1), device=x.device)
        counts = torch.zeros(num_graphs, 1, device=x.device)
        graph_embed.index_add_(0, batch.batch_idx, x)
        counts.index_add_(0, batch.batch_idx, torch.ones(x.size(0), 1, device=x.device))
        graph_embed = graph_embed / counts.clamp(min=1)

        return self.readout(graph_embed)

# =====================================================================
# EDITABLE SECTION END
# =====================================================================


# =====================================================================
# FIXED — Uni-Mol vocabulary, data loading, training, evaluation
# =====================================================================

# Uni-Mol atom vocabulary (mirrors dict.txt)
# [PAD]=0, [CLS]=1, [SEP]=2, [UNK]=3, C=4, N=5, O=6, S=7, H=8,
# Cl=9, F=10, Br=11, I=12, Si=13, P=14, B=15, Na=16, K=17, Al=18,
# Ca=19, Sn=20, As=21, Hg=22, Fe=23, Zn=24, Cr=25, Se=26, Gd=27,
# Au=28, Li=29, [MASK]=30
UNIMOL_ELEM_TO_IDX = {
    'C': 4, 'N': 5, 'O': 6, 'S': 7, 'H': 8, 'Cl': 9, 'F': 10,
    'Br': 11, 'I': 12, 'Si': 13, 'P': 14, 'B': 15, 'Na': 16,
    'K': 17, 'Al': 18, 'Ca': 19, 'Sn': 20, 'As': 21, 'Hg': 22,
    'Fe': 23, 'Zn': 24, 'Cr': 25, 'Se': 26, 'Gd': 27, 'Au': 28,
    'Li': 29,
}
UNIMOL_PAD_IDX = 0
UNIMOL_CLS_IDX = 1
UNIMOL_SEP_IDX = 2
UNIMOL_UNK_IDX = 3
UNIMOL_DICT_SIZE = 31  # 30 tokens + [MASK]

# Target normalization for regression tasks (from Uni-Mol official)
TARGET_NORM = {
    'esol': {'mean': -3.0501019503546094, 'std': 2.096441210089345},
    'freesolv': {'mean': -3.8030062305295944, 'std': 3.8478201171088138},
    'lipophilicity': {'mean': 2.186336, 'std': 1.203004},
}


@dataclass
class MolBatch:
    """Molecular batch data for both sparse (GNN) and dense (Transformer) formats."""
    # Sparse graph format
    x: torch.Tensor              # [total_atoms, atom_dim]
    edge_index: torch.Tensor     # [2, total_edges]
    edge_attr: torch.Tensor      # [total_edges, edge_dim]
    batch_idx: torch.Tensor      # [total_atoms] graph assignment

    # Dense format (Uni-Mol pipeline: atom tokens, coordinates, distances, edge types)
    atom_features: torch.Tensor  # [B, max_atoms, atom_dim]
    positions: torch.Tensor      # [B, max_atoms, 3]
    dist_matrix: torch.Tensor    # [B, max_atoms, max_atoms]
    mask: torch.Tensor           # [B, max_atoms] boolean

    # Uni-Mol specific
    atom_tokens: torch.Tensor    # [B, max_atoms] Uni-Mol vocabulary token ids
    edge_types: torch.Tensor     # [B, max_atoms, max_atoms] atom-pair type ids

    # Targets
    targets: torch.Tensor        # [B, num_tasks]
    target_mask: torch.Tensor    # [B, num_tasks] for missing labels


# =====================================================================
# LMDB data loading (official Uni-Mol pre-split data)
# =====================================================================

class LMDBReader:
    """Lazy LMDB reader — opens the environment on first access."""

    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        assert os.path.isfile(lmdb_path), f"LMDB not found: {lmdb_path}"
        env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False,
                        readahead=False, meminit=False, max_readers=256)
        with env.begin() as txn:
            self._len = len(list(txn.cursor().iternext(values=False)))
        env.close()
        self._env = None

    def _connect(self):
        if self._env is None:
            self._env = lmdb.open(self.lmdb_path, subdir=False, readonly=True,
                                  lock=False, readahead=False, meminit=False,
                                  max_readers=256)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        self._connect()
        data = self._env.begin().get(f"{idx}".encode("ascii"))
        return pickle.loads(data)


# Map from our dataset names to the official Uni-Mol directory names
DATASET_LMDB_NAME = {
    'bbbp': 'bbbp',
    'bace': 'bace',
    'tox21': 'tox21',
    'esol': 'esol',
    'freesolv': 'freesolv',
    'lipophilicity': 'lipo',
}

DATASET_CONFIG = {
    'bbbp': {
        'target_key': 'target',
        'num_tasks': 1,
        'task_type': 'classification',
    },
    'bace': {
        'target_key': 'target',
        'num_tasks': 1,
        'task_type': 'classification',
    },
    'tox21': {
        'target_key': 'target',
        'num_tasks': 12,
        'task_type': 'classification',
    },
    'esol': {
        'target_key': 'target',
        'num_tasks': 1,
        'task_type': 'regression',
    },
    'freesolv': {
        'target_key': 'target',
        'num_tasks': 1,
        'task_type': 'regression',
    },
    'lipophilicity': {
        'target_key': 'target',
        'num_tasks': 1,
        'task_type': 'regression',
    },
}


def _remove_polar_hydrogen(atoms, coordinates):
    """Remove trailing polar hydrogen atoms (matches Uni-Mol only_polar=1 mode)."""
    end_idx = 0
    for i, atom in enumerate(atoms[::-1]):
        if atom != 'H':
            break
        else:
            end_idx = i + 1
    if end_idx != 0:
        atoms = atoms[:-end_idx]
        coordinates = coordinates[:-end_idx]
    return atoms, coordinates


def _tokenize_atoms(atom_symbols):
    """Convert atom element symbols to Uni-Mol vocabulary token ids.
    Prepend [CLS] and append [SEP]."""
    tokens = [UNIMOL_CLS_IDX]
    for sym in atom_symbols:
        tokens.append(UNIMOL_ELEM_TO_IDX.get(sym, UNIMOL_UNK_IDX))
    tokens.append(UNIMOL_SEP_IDX)
    return tokens


class MoleculeDataset(Dataset):
    """Dataset for molecular property prediction.
    Reads directly from LMDB using the Uni-Mol pipeline:
    atom symbols + multi-conformer coordinates.

    Training: randomly sample 1 conformer per molecule.
    Val/Test (TTA): enumerate all conformers; dataset length = N * conf_size.
    """

    def __init__(self, lmdb_reader, num_tasks, dataset_name, seed=42,
                 is_train=True, conf_size=11, target_mean=None, target_std=None):
        self.lmdb_reader = lmdb_reader
        self.num_tasks = num_tasks
        self.dataset_name = dataset_name
        self.seed = seed
        self.is_train = is_train
        self.conf_size = conf_size
        self.target_mean = target_mean  # for regression normalization
        self.target_std = target_std
        self.n_molecules = len(lmdb_reader)

    def __len__(self):
        if self.is_train:
            return self.n_molecules
        else:
            # TTA: each molecule expanded to conf_size entries
            return self.n_molecules * self.conf_size

    def _get_entry_and_conf_idx(self, idx):
        """Return (LMDB entry, conformer index)."""
        if self.is_train:
            entry = self.lmdb_reader[idx]
            n_confs = len(entry.get('coordinates', []))
            # Sample a different conformer each epoch (matches reference
            # ConformerSampleDataset which seeds with (seed, epoch, idx))
            epoch = getattr(self, '_epoch', 0)
            rng = np.random.RandomState(hash((self.seed, epoch, idx)) & 0xFFFFFFFF)
            conf_idx = rng.randint(max(n_confs, 1)) if n_confs > 0 else 0
            return entry, conf_idx
        else:
            mol_idx = idx // self.conf_size
            conf_idx = idx % self.conf_size
            entry = self.lmdb_reader[mol_idx]
            n_confs = len(entry.get('coordinates', []))
            # Wrap around if conf_idx >= n_confs
            if n_confs > 0:
                conf_idx = conf_idx % n_confs
            else:
                conf_idx = 0
            return entry, conf_idx

    def set_epoch(self, epoch):
        """Update epoch so training conformer sampling varies per epoch (matches reference)."""
        self._epoch = int(epoch)

    def __getitem__(self, idx):
        entry, conf_idx = self._get_entry_and_conf_idx(idx)

        # Extract atoms and coordinates from LMDB entry
        atoms = np.array(entry.get('atoms', []))
        coordinates_list = entry.get('coordinates', [])

        if len(coordinates_list) > 0 and len(atoms) > 0:
            coordinates = np.array(coordinates_list[conf_idx], dtype=np.float32)
        else:
            coordinates = np.zeros((max(len(atoms), 1), 3), dtype=np.float32)

        # Remove polar hydrogens (matching Uni-Mol only_polar=1)
        if len(atoms) > 0:
            atoms, coordinates = _remove_polar_hydrogen(atoms, coordinates)

        # Normalize coordinates (center to mean)
        if len(coordinates) > 0:
            coordinates = coordinates - coordinates.mean(axis=0)

        # Tokenize atoms using Uni-Mol vocabulary (with [CLS] and [SEP])
        tokens = _tokenize_atoms(atoms)  # length = n_atoms + 2

        # Build extended coordinates with zeros for [CLS] and [SEP]
        n_atoms = len(atoms)
        ext_coords = np.zeros((n_atoms + 2, 3), dtype=np.float32)
        ext_coords[1:n_atoms + 1] = coordinates

        # Compute distance matrix on extended coordinates
        dist = scipy_distance_matrix(ext_coords, ext_coords).astype(np.float32)

        # Compute edge types: token_i * DICT_SIZE + token_j
        tok_arr = np.array(tokens, dtype=np.int64)
        edge_type = tok_arr[:, None] * UNIMOL_DICT_SIZE + tok_arr[None, :]

        # Parse target
        target = entry.get('target', None)
        if target is None:
            t = [0.0] * self.num_tasks
            m = [0.0] * self.num_tasks
        elif isinstance(target, (list, tuple, np.ndarray)):
            t, m = [], []
            for val in target:
                if val is None or (isinstance(val, float) and np.isnan(val)) or val == -1:
                    t.append(0.0)
                    m.append(0.0)
                else:
                    t.append(float(val))
                    m.append(1.0)
        else:
            if target is None or (isinstance(target, float) and np.isnan(target)) or target == -1:
                t = [0.0]
                m = [0.0]
            else:
                t = [float(target)]
                m = [1.0]
        while len(t) < self.num_tasks:
            t.append(0.0)
            m.append(0.0)
        t = t[:self.num_tasks]
        m = m[:self.num_tasks]

        # Apply target normalization for regression tasks
        if self.target_mean is not None and self.target_std is not None:
            t_norm = []
            for i, (val, mask_val) in enumerate(zip(t, m)):
                if mask_val > 0.5:
                    t_norm.append((val - self.target_mean[i]) / self.target_std[i])
                else:
                    t_norm.append(0.0)
            t = t_norm

        # Also build GNN features from SMILES for GNN-based models
        smi = entry.get('smi', '')
        gnn_feats = self._build_gnn_features(smi, atoms, coordinates)

        return {
            # GNN sparse format
            'atom_feats': gnn_feats['atom_feats'],
            'edge_index': gnn_feats['edge_index'],
            'edge_attr': gnn_feats['edge_attr'],
            'positions': torch.from_numpy(coordinates) if len(coordinates) > 0 else torch.zeros(1, 3),
            'num_atoms': gnn_feats['num_atoms'],
            # Uni-Mol format
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'ext_coords': torch.from_numpy(ext_coords),
            'dist_matrix': torch.from_numpy(dist),
            'edge_types': torch.from_numpy(edge_type),
            'num_tokens': len(tokens),
            # Targets
            'targets': torch.tensor(t, dtype=torch.float32),
            'target_mask': torch.tensor(m, dtype=torch.float32),
            # SMILES (passed through for models that need molecule-level features)
            'smiles': smi,
            # Molecule index for TTA aggregation
            'mol_idx': idx if self.is_train else idx // self.conf_size,
        }

    def _build_gnn_features(self, smi, atoms_arr, coordinates):
        """Build GNN (sparse graph) features from SMILES for GNN-based models."""
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            return {
                'atom_feats': torch.zeros(1, ATOM_DIM),
                'edge_index': torch.zeros(2, 0, dtype=torch.long),
                'edge_attr': torch.zeros(0, EDGE_DIM),
                'num_atoms': 1,
            }

        atom_feats_list = []
        for atom in mol.GetAtoms():
            atom_feats_list.append(atom_features(atom))
        atom_feats_t = torch.tensor(atom_feats_list, dtype=torch.float32)

        edge_indices = []
        edge_feats = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = bond_features(bond)
            edge_indices.extend([[i, j], [j, i]])
            edge_feats.extend([bf, bf])

        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, EDGE_DIM, dtype=torch.float32)

        return {
            'atom_feats': atom_feats_t,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'num_atoms': atom_feats_t.size(0),
        }


def collate_mols(batch_list):
    """Collate variable-size molecular graphs into MolBatch."""
    atom_feats_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_idx_list = []
    positions_list = []
    targets_list = []
    target_mask_list = []

    atom_offset = 0
    max_atoms = max(b['num_atoms'] for b in batch_list)
    max_tokens = max(b['num_tokens'] for b in batch_list)
    B = len(batch_list)

    # Dense tensors for GNN
    dense_atoms = torch.zeros(B, max_atoms, ATOM_DIM)
    dense_pos = torch.zeros(B, max_atoms, 3)
    dense_mask = torch.zeros(B, max_atoms)

    # Dense tensors for Uni-Mol
    tokens_padded = torch.full((B, max_tokens), UNIMOL_PAD_IDX, dtype=torch.long)
    dist_padded = torch.zeros(B, max_tokens, max_tokens)
    edge_types_padded = torch.zeros(B, max_tokens, max_tokens, dtype=torch.long)
    token_mask = torch.zeros(B, max_tokens)

    for i, b in enumerate(batch_list):
        n = b['num_atoms']
        nt = b['num_tokens']

        atom_feats_list.append(b['atom_feats'])
        positions_list.append(b['positions'])

        if b['edge_index'].size(1) > 0:
            edge_index_list.append(b['edge_index'] + atom_offset)
            edge_attr_list.append(b['edge_attr'])

        batch_idx_list.append(torch.full((n,), i, dtype=torch.long))

        # Dense format for GNN
        dense_atoms[i, :n] = b['atom_feats']
        pos = b['positions']
        if pos.size(0) <= max_atoms:
            dense_pos[i, :pos.size(0)] = pos
        dense_mask[i, :n] = 1.0

        # Dense format for Uni-Mol
        tokens_padded[i, :nt] = b['tokens']
        dist_padded[i, :nt, :nt] = b['dist_matrix']
        edge_types_padded[i, :nt, :nt] = b['edge_types']
        token_mask[i, :nt] = 1.0

        targets_list.append(b['targets'])
        target_mask_list.append(b['target_mask'])
        atom_offset += n

    # Build sparse tensors
    x = torch.cat(atom_feats_list, dim=0)
    batch_idx = torch.cat(batch_idx_list, dim=0)

    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, EDGE_DIM)

    # Distance matrix for dense GNN format
    diff = dense_pos.unsqueeze(2) - dense_pos.unsqueeze(1)
    gnn_dist_matrix = torch.sqrt((diff ** 2).sum(-1) + 1e-8)

    targets = torch.stack(targets_list, dim=0)
    target_mask = torch.stack(target_mask_list, dim=0)

    return MolBatch(
        x=x, edge_index=edge_index, edge_attr=edge_attr, batch_idx=batch_idx,
        atom_features=dense_atoms, positions=dense_pos,
        dist_matrix=gnn_dist_matrix, mask=dense_mask,
        atom_tokens=tokens_padded, edge_types=edge_types_padded,
        targets=targets, target_mask=target_mask,
    ), dist_padded, token_mask


def collate_mols_wrapper(batch_list):
    """Wrapper that stores extra tensors inside MolBatch for access."""
    mol_batch, dist_padded, token_mask = collate_mols(batch_list)
    # Store Uni-Mol distance and token mask as extra attributes
    mol_batch._unimol_dist = dist_padded
    mol_batch._unimol_token_mask = token_mask
    mol_batch._mol_indices = torch.tensor([b['mol_idx'] for b in batch_list], dtype=torch.long)
    # SMILES list for models that compute molecule-level features (e.g. RDKit descriptors)
    mol_batch._smiles = [b.get('smiles', '') for b in batch_list]
    return mol_batch


def load_dataset_splits(dataset_name, data_dir, seed=42, conf_size=11):
    """Load pre-split train/valid/test data from official Uni-Mol LMDB files.

    Args:
        dataset_name: one of bbbp, bace, tox21, esol, freesolv, lipophilicity
        data_dir: path to the molecular_property_prediction directory
        seed: random seed for conformer sampling
        conf_size: number of conformers for TTA (val/test)

    Returns:
        dict of MoleculeDataset for train/valid/test, plus task_type and num_tasks
    """
    config = DATASET_CONFIG[dataset_name]
    lmdb_name = DATASET_LMDB_NAME[dataset_name]
    num_tasks = config['num_tasks']
    task_type = config['task_type']

    # Target normalization for regression tasks
    target_mean = None
    target_std = None
    if dataset_name in TARGET_NORM:
        norm = TARGET_NORM[dataset_name]
        target_mean = [norm['mean']] if not isinstance(norm['mean'], list) else norm['mean']
        target_std = [norm['std']] if not isinstance(norm['std'], list) else norm['std']

    datasets = {}
    for split in ['train', 'valid', 'test']:
        lmdb_path = os.path.join(data_dir, lmdb_name, f'{split}.lmdb')
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB file not found: {lmdb_path}")
        reader = LMDBReader(lmdb_path)
        is_train = (split == 'train')
        datasets[split] = MoleculeDataset(
            lmdb_reader=reader,
            num_tasks=num_tasks,
            dataset_name=dataset_name,
            seed=seed,
            is_train=is_train,
            conf_size=conf_size,
            target_mean=target_mean if task_type == 'regression' else None,
            target_std=target_std if task_type == 'regression' else None,
        )

    return datasets, task_type, num_tasks


# =====================================================================
# Training and evaluation
# =====================================================================

def train_epoch(model, loader, optimizer, task_type, device, scheduler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad()

        preds = model(batch)

        if task_type == 'classification':
            bce = F.binary_cross_entropy_with_logits(
                preds, batch.targets, reduction='none',
            )
            loss = (bce * batch.target_mask).sum() / batch.target_mask.sum().clamp(min=1)
        else:
            diff = (preds - batch.targets) ** 2
            loss = (diff * batch.target_mask).sum() / batch.target_mask.sum().clamp(min=1)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, task_type, device, dataset_name=None, is_tta=True, conf_size=11):
    """Evaluate model. For TTA (val/test): average predictions over conformers per molecule."""
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []
    all_mol_indices = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        preds = model(batch)
        all_preds.append(preds.cpu())
        all_targets.append(batch.targets.cpu())
        all_masks.append(batch.target_mask.cpu())
        if hasattr(batch, '_mol_indices'):
            all_mol_indices.append(batch._mol_indices.cpu())

    if not all_preds:
        return (0.0, 'rocauc') if task_type == 'classification' else (float('inf'), 'rmse')

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)

    # TTA aggregation: average predictions over conformers per molecule
    if is_tta and all_mol_indices:
        mol_indices = torch.cat(all_mol_indices, dim=0)
        unique_mols = mol_indices.unique(sorted=True)
        agg_preds = []
        agg_targets = []
        agg_masks = []
        for mol_id in unique_mols:
            sel = mol_indices == mol_id
            agg_preds.append(preds[sel].mean(dim=0))
            agg_targets.append(targets[sel][0])  # targets same for all conformers
            agg_masks.append(masks[sel][0])
        preds = torch.stack(agg_preds, dim=0)
        targets = torch.stack(agg_targets, dim=0)
        masks = torch.stack(agg_masks, dim=0)

    # Denormalize predictions for regression tasks before computing RMSE
    if task_type == 'regression' and dataset_name in TARGET_NORM:
        norm = TARGET_NORM[dataset_name]
        mean = norm['mean'] if isinstance(norm['mean'], list) else [norm['mean']]
        std = norm['std'] if isinstance(norm['std'], list) else [norm['std']]
        mean_t = torch.tensor(mean, dtype=preds.dtype)
        std_t = torch.tensor(std, dtype=preds.dtype)
        preds = preds * std_t + mean_t
        targets = targets * std_t + mean_t

    if task_type == 'classification':
        from sklearn.metrics import roc_auc_score
        scores = []
        for t in range(preds.size(1)):
            valid = masks[:, t] > 0
            if valid.sum() < 2:
                continue
            y_true = targets[valid, t].numpy()
            y_score = torch.sigmoid(preds[valid, t]).numpy()
            if len(np.unique(y_true)) < 2:
                continue
            try:
                scores.append(roc_auc_score(y_true, y_score))
            except ValueError:
                continue
        metric = float(np.mean(scores)) if scores else 0.0
        return metric, 'rocauc'
    else:
        diff_sq = ((preds - targets) ** 2 * masks).sum() / masks.sum().clamp(min=1)
        rmse = float(torch.sqrt(diff_sq))
        return rmse, 'rmse'


def batch_to_device(batch, device):
    new_batch = MolBatch(
        x=batch.x.to(device),
        edge_index=batch.edge_index.to(device),
        edge_attr=batch.edge_attr.to(device),
        batch_idx=batch.batch_idx.to(device),
        atom_features=batch.atom_features.to(device),
        positions=batch.positions.to(device),
        dist_matrix=batch.dist_matrix.to(device),
        mask=batch.mask.to(device),
        atom_tokens=batch.atom_tokens.to(device),
        edge_types=batch.edge_types.to(device),
        targets=batch.targets.to(device),
        target_mask=batch.target_mask.to(device),
    )
    # Transfer extra attributes
    if hasattr(batch, '_unimol_dist'):
        new_batch._unimol_dist = batch._unimol_dist.to(device)
    if hasattr(batch, '_unimol_token_mask'):
        new_batch._unimol_token_mask = batch._unimol_token_mask.to(device)
    if hasattr(batch, '_mol_indices'):
        new_batch._mol_indices = batch._mol_indices
    if hasattr(batch, '_smiles'):
        new_batch._smiles = batch._smiles
    return new_batch


def load_pretrained_weights(model, ckpt_path):
    """Load pretrained weights with detailed debugging output.
    Prints number of loaded keys and names of keys that failed to load.
    """
    if not os.path.exists(ckpt_path):
        print(f"[Checkpoint] Pretrained weights not found at {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)

    own_state = model.state_dict()
    loaded_keys = []
    missing_keys = []
    shape_mismatch_keys = []

    for key, val in state.items():
        if key in own_state:
            if own_state[key].shape == val.shape:
                own_state[key].copy_(val)
                loaded_keys.append(key)
            else:
                shape_mismatch_keys.append(
                    f"  {key}: ckpt={list(val.shape)} vs model={list(own_state[key].shape)}")
        else:
            missing_keys.append(key)

    model.load_state_dict(own_state, strict=False)

    print(f"[Checkpoint] Successfully loaded {len(loaded_keys)} keys")
    if shape_mismatch_keys:
        print(f"[Checkpoint] Shape mismatch ({len(shape_mismatch_keys)} keys):")
        for s in shape_mismatch_keys[:20]:
            print(s)
    if missing_keys:
        print(f"[Checkpoint] Missing in model ({len(missing_keys)} keys):")
        for k in missing_keys[:20]:
            print(f"  {k}")
    not_loaded = [k for k in own_state if k not in state]
    if not_loaded:
        print(f"[Checkpoint] Not in checkpoint ({len(not_loaded)} keys):")
        for k in not_loaded[:20]:
            print(f"  {k}")


def train_and_evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pre-split data from official LMDB files
    datasets, task_type, num_tasks = load_dataset_splits(
        args.dataset, args.data_dir, seed=args.seed, conf_size=11
    )
    train_ds = datasets['train']
    val_ds = datasets['valid']
    test_ds = datasets['test']

    print(f"Dataset: {args.dataset}, type: {task_type}, tasks: {num_tasks}")
    print(f"Split: train={train_ds.n_molecules}, val={val_ds.n_molecules}, test={test_ds.n_molecules}")
    print(f"TTA conf_size: {val_ds.conf_size} (val/test datasets expanded)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_mols_wrapper, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_mols_wrapper, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_mols_wrapper, num_workers=2)

    # Model — baseline implementations may honor `pooler_dropout` attr on the
    # class (e.g. Uni-Mol baseline) to match reference per-dataset settings.
    MoleculeModel.pooler_dropout = args.pooler_dropout
    model = MoleculeModel(
        atom_dim=ATOM_DIM,
        edge_dim=EDGE_DIM,
        num_tasks=num_tasks,
        task_type=task_type,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer: AdamW with betas matching Uni-Mol reference (eps=1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.99), eps=1e-6,
                                  weight_decay=1e-5)

    # Polynomial decay with linear warmup (Uni-Mol reference scheduler).
    # Steps computed from loader length * epochs.
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = max(steps_per_epoch * args.epochs, 1)
    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        # Polynomial decay with power=1.0 to near-zero over remaining steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        return max(1.0 - progress, 0.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training with early stopping
    best_val_metric = None
    best_epoch = 0
    patience_counter = 0
    patience = 20

    for epoch in range(1, args.epochs + 1):
        # Update training set epoch so per-molecule conformer choice varies
        if hasattr(train_ds, 'set_epoch'):
            train_ds.set_epoch(epoch)
        train_loss = train_epoch(model, train_loader, optimizer, task_type, device,
                                 scheduler=scheduler)

        val_metric, metric_name = evaluate(
            model, val_loader, task_type, device,
            dataset_name=args.dataset, is_tta=True, conf_size=11)

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"TRAIN_METRICS epoch={epoch} loss={train_loss:.6f} lr={cur_lr:.2e} val_{metric_name}={val_metric:.6f}")

        # Early stopping logic
        improved = False
        if best_val_metric is None:
            improved = True
        elif task_type == 'classification' and val_metric > best_val_metric:
            improved = True
        elif task_type == 'regression' and val_metric < best_val_metric:
            improved = True

        if improved:
            best_val_metric = val_metric
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt'), weights_only=True))
    test_metric, metric_name = evaluate(
        model, test_loader, task_type, device,
        dataset_name=args.dataset, is_tta=True, conf_size=11)
    print(f"TEST_METRICS {metric_name}={test_metric:.6f}")
    print(f"Best val {metric_name}: {best_val_metric:.6f} at epoch {best_epoch}")


def main():
    parser = argparse.ArgumentParser(description="Molecular Property Prediction")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['bbbp', 'bace', 'tox21', 'esol', 'freesolv', 'lipophilicity'])
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to molecular_property_prediction directory')
    parser.add_argument('--task-type', type=str, default=None,
                        help='Override task type (classification/regression)')
    parser.add_argument('--num-tasks', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--warmup-ratio', type=float, default=0.0,
                        help='Linear warmup fraction of total training steps')
    parser.add_argument('--pooler-dropout', type=float, default=0.0,
                        help='Dropout on CLS pooler features (Uni-Mol style)')
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_and_evaluate(args)


if __name__ == '__main__':
    main()
