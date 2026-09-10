# Task: Molecular Property Prediction

## Research Question
Design a molecular representation model for predicting chemical properties (toxicity, blood-brain barrier penetration, enzyme inhibition, etc.) from molecular structure. The goal is to learn effective molecular representations that generalize across diverse property prediction tasks.

## Background
Molecular property prediction is a core task in drug discovery and materials science. Given a molecule (as a SMILES string → molecular graph + optional 3D coordinates), the model must predict one or more chemical properties. Key challenges include:
- **Molecular representation**: How to encode atoms, bonds, and 3D geometry into informative features.
- **Multi-task learning**: Some datasets have multiple targets with missing labels (e.g., Tox21 has 12 assays).
- **Scaffold generalization**: The scaffold split ensures the model generalizes to structurally novel molecules.

Existing approaches include:
- **D-MPNN** (Yang et al., "Analyzing Learned Molecular Representations for Property Prediction", J. Chem. Inf. Model. 2019, 59(8):3370–3388; arXiv:1904.01561). Directed message passing on bonds rather than atoms to avoid "message collision". Reference implementation: Chemprop (https://github.com/chemprop/chemprop).
- **GIN** (Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019; arXiv:1810.00826). Graph Isomorphism Network with sum aggregation that matches the discriminative power of the Weisfeiler–Lehman test.
- **Uni-Mol** (Zhou et al., "Uni-Mol: A Universal 3D Molecular Representation Learning Framework", ICLR 2023; OpenReview 6K2RM6wVqKu; ChemRxiv 628e5b4d5d948517f5ce6d72). SE(3)-invariant Transformer with 3D distance attention bias, pretrained on ~209M molecular conformations. Code: https://github.com/deepmodeling/Uni-Mol.

## What to Implement
Implement the `MoleculeModel` class in `custom_molprop.py`. You must implement:
1. `__init__(self, atom_dim, edge_dim, num_tasks, task_type)`: Set up your model architecture.
2. `forward(self, batch) -> Tensor`: Return predictions of shape `[B, num_tasks]`.

## Batch Format (MolBatch)
```python
@dataclass
class MolBatch:
    # Sparse graph format (for GNN models)
    x: Tensor              # [total_atoms, atom_dim] node features
    edge_index: Tensor     # [2, total_edges] COO format
    edge_attr: Tensor      # [total_edges, edge_dim] bond features
    batch_idx: Tensor      # [total_atoms] graph assignment (0..B-1)

    # Dense format (for Transformer models)
    atom_features: Tensor  # [B, max_atoms, atom_dim] zero-padded
    positions: Tensor      # [B, max_atoms, 3] 3D coordinates
    dist_matrix: Tensor    # [B, max_atoms, max_atoms] pairwise distances
    mask: Tensor           # [B, max_atoms] 1=real atom, 0=padding

    # Uni-Mol specific (from LMDB pipeline)
    atom_tokens: Tensor    # [B, max_tokens] Uni-Mol vocabulary token ids (with [CLS]/[SEP])
    edge_types: Tensor     # [B, max_tokens, max_tokens] atom-pair type ids

    # Targets (normalized for regression tasks)
    targets: Tensor        # [B, num_tasks]
    target_mask: Tensor    # [B, num_tasks] 1=valid label, 0=missing
```

Additional attributes set dynamically on the batch:
- `batch._unimol_dist`: [B, max_tokens, max_tokens] distance matrix for Uni-Mol tokens.
- `batch._unimol_token_mask`: [B, max_tokens] 1=valid token, 0=padding.

## Atom Features (`ATOM_DIM = 136`)
One-hot encodings of: atomic_num (118), degree (6), formal_charge (5), num_Hs (5), hybridization (5), aromatic (1), in_ring (1).

## Bond Features (`EDGE_DIM = 9`)
One-hot encodings of: bond_type (4), stereo (3), conjugated (1), in_ring (1).

## Fixed Pipeline
SMILES preprocessing, conformer generation, scaffold splitting, training loop, optimizer schedule, normalization for regression targets, masked loss for missing labels, and test-time augmentation are all fixed by the scaffold.

## Evaluation
The model is tested on three MoleculeNet classification benchmarks with scaffold splits (metric: ROC-AUC, higher is better):
- **BBBP**: Blood-brain barrier penetration (2,039 molecules, 1 task).
- **BACE**: Beta-secretase 1 inhibition (1,513 molecules, 1 task).
- **Tox21**: Toxicity across 12 assays (7,831 molecules, 12 tasks, multi-task with missing labels).

ROC-AUC is averaged over valid labels per task and across tasks.

## Editable Region
The section between `EDITABLE SECTION START` and `EDITABLE SECTION END` markers in `custom_molprop.py` is editable. You may define helper classes, layers, or functions within this region. The region must contain a `MoleculeModel` class with the specified interface.

## Available Resources
- 3D conformers from LMDB (Uni-Mol pipeline: coordinates normalized, polar H removed).
- Uni-Mol vocabulary tokens and edge types available in batch.
- Uni-Mol pre-trained weights available inside the container at the path used by the `unimol` baseline.
- Test-time augmentation: predictions averaged over 11 conformers at val/test time.
