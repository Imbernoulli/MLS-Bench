# MLS-Bench: ai4sci-vs-contrastive-scoring

# Task: Scoring Objective Design for Virtual Screening

## Research Question
Design the scoring objective — including projection heads, embedding space, and training loss — for contrastive protein-ligand virtual screening. Given pretrained backbone encoders (Uni-Mol for molecules/pockets, ESM-2 for protein sequences) that are fine-tuned jointly end-to-end with the scoring module, how should their features be projected, embedded, and trained to best discriminate active binders from decoys?

## Background
Virtual screening computationally ranks large compound libraries against a protein target to identify potential drug candidates. Modern approaches use learned representations: encode protein pockets and molecules into a shared embedding space, then rank by similarity. Key design choices include:

- **Projection heads**: How to project backbone features (512-dim Uni-Mol, 480-dim ESM-2) into a shared space.
- **Embedding geometry**: Euclidean (L2-normalized dot product), hyperbolic (Lorentz hyperboloid), spherical, or other manifolds.
- **Training loss**: In-batch contrastive (CLIP-style), ranking-aware losses, activity-dependent constraints, cone hierarchy.

Existing approaches range from simple CLIP-style contrastive learning to hyperbolic geometry with cone hierarchy constraints:

- **DrugCLIP** (Gao et al., "DrugCLIP: Contrastive Protein-Molecule Representation Learning for Virtual Screening", NeurIPS 2023; arXiv:2310.06367). CLIP-style symmetric in-batch contrastive loss between pocket and molecule embeddings. Code: https://github.com/bowen-gao/DrugCLIP.
- **HypSeek** (Wang et al., "Learning Protein-Ligand Binding in Hyperbolic Space", AAAI 2026; arXiv:2508.15480). Three-tower model (pocket, ligand, protein sequence) embedded in Lorentz hyperbolic space, trained with a hierarchical contrastive constraint (HCC) loss and an entailment-cone hierarchy regularizer. Code: https://github.com/jianhuiwemi/HypSeek.

## Reference Baselines
- **vanilla_clip**: DrugCLIP-style CLIP contrastive scoring. Euclidean L2-normalized embeddings with symmetric in-batch softmax contrastive loss between pocket and molecule representations.
- **hcc**: HypSeek HCC loss in Euclidean space. Adds an activity-aware ranking loss on top of the vanilla contrastive objective; embeddings remain in Euclidean space.
- **hcc_hyp_cone**: Full HypSeek — Lorentz hyperboloid embeddings with learnable curvature, HCC contrastive ranking loss, and an entailment-cone hierarchy regularizer (AAAI 2026).

Backbone references: Uni-Mol (Zhou et al., ICLR 2023, OpenReview 6K2RM6wVqKu) and ESM-2 (Lin et al., Science 2023, "Evolutionary-scale prediction of atomic-level protein structure with a language model").

## What to Implement
Implement the `CustomScoring` class in `custom_scoring.py`. You must implement:
1. `__init__`: Define projection heads, embedding parameters, loss hyperparameters.
2. `project_mol(mol_feat)`: Project molecule features `[B, 512]` → `[B, embed_dim]`.
3. `project_pocket(poc_feat)`: Project pocket features `[B, 512]` → `[B, embed_dim]`.
4. `project_protein(prot_feat)`: Project protein features `[B, 480]` → `[B, embed_dim]`.
5. `compute_loss(mol_emb, poc_emb, prot_emb, batch_list, act_list, ...)`: Training loss.
6. `score(mol_reps, pocket_reps, prot_reps)`: Evaluation scoring (numpy arrays).

## Available Components
- Backbone features (fine-tuned jointly): `mol_feat` `[B, 512]`, `poc_feat` `[B, 512]`, `prot_feat` `[B, 480]`.
- Lorentz hyperbolic operations: `exp_map0`, `pairwise_dist`, `half_aperture`, `oxy_angle` from `unimol.losses.lorentz`.
- Training data provides: `batch_list` (pocket→ligand mapping), `act_list` (pIC50 activities), `uniprot_poc/mol` (for false-negative masking), `pocket_lig_smiles/lig_smiles` (for duplicate masking).

## Fixed Pipeline
The backbone encoders, data loaders, training loop, and evaluation scripts are fixed. Backbone parameters are loaded from pretrained weights and fine-tuned jointly with the scoring module.

## Evaluation
The model is evaluated on three virtual screening benchmarks (zero-shot, no target-specific training):
1. **DUD-E** (102 targets): Active compounds vs property-matched decoys.
2. **LIT-PCBA** (15 targets): Realistic screening with confirmed actives/inactives.
3. **DEKOIS 2.0** (81 targets): Challenging decoy benchmark.

Metrics (averaged across targets): **AUROC**, **BEDROC** (α=80.5), **EF** at 0.5%/1%/5%. Higher is better for all of them.

## Editable Region
The entire `custom_scoring.py` file is editable. You may define any helper classes or functions within this file. The backbone encoders and training loop are fixed; backbone parameters are loaded from pretrained weights and fine-tuned jointly with the scoring module.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/HypSeek/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `HypSeek/unimol/custom_scoring.py`
- editable: **entire file**


Other files you may **read** for context (do not modify):
- `HypSeek/unimol/models/custom_vs_model.py`
- `HypSeek/unimol/losses/custom_vs_loss.py`
- `HypSeek/unimol/losses/lorentz.py`


## Readable Context


### `HypSeek/unimol/custom_scoring.py`  [EDITABLE — entire file only]

```python
     1: """Custom scoring module for contrastive virtual screening.
     2: 
     3: This module defines the projection heads, embedding space mapping, and
     4: training loss for protein-ligand virtual screening.
     5: 
     6: Interface:
     7:     project_mol(mol_feat)      -> [B, embed_dim]  molecule embeddings
     8:     project_pocket(poc_feat)   -> [B, embed_dim]  pocket embeddings
     9:     project_protein(prot_feat) -> [B, embed_dim]  protein embeddings
    10:     compute_loss(mol_emb, poc_emb, prot_emb, ...) -> (loss, dict)
    11:     score(mol_reps, pocket_reps, prot_reps) -> [N_mol] numpy scores
    12: 
    13: Available utilities (imported in model wrapper):
    14:     torch, torch.nn, torch.nn.functional, numpy, math
    15:     lorentz ops: from unimol.losses.lorentz import exp_map0, pairwise_dist,
    16:                  half_aperture, oxy_angle, pairwise_inner, minkowski_dot
    17: 
    18: Backbone encoder outputs (frozen, not editable):
    19:     mol_feat:  [B, 512]  CLS token from UniMol molecule encoder
    20:     poc_feat:  [B, 512]  CLS token from UniMol pocket encoder
    21:     prot_feat: [B, 480]  CLS token from ESM2 protein sequence encoder
    22: """
    23: 
    24: import math
    25: import numpy as np
    26: import torch
    27: import torch.nn as nn
    28: import torch.nn.functional as F
    29: 
    30: 
    31: class CustomScoring(nn.Module):
    32:     """Scoring module for contrastive protein-ligand virtual screening.
    33: 
    34:     Handles projection of frozen encoder features into a shared embedding
    35:     space and computes the training loss for ranking actives above decoys.
    36:     """
    37: 
    38:     def __init__(self, mol_dim=512, pocket_dim=512, protein_dim=480, embed_dim=128):
    39:         super().__init__()
    40:         # Projection heads following paper's NonLinearHead pattern
    41:         # (vendor/external_packages/HypSeek/unimol/models/unimol.py:345-360):
    42:         # hidden=input_dim, i.e. Linear(in,in) -> ReLU -> Linear(in,embed_dim).
    43:         self.mol_project = nn.Sequential(
    44:             nn.Linear(mol_dim, mol_dim), nn.ReLU(), nn.Linear(mol_dim, embed_dim)
    45:         )
    46:         self.pocket_project = nn.Sequential(
    47:             nn.Linear(pocket_dim, pocket_dim), nn.ReLU(), nn.Linear(pocket_dim, embed_dim)
    48:         )
    49:         self.protein_project = nn.Sequential(
    50:             nn.Linear(protein_dim, protein_dim), nn.ReLU(), nn.Linear(protein_dim, embed_dim)
    51:         )
    52:         # Learnable temperature (log scale).
    53:         # log(13) matches the paper's three_hybrid_model.py:58 — see
    54:         # vendor/external_packages/HypSeek/unimol/models/three_hybrid_model.py.
    55:         self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(13))
    56: 
    57:     def project_mol(self, mol_feat):
    58:         """Project molecule encoder features to embedding space."""
    59:         return F.normalize(self.mol_project(mol_feat), dim=-1)
    60: 
    61:     def project_pocket(self, poc_feat):
    62:         """Project pocket encoder features to embedding space."""
    63:         return F.normalize(self.pocket_project(poc_feat), dim=-1)
    64: 
    65:     def project_protein(self, prot_feat):
    66:         """Project protein encoder features to embedding space."""
    67:         return F.normalize(self.protein_project(prot_feat), dim=-1)
    68: 
    69:     def compute_loss(self, mol_emb, poc_emb, prot_emb,
    70:                      batch_list, act_list,
    71:                      uniprot_poc=None, uniprot_mol=None,
    72:                      pocket_lig_smiles=None, lig_smiles=None):
    73:         """Compute training loss.
    74: 
    75:         Args:
    76:             mol_emb:  [N_mol, D] molecule embeddings (all ligands in batch)
    77:             poc_emb:  [N_poc, D] pocket embeddings (one per assay)
    78:             prot_emb: [N_poc, D] protein embeddings (one per assay)
    79:             batch_list: list of (start, end) tuples mapping pocket i to its
    80:                         ligands mol_emb[start:end]
    81:             act_list:   list of activity values (pIC50) per pocket's ligands
    82:             uniprot_poc: UniProt IDs for pockets (for false-negative masking)
    83:             uniprot_mol: UniProt IDs for molecules (for false-negative masking)
    84:             pocket_lig_smiles: known ligand SMILES per pocket
    85:             lig_smiles: SMILES for each molecule in batch
    86: 
    87:         Returns:
    88:             loss: scalar training loss
    89:             log_dict: dict with loss components and sim_masked for validation
    90:         """
    91:         logit_scale = self.logit_scale.exp().detach()
    92:         B = poc_emb.size(0)
    93: 
    94:         # Similarity matrix: [N_poc, N_mol]
    95:         logits = poc_emb @ mol_emb.T * logit_scale
    96: 
    97:         # Build false-negative mask (same protein or known binder)
    98:         mask = torch.zeros_like(logits, dtype=torch.bool)
    99:         if uniprot_poc is not None and uniprot_mol is not None:
   100:             for i in range(B):
   101:                 for j in range(logits.size(1)):
   102:                     if uniprot_poc[i] == uniprot_mol[j]:
   103:                         mask[i, j] = True
   104:         if pocket_lig_smiles is not None:
   105:             for i in range(B):
   106:                 bad = pocket_lig_smiles[i]
   107:                 for j in range(logits.size(1)):
   108:                     if lig_smiles[j] in bad:
   109:                         mask[i, j] = True
   110: 
   111:         minus_inf = torch.finfo(logits.dtype).min
   112:         sim_masked = logits.masked_fill(mask, minus_inf)
   113: 
   114:         # === Symmetric contrastive loss ===
   115:         # Pocket-to-ligand: each pocket retrieves its ligands
   116:         idx2poc = []
   117:         for i, (s, e) in enumerate(batch_list):
   118:             idx2poc += [i] * (e - s)
   119:         targets = torch.tensor(idx2poc, dtype=torch.long, device=logits.device)
   120: 
   121:         lprobs_pocket = F.log_softmax(sim_masked.T, dim=-1)
   122:         loss_pocket_list = []
   123:         for i, (s, e) in enumerate(batch_list):
   124:             L_i = e - s
   125:             if L_i == 0:
   126:                 continue
   127:             rows = list(range(s, e))
   128:             lprobs_sub = lprobs_pocket[rows]
   129:             targ_sub = targets[rows]
   130:             loss_tmp = F.nll_loss(lprobs_sub, targ_sub, reduction="none")
   131:             loss_pocket_list.append(loss_tmp.sum() / math.sqrt(L_i))
   132:         loss_pocket = torch.stack(loss_pocket_list).sum() if loss_pocket_list else torch.tensor(0.0, device=logits.device)
   133: 
   134:         # Ligand-to-pocket: each ligand retrieves its pocket
   135:         loss_mol_list = []
   136:         for i in range(B):
   137:             s, e = batch_list[i]
   138:             for k in range(s, e):
   139:                 row_mask = torch.full_like(sim_masked[i], minus_inf)
   140:                 row_mask[k] = 0
   141:                 lprobs = F.log_softmax(row_mask + sim_masked[i], dim=-1)
   142:                 loss_mol_list.append(-lprobs[k] / math.sqrt(e - s))
   143:         loss_mol = torch.stack(loss_mol_list).sum() if loss_mol_list else torch.tensor(0.0, device=logits.device)
   144: 
   145:         loss = loss_pocket + loss_mol
   146: 
   147:         return loss, {
   148:             "loss": loss.item(),
   149:             "loss_pocket": loss_pocket.item(),
   150:             "loss_mol": loss_mol.item(),
   151:             "sim_masked": sim_masked,
   152:         }
   153: 
   154:     def score(self, mol_reps, pocket_reps, prot_reps=None):
   155:         """Score molecules against pocket/protein for evaluation.
   156: 
   157:         Args:
   158:             mol_reps:    [N_mol, D] numpy array of molecule embeddings
   159:             pocket_reps: [N_poc, D] numpy array of pocket embeddings
   160:             prot_reps:   [N_prot, D] numpy array of protein embeddings (optional)
   161: 
   162:         Returns:
   163:             scores: [N_mol] numpy array of final scores per molecule
   164:         """
   165:         poc_scores = (pocket_reps @ mol_reps.T).max(axis=0)
   166:         if prot_reps is not None:
   167:             prot_scores = (prot_reps @ mol_reps.T).max(axis=0)
   168:             return poc_scores + prot_scores
   169:         return poc_scores
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `hcc` baseline — editable region  [READ-ONLY — reference implementation]

In `HypSeek/unimol/custom_scoring.py`:

```python
     1: """HCC scoring module: Euclidean contrastive + ranking loss."""
     2: 
     3: import math
     4: import numpy as np
     5: import torch
     6: import torch.nn as nn
     7: import torch.nn.functional as F
     8: 
     9: 
    10: class CustomScoring(nn.Module):
    11:     """HCC: Hierarchical Contrastive Cosine in Euclidean space.
    12: 
    13:     Adds ranking loss that enforces more active ligands score higher
    14:     within each pocket's ligand set, weighted by 1/log(rank+2) (DCG-style).
    15:     """
    16: 
    17:     def __init__(self, mol_dim=512, pocket_dim=512, protein_dim=480, embed_dim=128):
    18:         super().__init__()
    19:         # NonLinearHead pattern used by the HypSeek implementation.
    20:         self.mol_project = nn.Sequential(
    21:             nn.Linear(mol_dim, mol_dim), nn.ReLU(), nn.Linear(mol_dim, embed_dim)
    22:         )
    23:         self.pocket_project = nn.Sequential(
    24:             nn.Linear(pocket_dim, pocket_dim), nn.ReLU(), nn.Linear(pocket_dim, embed_dim)
    25:         )
    26:         self.protein_project = nn.Sequential(
    27:             nn.Linear(protein_dim, protein_dim), nn.ReLU(), nn.Linear(protein_dim, embed_dim)
    28:         )
    29:         self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(13))
    30: 
    31:     def project_mol(self, mol_feat):
    32:         return F.normalize(self.mol_project(mol_feat), dim=-1)
    33: 
    34:     def project_pocket(self, poc_feat):
    35:         return F.normalize(self.pocket_project(poc_feat), dim=-1)
    36: 
    37:     def project_protein(self, prot_feat):
    38:         return F.normalize(self.protein_project(prot_feat), dim=-1)
    39: 
    40:     def _compute_hcc_pair(self, emb_poc, emb_mol, batch_list, act_list,
    41:                           uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles,
    42:                           logit_scale):
    43:         """Compute HCC loss for one pathway (pocket-mol or protein-mol)."""
    44:         B = emb_poc.size(0)
    45:         logits = emb_poc @ emb_mol.T * logit_scale
    46: 
    47:         # False-negative mask
    48:         mask = torch.zeros_like(logits, dtype=torch.bool)
    49:         if uniprot_poc is not None and uniprot_mol is not None:
    50:             for i in range(B):
    51:                 for j in range(logits.size(1)):
    52:                     if uniprot_poc[i] == uniprot_mol[j]:
    53:                         mask[i, j] = True
    54:         if pocket_lig_smiles is not None:
    55:             for i in range(B):
    56:                 bad = pocket_lig_smiles[i]
    57:                 for j in range(logits.size(1)):
    58:                     if lig_smiles[j] in bad:
    59:                         mask[i, j] = True
    60: 
    61:         minus_inf = torch.finfo(logits.dtype).min
    62:         sim_masked = logits.masked_fill(mask, minus_inf)
    63: 
    64:         # Pocket retrieves ligands
    65:         idx2poc = []
    66:         for i, (s, e) in enumerate(batch_list):
    67:             idx2poc += [i] * (e - s)
    68:         targets = torch.tensor(idx2poc, dtype=torch.long, device=logits.device)
    69:         lprobs_pocket_all = F.log_softmax(sim_masked.T, dim=-1)
    70: 
    71:         loss_pocket_list = []
    72:         for i, (s, e) in enumerate(batch_list):
    73:             L_i = e - s
    74:             if L_i == 0:
    75:                 continue
    76:             rows = list(range(s, e))
    77:             lprobs_sub = lprobs_pocket_all[rows]
    78:             targ_sub = targets[rows]
    79:             loss_tmp = F.nll_loss(lprobs_sub, targ_sub, reduction="none")
    80:             loss_pocket_list.append(loss_tmp.sum() / math.sqrt(L_i))
    81:         loss_pocket = torch.stack(loss_pocket_list).sum() if loss_pocket_list else torch.tensor(0.0, device=logits.device)
    82: 
    83:         # Ligand retrieves pocket (skip low-activity ligands in multi-ligand pockets)
    84:         loss_mol_list = []
    85:         for i in range(B):
    86:             s, e = batch_list[i]
    87:             acts = act_list[i]
    88:             L_i = e - s
    89:             for k in range(s, e):
    90:                 row_mask = torch.full_like(sim_masked[i], minus_inf)
    91:                 row_mask[k] = 0
    92:                 lprobs = F.log_softmax(row_mask + sim_masked[i], dim=-1)
    93:                 if L_i > 1 and acts[k - s] < 5:
    94:                     continue
    95:                 loss_mol_list.append(-lprobs[k] / math.sqrt(L_i))
    96:         loss_mol = torch.stack(loss_mol_list).sum() if loss_mol_list else torch.tensor(0.0, device=logits.device)
    97: 
    98:         # Ranking loss: within each pocket, rank by activity
    99:         loss_rank_list = []
   100:         for i in range(B):
   101:             s, e = batch_list[i]
   102:             acts = act_list[i]
   103:             L_i = e - s
   104:             if L_i <= 2:
   105:                 continue
   106:             out_i = sim_masked[i, s:e]
   107:             for k_rel in range(L_i - 1):
   108:                 m = torch.zeros_like(out_i)
   109:                 for idx in range(L_i):
   110:                     if idx == k_rel:
   111:                         continue
   112:                     if acts[k_rel] - math.log10(3) <= acts[idx]:
   113:                         m[idx] = minus_inf
   114:                 lprobs_rank = F.log_softmax(m + out_i, dim=-1)
   115:                 loss_rank_list.append(-lprobs_rank[k_rel] / (math.log(k_rel + 2) * math.sqrt(L_i)))
   116:         loss_rank = torch.stack(loss_rank_list).sum() if loss_rank_list else torch.tensor(0.0, device=logits.device)
   117: 
   118:         total = loss_pocket + loss_mol + loss_rank
   119:         return {
   120:             "loss": total,
   121:             "loss_pocket": loss_pocket,
   122:             "loss_mol": loss_mol,
   123:             "loss_rank": loss_rank,
   124:             "sim_masked": sim_masked,
   125:         }
   126: 
   127:     def compute_loss(self, mol_emb, poc_emb, prot_emb,
   128:                      batch_list, act_list,
   129:                      uniprot_poc=None, uniprot_mol=None,
   130:                      pocket_lig_smiles=None, lig_smiles=None):
   131:         logit_scale = self.logit_scale.exp().detach()
   132: 
   133:         # HCC for pocket-molecule pathway
   134:         loss_dict_poc = self._compute_hcc_pair(
   135:             poc_emb, mol_emb, batch_list, act_list,
   136:             uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles,
   137:             logit_scale,
   138:         )
   139:         # HCC for protein-molecule pathway
   140:         loss_dict_prot = self._compute_hcc_pair(
   141:             prot_emb, mol_emb, batch_list, act_list,
   142:             uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles,
   143:             logit_scale,
   144:         )
   145:         loss = loss_dict_poc["loss"] + loss_dict_prot["loss"]
   146: 
   147:         return loss, {
   148:             "loss": loss.item(),
   149:             "loss_poc": loss_dict_poc["loss"].item(),
   150:             "loss_prot": loss_dict_prot["loss"].item(),
   151:             "sim_masked": loss_dict_poc["sim_masked"],
   152:         }
   153: 
   154:     def score(self, mol_reps, pocket_reps, prot_reps=None):
   155:         poc_scores = (pocket_reps @ mol_reps.T).max(axis=0)
   156:         if prot_reps is not None:
   157:             prot_scores = (prot_reps @ mol_reps.T).max(axis=0)
   158:             return poc_scores + prot_scores
   159:         return poc_scores
```

### `hcc_hyp_cone` baseline — editable region  [READ-ONLY — reference implementation]

In `HypSeek/unimol/custom_scoring.py`:

```python
     1: """Full HypSeek scoring: Hyperbolic HCC + Cone Hierarchy."""
     2: 
     3: import math
     4: import numpy as np
     5: import torch
     6: import torch.nn as nn
     7: import torch.nn.functional as F
     8: from unimol.losses import lorentz as L
     9: 
    10: 
    11: class CustomScoring(nn.Module):
    12:     """Full HypSeek: Lorentz hyperbolic embeddings + HCC + cone hierarchy.
    13: 
    14:     Maps projected features onto a Lorentz hyperboloid via exp_map0,
    15:     trains with HCC contrastive-ranking loss plus cone hierarchy
    16:     constraints (radial + angular).
    17:     """
    18: 
    19:     def __init__(self, mol_dim=512, pocket_dim=512, protein_dim=480, embed_dim=128):
    20:         super().__init__()
    21:         # Projection heads (NonLinearHead equivalent: hidden=input_dim)
    22:         # Paper unimol/models/unimol.py:345-360 — NonLinearHead(in, out, 'relu')
    23:         # uses hidden=in by default: Linear(in,in) -> ReLU -> Linear(in,out).
    24:         self.mol_project = nn.Sequential(
    25:             nn.Linear(mol_dim, mol_dim), nn.ReLU(), nn.Linear(mol_dim, embed_dim)
    26:         )
    27:         self.pocket_project = nn.Sequential(
    28:             nn.Linear(pocket_dim, pocket_dim), nn.ReLU(), nn.Linear(pocket_dim, embed_dim)
    29:         )
    30:         self.protein_project = nn.Sequential(
    31:             nn.Linear(protein_dim, protein_dim), nn.ReLU(), nn.Linear(protein_dim, embed_dim)
    32:         )
    33: 
    34:         # Learnable scale parameters (log-space, clamped to exp(alpha) <= 1)
    35:         self.mol_alpha = nn.Parameter(torch.tensor([embed_dim ** -0.5]).log())
    36:         self.pocket_alpha = nn.Parameter(torch.tensor([embed_dim ** -0.5]).log())
    37:         self.protein_alpha = nn.Parameter(torch.tensor([embed_dim ** -0.5]).log())
    38: 
    39:         # Learnable curvature (log-space)
    40:         self.curv = nn.Parameter(torch.tensor([1.0]).log(), requires_grad=True)
    41:         self._curv_minmax = {"max": math.log(10.0), "min": math.log(0.1)}
    42: 
    43:         # Temperature
    44:         self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(13))
    45: 
    46:         # Cone hierarchy hyperparameters
    47:         self.bounds = torch.tensor([5.0, 7.0, 9.0], dtype=torch.float32)
    48:         self.chl_r0 = 0.5
    49:         self.chl_dr = 0.5
    50:         self.chl_eta0 = 0.7
    51:         self.chl_deta = 0.2
    52:         self.lambda_rad = 0.5
    53:         self.lambda_ang = 0.5
    54:         self.gamma_chl = 0.1
    55:         self.lambda_angu = 0.10
    56:         self.lambda_het = 0.10
    57: 
    58:     def _clamp_params(self):
    59:         """Clamp scale and curvature parameters."""
    60:         self.mol_alpha.data = torch.clamp(self.mol_alpha.data, max=0.0)
    61:         self.pocket_alpha.data = torch.clamp(self.pocket_alpha.data, max=0.0)
    62:         self.protein_alpha.data = torch.clamp(self.protein_alpha.data, max=0.0)
    63:         self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
    64: 
    65:     def _project_to_hyperboloid(self, feat, proj_head, alpha):
    66:         """Project features to Lorentz hyperboloid."""
    67:         u = proj_head(feat) * alpha.exp()
    68:         with torch.autocast(u.device.type, dtype=torch.float32):
    69:             h = L.exp_map0(u, self.curv.exp())
    70:         return h
    71: 
    72:     def project_mol(self, mol_feat):
    73:         self._clamp_params()
    74:         return self._project_to_hyperboloid(mol_feat, self.mol_project, self.mol_alpha)
    75: 
    76:     def project_pocket(self, poc_feat):
    77:         self._clamp_params()
    78:         return self._project_to_hyperboloid(poc_feat, self.pocket_project, self.pocket_alpha)
    79: 
    80:     def project_protein(self, prot_feat):
    81:         self._clamp_params()
    82:         return self._project_to_hyperboloid(prot_feat, self.protein_project, self.protein_alpha)
    83: 
    84:     def _compute_hcc_pair(self, emb_poc, emb_mol, batch_list, act_list,
    85:                           uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles,
    86:                           logit_scale):
    87:         """HCC loss for one pathway (space component dot product).
    88: 
    89:         Paper three_hybrid_loss.py:187 takes emb_poc[:, 1:] @ emb_mol[:, 1:].T,
    90:         i.e. drops index 0 of the space component before similarity.
    91:         """
    92:         B = emb_poc.size(0)
    93:         emb_poc = emb_poc[:, 1:]
    94:         emb_mol = emb_mol[:, 1:]
    95:         logits = torch.matmul(emb_poc, emb_mol.T) * logit_scale
    96: 
    97:         N_mol = emb_mol.size(0)
    98:         mask = torch.zeros_like(logits, dtype=torch.bool)
    99:         if uniprot_poc is not None and uniprot_mol is not None:
   100:             for i in range(B):
   101:                 for j in range(N_mol):
   102:                     if uniprot_poc[i] == uniprot_mol[j]:
   103:                         mask[i, j] = True
   104:         if pocket_lig_smiles is not None:
   105:             for i in range(B):
   106:                 bad = pocket_lig_smiles[i]
   107:                 for j in range(N_mol):
   108:                     if lig_smiles[j] in bad:
   109:                         mask[i, j] = True
   110: 
   111:         minus_inf = torch.finfo(logits.dtype).min
   112:         sim_masked = logits.masked_fill(mask, minus_inf)
   113: 
   114:         # Pocket retrieves ligands
   115:         loss_mol_list, loss_rank_list = [], []
   116:         for i in range(B):
   117:             s, e = batch_list[i]
   118:             acts = act_list[i]
   119:             L_i = e - s
   120:             out_i = sim_masked[i, s:e]
   121:             for k in range(s, e):
   122:                 row_mask = torch.full_like(sim_masked[i], minus_inf)
   123:                 row_mask[k] = 0
   124:                 lprobs = F.log_softmax(row_mask + sim_masked[i], dim=-1)
   125:                 if L_i > 1 and acts[k - s] < 5:
   126:                     continue
   127:                 loss_mol_list.append(-lprobs[k] / math.sqrt(L_i))
   128:             if L_i > 2:
   129:                 for k_rel in range(L_i - 1):
   130:                     m = torch.zeros_like(out_i)
   131:                     for idx in range(L_i):
   132:                         if idx == k_rel:
   133:                             continue
   134:                         if acts[k_rel] - math.log10(3) <= acts[idx]:
   135:                             m[idx] = minus_inf
   136:                     lprobs_rank = F.log_softmax(m + out_i, dim=-1)
   137:                     loss_rank_list.append(-lprobs_rank[k_rel] / (math.log(k_rel + 2) * math.sqrt(L_i)))
   138:         loss_mol = torch.stack(loss_mol_list).sum() if loss_mol_list else torch.tensor(0.0, device=logits.device)
   139:         loss_rank = torch.stack(loss_rank_list).sum() if loss_rank_list else torch.tensor(0.0, device=logits.device)
   140: 
   141:         # Ligand-to-pocket
   142:         idx2poc = []
   143:         for i, (s, e) in enumerate(batch_list):
   144:             idx2poc += [i] * (e - s)
   145:         targets = torch.tensor(idx2poc, dtype=torch.long, device=logits.device)
   146:         lprobs_pocket_all = F.log_softmax(sim_masked.T, dim=-1)
   147:         loss_pocket_list = []
   148:         for i, (s, e) in enumerate(batch_list):
   149:             L_i = e - s
   150:             if L_i == 0:
   151:                 continue
   152:             rows = list(range(s, e))
   153:             lprobs_sub = lprobs_pocket_all[rows]
   154:             targ_sub = targets[rows]
   155:             loss_tmp = F.nll_loss(lprobs_sub, targ_sub, reduction="none")
   156:             loss_pocket_list.append(loss_tmp.sum() / math.sqrt(L_i))
   157:         loss_pocket = torch.stack(loss_pocket_list).sum() if loss_pocket_list else torch.tensor(0.0, device=logits.device)
   158: 
   159:         total = loss_pocket + loss_mol + loss_rank
   160:         return {"loss": total, "loss_pocket": loss_pocket, "loss_mol": loss_mol,
   161:                 "loss_rank": loss_rank, "sim_masked": sim_masked}
   162: 
   163:     def compute_loss(self, mol_emb, poc_emb, prot_emb,
   164:                      batch_list, act_list,
   165:                      uniprot_poc=None, uniprot_mol=None,
   166:                      pocket_lig_smiles=None, lig_smiles=None):
   167:         kappa = self.curv.exp().detach()
   168:         logit_scale = self.logit_scale.exp().detach()
   169:         B = poc_emb.size(0)
   170: 
   171:         # === Cone Hierarchy Loss ===
   172:         # Match paper three_hybrid_loss.py:73-74 — drop index 0 of space dim.
   173:         poc_space = poc_emb[:, 1:]
   174:         lig_space = mol_emb[:, 1:]
   175:         poc_idx = []
   176:         for i, (s, e) in enumerate(batch_list):
   177:             poc_idx += [i] * (e - s)
   178:         poc_idx = torch.tensor(poc_idx, device=poc_emb.device)
   179: 
   180:         poc_sel = poc_space[poc_idx]
   181:         dist_mat = L.pairwise_dist(poc_sel, lig_space, curv=kappa)
   182:         dist = dist_mat.diagonal()
   183:         device = dist.device
   184:         phi = L.oxy_angle(lig_space, poc_space[poc_idx], curv=kappa)
   185:         omega = L.half_aperture(poc_space[poc_idx], curv=kappa)
   186:         act_flat = torch.tensor(
   187:             [x for sub in act_list for x in sub],
   188:             device=poc_emb.device, dtype=torch.float32,
   189:         )
   190:         bounds = self.bounds.to(poc_emb.device)
   191:         bucket = torch.bucketize(act_flat, bounds)
   192:         r_k = self.chl_r0 + bucket.float() * self.chl_dr
   193:         eta_k = self.chl_eta0 - bucket.float() * self.chl_deta
   194:         Nl = dist.size(0)
   195:         L_rad = F.relu(dist - r_k).sum() / math.sqrt(Nl)
   196:         L_ang = F.relu(phi - eta_k * omega).sum() / math.sqrt(Nl)
   197:         loss_cone = self.lambda_rad * L_rad + self.lambda_ang * L_ang
   198: 
   199:         # Angular regularization
   200:         m_margin = 0.15
   201:         R_ang = F.relu(phi - eta_k * omega + m_margin).sum() / math.sqrt(Nl)
   202: 
   203:         # Heterogeneous ranking regularization
   204:         R_het = torch.zeros(1, device=device)
   205:         cnt_het = 0
   206:         beta = 80.5
   207:         offset = 0
   208:         for i_poc, (s, e) in enumerate(batch_list):
   209:             L_i = e - s
   210:             if L_i < 1:
   211:                 continue
   212:             d_i = dist[offset : offset + L_i].detach()
   213:             rank = (d_i.unsqueeze(0) < d_i.unsqueeze(1)).float().sum(1) + 1
   214:             w = torch.exp(-beta * (rank - 1) / L_i)
   215:             logits_row = torch.matmul(poc_space[i_poc : i_poc + 1], lig_space.T) * logit_scale
   216:             row_probs = F.softmax(logits_row[0, s:e], dim=-1)
   217:             pos_mask = act_flat[offset : offset + L_i] < 5
   218:             if pos_mask.any():
   219:                 R_het += -(w[pos_mask] * row_probs[pos_mask].log()).sum() / (w[pos_mask].sum() + 1e-9)
   220:                 cnt_het += 1
   221:             offset += L_i
   222:         R_het = R_het / max(cnt_het, 1)
   223:         loss_reg = self.lambda_het * R_het + self.lambda_angu * R_ang
   224: 
   225:         # === HCC for both pathways ===
   226:         loss_dict_poc = self._compute_hcc_pair(
   227:             poc_emb, mol_emb, batch_list, act_list,
   228:             uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles, logit_scale,
   229:         )
   230:         loss_dict_prot = self._compute_hcc_pair(
   231:             prot_emb, mol_emb, batch_list, act_list,
   232:             uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles, logit_scale,
   233:         )
   234: 
   235:         loss_hcc = loss_dict_poc["loss"] + loss_dict_prot["loss"]
   236:         total_loss = loss_hcc + self.gamma_chl * loss_cone + loss_reg
   237: 
   238:         return total_loss, {
   239:             "loss": total_loss.item(),
   240:             "loss_hcc": loss_hcc.item(),
   241:             "loss_cone": loss_cone.item(),
   242:             "loss_reg": loss_reg.item(),
   243:             "sim_masked": loss_dict_poc["sim_masked"],
   244:         }
   245: 
   246:     def score(self, mol_reps, pocket_reps, prot_reps=None):
   247:         """Score using full 128-d hyperbolic embedding (paper convention).
   248: 
   249:         NOTE: this method is dead code — virtual-screening evaluation goes
   250:         through unimol/tasks/test_task.py:test_dude_target which scores via
   251:         full-embedding dot product (test_task.py:797-803, no [:, 1:] slice).
   252:         We keep this implementation aligned with that upstream convention so
   253:         any future caller stays consistent with paper evaluation.
   254:         """
   255:         poc_scores = (pocket_reps @ mol_reps.T).max(axis=0)
   256:         if prot_reps is not None:
   257:             prot_scores = (prot_reps @ mol_reps.T).max(axis=0)
   258:             return poc_scores + prot_scores
   259:         return poc_scores
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
