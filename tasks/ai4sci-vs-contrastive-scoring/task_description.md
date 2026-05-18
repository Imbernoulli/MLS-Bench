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
