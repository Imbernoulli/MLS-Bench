# Task: Protein Mutation Effect Prediction

## Research Question
Design a supervised prediction architecture that maps pre-computed protein language model (PLM) embeddings to protein fitness scores, improving over simple linear or shallow models for mutation effect prediction.

## Background
Predicting the functional effect of amino acid mutations is a central problem in protein engineering and clinical genetics. Deep mutational scanning (DMS) experiments measure the fitness effect of thousands of mutations in a protein, but are expensive and time-consuming. Computational prediction of these effects can accelerate protein design.

The task uses frozen ESM-2 (650M) protein language model representations (Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model", Science 2023; arXiv:2206.13517 / bioRxiv 2022.07.20.500902) and asks for a supervised prediction head over those embeddings.

Key considerations:
- **Embedding structure**: ESM-2 embeddings encode rich structural and evolutionary information in 1280 dimensions. How best to exploit this high-dimensional representation?
- **Delta features**: The difference between mutant and wild-type embeddings directly encodes what changed due to the mutation.
- **Generalization across folds**: The model must generalize across cross-validation splits, not just memorize training examples.

## What to Implement
Implement the `MutationPredictor` class in `custom_mutation_pred.py`. You must implement:
1. `__init__(self, embed_dim)`: Set up your model architecture. `embed_dim` is 1280 (ESM-2 650M).
2. `forward(self, embedding, delta_embedding) -> Tensor`: Return predictions of shape `[B]`.

## Input Format
The model receives two inputs per mutant:
- `embedding`: `[B, 1280]` — Mean-pooled ESM-2 (650M) representation of the mutant sequence.
- `delta_embedding`: `[B, 1280]` — Difference from wild-type embedding (`mutant_emb - wt_emb`).

## Output Format
- Return a tensor of shape `[B]` with predicted fitness scores (real-valued).

## Fixed Pipeline
The data pipeline, train/test loop, embedding extraction, and cross-validation splits are all fixed by the scaffold. The only learnable degrees of freedom are (a) the `MutationPredictor` architecture and (b) optimizer hyperparameters exposed via `CONFIG_OVERRIDES` in `main()` (allowed keys: `learning_rate`, `weight_decay`).

## Evaluation
The model is evaluated on DMS assays from the ProteinGym benchmark (Notin et al., "ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design", NeurIPS 2023 Datasets & Benchmarks):

- **BLAT_ECOLX** (Beta-lactamase, OrganismalFitness, 4783 single mutants): Antibiotic resistance enzyme from E. coli.
- **ESTA_BACSU** (Esterase, Stability, 2172 single mutants): Thermostability of a B. subtilis esterase.
- **RASH_HUMAN** (K-Ras GTPase, Activity, 3134 single mutants): Oncogene activity in human cells.

**Metric**: Spearman rank correlation between predicted and true fitness scores, averaged over 5-fold cross-validation using ProteinGym's pre-defined **random** folds. Higher is better.

> ⚠️ **Evaluation protocol note.** ProteinGym's supervised leaderboard averages
> Spearman over three fold strategies — `random`, `modulo` (every 5th residue),
> and `contiguous` (held-out sequence blocks). This task uses **only the
> `random` fold strategy**, which is the easiest of the three and tends to
> give higher Spearman than the published ProteinGym SOTA averages. Numbers
> reported here are therefore not directly comparable to the ProteinGym
> supervised leaderboard; treat them as within-benchmark-relative scores.

## Baselines
Reference baselines on the same fixed pipeline:
- **Ridge regression** on concatenated `[embedding, delta_embedding]` features.
- **MLP** prediction head over the same concatenated features.
- **Reshape-CNN** that reshapes the 1280-dim embedding into a 2D grid and applies small convolutions before regression.

All baselines see the same ESM-2 embeddings, the same CV splits, and the same train/test loop; they differ only in the prediction head.

## Editable Region
The `MutationPredictor` class lives between `EDITABLE SECTION START` and `EDITABLE SECTION END` markers in `custom_mutation_pred.py`. You may define helper classes, layers, or functions within this region. The region must contain a `MutationPredictor` class that is an `nn.Module` with the specified interface.

You may additionally set training-loop hyperparameters by writing into the small `CONFIG_OVERRIDES = {}` dict in `main()` (a small editable region near the bottom of the file). Allowed keys: `learning_rate`, `weight_decay`.
