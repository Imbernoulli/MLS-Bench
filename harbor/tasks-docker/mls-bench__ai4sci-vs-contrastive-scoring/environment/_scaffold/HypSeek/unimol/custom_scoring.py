"""Custom scoring module for contrastive virtual screening.

This module defines the projection heads, embedding space mapping, and
training loss for protein-ligand virtual screening.

Interface:
    project_mol(mol_feat)      -> [B, embed_dim]  molecule embeddings
    project_pocket(poc_feat)   -> [B, embed_dim]  pocket embeddings
    project_protein(prot_feat) -> [B, embed_dim]  protein embeddings
    compute_loss(mol_emb, poc_emb, prot_emb, ...) -> (loss, dict)
    score(mol_reps, pocket_reps, prot_reps) -> [N_mol] numpy scores

Available utilities (imported in model wrapper):
    torch, torch.nn, torch.nn.functional, numpy, math
    lorentz ops: from unimol.losses.lorentz import exp_map0, pairwise_dist,
                 half_aperture, oxy_angle, pairwise_inner, minkowski_dot

Backbone encoder outputs (frozen, not editable):
    mol_feat:  [B, 512]  CLS token from UniMol molecule encoder
    poc_feat:  [B, 512]  CLS token from UniMol pocket encoder
    prot_feat: [B, 480]  CLS token from ESM2 protein sequence encoder
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomScoring(nn.Module):
    """Scoring module for contrastive protein-ligand virtual screening.

    Handles projection of frozen encoder features into a shared embedding
    space and computes the training loss for ranking actives above decoys.
    """

    def __init__(self, mol_dim=512, pocket_dim=512, protein_dim=480, embed_dim=128):
        super().__init__()
        # Projection heads following paper's NonLinearHead pattern
        # (vendor/external_packages/HypSeek/unimol/models/unimol.py:345-360):
        # hidden=input_dim, i.e. Linear(in,in) -> ReLU -> Linear(in,embed_dim).
        self.mol_project = nn.Sequential(
            nn.Linear(mol_dim, mol_dim), nn.ReLU(), nn.Linear(mol_dim, embed_dim)
        )
        self.pocket_project = nn.Sequential(
            nn.Linear(pocket_dim, pocket_dim), nn.ReLU(), nn.Linear(pocket_dim, embed_dim)
        )
        self.protein_project = nn.Sequential(
            nn.Linear(protein_dim, protein_dim), nn.ReLU(), nn.Linear(protein_dim, embed_dim)
        )
        # Learnable temperature (log scale).
        # log(13) matches the paper's three_hybrid_model.py:58 — see
        # vendor/external_packages/HypSeek/unimol/models/three_hybrid_model.py.
        self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(13))

    def project_mol(self, mol_feat):
        """Project molecule encoder features to embedding space."""
        return F.normalize(self.mol_project(mol_feat), dim=-1)

    def project_pocket(self, poc_feat):
        """Project pocket encoder features to embedding space."""
        return F.normalize(self.pocket_project(poc_feat), dim=-1)

    def project_protein(self, prot_feat):
        """Project protein encoder features to embedding space."""
        return F.normalize(self.protein_project(prot_feat), dim=-1)

    def compute_loss(self, mol_emb, poc_emb, prot_emb,
                     batch_list, act_list,
                     uniprot_poc=None, uniprot_mol=None,
                     pocket_lig_smiles=None, lig_smiles=None):
        """Compute training loss.

        Args:
            mol_emb:  [N_mol, D] molecule embeddings (all ligands in batch)
            poc_emb:  [N_poc, D] pocket embeddings (one per assay)
            prot_emb: [N_poc, D] protein embeddings (one per assay)
            batch_list: list of (start, end) tuples mapping pocket i to its
                        ligands mol_emb[start:end]
            act_list:   list of activity values (pIC50) per pocket's ligands
            uniprot_poc: UniProt IDs for pockets (for false-negative masking)
            uniprot_mol: UniProt IDs for molecules (for false-negative masking)
            pocket_lig_smiles: known ligand SMILES per pocket
            lig_smiles: SMILES for each molecule in batch

        Returns:
            loss: scalar training loss
            log_dict: dict with loss components and sim_masked for validation
        """
        logit_scale = self.logit_scale.exp().detach()
        B = poc_emb.size(0)

        # Similarity matrix: [N_poc, N_mol]
        logits = poc_emb @ mol_emb.T * logit_scale

        # Build false-negative mask (same protein or known binder)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        if uniprot_poc is not None and uniprot_mol is not None:
            for i in range(B):
                for j in range(logits.size(1)):
                    if uniprot_poc[i] == uniprot_mol[j]:
                        mask[i, j] = True
        if pocket_lig_smiles is not None:
            for i in range(B):
                bad = pocket_lig_smiles[i]
                for j in range(logits.size(1)):
                    if lig_smiles[j] in bad:
                        mask[i, j] = True

        minus_inf = torch.finfo(logits.dtype).min
        sim_masked = logits.masked_fill(mask, minus_inf)

        # === Symmetric contrastive loss ===
        # Pocket-to-ligand: each pocket retrieves its ligands
        idx2poc = []
        for i, (s, e) in enumerate(batch_list):
            idx2poc += [i] * (e - s)
        targets = torch.tensor(idx2poc, dtype=torch.long, device=logits.device)

        lprobs_pocket = F.log_softmax(sim_masked.T, dim=-1)
        loss_pocket_list = []
        for i, (s, e) in enumerate(batch_list):
            L_i = e - s
            if L_i == 0:
                continue
            rows = list(range(s, e))
            lprobs_sub = lprobs_pocket[rows]
            targ_sub = targets[rows]
            loss_tmp = F.nll_loss(lprobs_sub, targ_sub, reduction="none")
            loss_pocket_list.append(loss_tmp.sum() / math.sqrt(L_i))
        loss_pocket = torch.stack(loss_pocket_list).sum() if loss_pocket_list else torch.tensor(0.0, device=logits.device)

        # Ligand-to-pocket: each ligand retrieves its pocket
        loss_mol_list = []
        for i in range(B):
            s, e = batch_list[i]
            for k in range(s, e):
                row_mask = torch.full_like(sim_masked[i], minus_inf)
                row_mask[k] = 0
                lprobs = F.log_softmax(row_mask + sim_masked[i], dim=-1)
                loss_mol_list.append(-lprobs[k] / math.sqrt(e - s))
        loss_mol = torch.stack(loss_mol_list).sum() if loss_mol_list else torch.tensor(0.0, device=logits.device)

        loss = loss_pocket + loss_mol

        return loss, {
            "loss": loss.item(),
            "loss_pocket": loss_pocket.item(),
            "loss_mol": loss_mol.item(),
            "sim_masked": sim_masked,
        }

    def score(self, mol_reps, pocket_reps, prot_reps=None):
        """Score molecules against pocket/protein for evaluation.

        Args:
            mol_reps:    [N_mol, D] numpy array of molecule embeddings
            pocket_reps: [N_poc, D] numpy array of pocket embeddings
            prot_reps:   [N_prot, D] numpy array of protein embeddings (optional)

        Returns:
            scores: [N_mol] numpy array of final scores per molecule
        """
        poc_scores = (pocket_reps @ mol_reps.T).max(axis=0)
        if prot_reps is not None:
            prot_scores = (prot_reps @ mol_reps.T).max(axis=0)
            return poc_scores + prot_scores
        return poc_scores
