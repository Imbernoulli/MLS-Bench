"""HCC (Hierarchical Contrastive Cosine) baseline.

Euclidean embeddings with contrastive + ranking loss.
Adds activity-aware ranking loss on top of vanilla contrastive.

Reference: HypSeek (Wang et al., arXiv:2508.15480 / NeurIPS 2025 AI4Science workshop).
    vendor/external_packages/HypSeek/unimol/losses/three_hybrid_loss.py
"""

_FILE = "HypSeek/unimol/custom_scoring.py"

_CONTENT = '''\
"""HCC scoring module: Euclidean contrastive + ranking loss."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomScoring(nn.Module):
    """HCC: Hierarchical Contrastive Cosine in Euclidean space.

    Adds ranking loss that enforces more active ligands score higher
    within each pocket's ligand set, weighted by 1/log(rank+2) (DCG-style).
    """

    def __init__(self, mol_dim=512, pocket_dim=512, protein_dim=480, embed_dim=128):
        super().__init__()
        # NonLinearHead pattern used by the HypSeek implementation.
        self.mol_project = nn.Sequential(
            nn.Linear(mol_dim, mol_dim), nn.ReLU(), nn.Linear(mol_dim, embed_dim)
        )
        self.pocket_project = nn.Sequential(
            nn.Linear(pocket_dim, pocket_dim), nn.ReLU(), nn.Linear(pocket_dim, embed_dim)
        )
        self.protein_project = nn.Sequential(
            nn.Linear(protein_dim, protein_dim), nn.ReLU(), nn.Linear(protein_dim, embed_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(13))

    def project_mol(self, mol_feat):
        return F.normalize(self.mol_project(mol_feat), dim=-1)

    def project_pocket(self, poc_feat):
        return F.normalize(self.pocket_project(poc_feat), dim=-1)

    def project_protein(self, prot_feat):
        return F.normalize(self.protein_project(prot_feat), dim=-1)

    def _compute_hcc_pair(self, emb_poc, emb_mol, batch_list, act_list,
                          uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles,
                          logit_scale):
        """Compute HCC loss for one pathway (pocket-mol or protein-mol)."""
        B = emb_poc.size(0)
        logits = emb_poc @ emb_mol.T * logit_scale

        # False-negative mask
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

        # Pocket retrieves ligands
        idx2poc = []
        for i, (s, e) in enumerate(batch_list):
            idx2poc += [i] * (e - s)
        targets = torch.tensor(idx2poc, dtype=torch.long, device=logits.device)
        lprobs_pocket_all = F.log_softmax(sim_masked.T, dim=-1)

        loss_pocket_list = []
        for i, (s, e) in enumerate(batch_list):
            L_i = e - s
            if L_i == 0:
                continue
            rows = list(range(s, e))
            lprobs_sub = lprobs_pocket_all[rows]
            targ_sub = targets[rows]
            loss_tmp = F.nll_loss(lprobs_sub, targ_sub, reduction="none")
            loss_pocket_list.append(loss_tmp.sum() / math.sqrt(L_i))
        loss_pocket = torch.stack(loss_pocket_list).sum() if loss_pocket_list else torch.tensor(0.0, device=logits.device)

        # Ligand retrieves pocket (skip low-activity ligands in multi-ligand pockets)
        loss_mol_list = []
        for i in range(B):
            s, e = batch_list[i]
            acts = act_list[i]
            L_i = e - s
            for k in range(s, e):
                row_mask = torch.full_like(sim_masked[i], minus_inf)
                row_mask[k] = 0
                lprobs = F.log_softmax(row_mask + sim_masked[i], dim=-1)
                if L_i > 1 and acts[k - s] < 5:
                    continue
                loss_mol_list.append(-lprobs[k] / math.sqrt(L_i))
        loss_mol = torch.stack(loss_mol_list).sum() if loss_mol_list else torch.tensor(0.0, device=logits.device)

        # Ranking loss: within each pocket, rank by activity
        loss_rank_list = []
        for i in range(B):
            s, e = batch_list[i]
            acts = act_list[i]
            L_i = e - s
            if L_i <= 2:
                continue
            out_i = sim_masked[i, s:e]
            for k_rel in range(L_i - 1):
                m = torch.zeros_like(out_i)
                for idx in range(L_i):
                    if idx == k_rel:
                        continue
                    if acts[k_rel] - math.log10(3) <= acts[idx]:
                        m[idx] = minus_inf
                lprobs_rank = F.log_softmax(m + out_i, dim=-1)
                loss_rank_list.append(-lprobs_rank[k_rel] / (math.log(k_rel + 2) * math.sqrt(L_i)))
        loss_rank = torch.stack(loss_rank_list).sum() if loss_rank_list else torch.tensor(0.0, device=logits.device)

        total = loss_pocket + loss_mol + loss_rank
        return {
            "loss": total,
            "loss_pocket": loss_pocket,
            "loss_mol": loss_mol,
            "loss_rank": loss_rank,
            "sim_masked": sim_masked,
        }

    def compute_loss(self, mol_emb, poc_emb, prot_emb,
                     batch_list, act_list,
                     uniprot_poc=None, uniprot_mol=None,
                     pocket_lig_smiles=None, lig_smiles=None):
        logit_scale = self.logit_scale.exp().detach()

        # HCC for pocket-molecule pathway
        loss_dict_poc = self._compute_hcc_pair(
            poc_emb, mol_emb, batch_list, act_list,
            uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles,
            logit_scale,
        )
        # HCC for protein-molecule pathway
        loss_dict_prot = self._compute_hcc_pair(
            prot_emb, mol_emb, batch_list, act_list,
            uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles,
            logit_scale,
        )
        loss = loss_dict_poc["loss"] + loss_dict_prot["loss"]

        return loss, {
            "loss": loss.item(),
            "loss_poc": loss_dict_poc["loss"].item(),
            "loss_prot": loss_dict_prot["loss"].item(),
            "sim_masked": loss_dict_poc["sim_masked"],
        }

    def score(self, mol_reps, pocket_reps, prot_reps=None):
        poc_scores = (pocket_reps @ mol_reps.T).max(axis=0)
        if prot_reps is not None:
            prot_scores = (prot_reps @ mol_reps.T).max(axis=0)
            return poc_scores + prot_scores
        return poc_scores
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 1,
        "end_line": -1,
        "content": _CONTENT,
    },
]
