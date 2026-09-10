"""HCC + Hyperbolic + Cone Hierarchy baseline (full HypSeek).

Lorentz hyperboloid embeddings with learnable curvature, HCC contrastive
ranking loss, cone hierarchy loss, and heterogeneous regularization.

Reference: HypSeek (AAAI 2026) — Wang et al.
    vendor/external_packages/HypSeek/unimol/losses/three_hybrid_loss.py
    vendor/external_packages/HypSeek/unimol/models/three_hybrid_model.py
"""

_FILE = "HypSeek/unimol/custom_scoring.py"

_CONTENT = '''\
"""Full HypSeek scoring: Hyperbolic HCC + Cone Hierarchy."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unimol.losses import lorentz as L


class CustomScoring(nn.Module):
    """Full HypSeek: Lorentz hyperbolic embeddings + HCC + cone hierarchy.

    Maps projected features onto a Lorentz hyperboloid via exp_map0,
    trains with HCC contrastive-ranking loss plus cone hierarchy
    constraints (radial + angular).
    """

    def __init__(self, mol_dim=512, pocket_dim=512, protein_dim=480, embed_dim=128):
        super().__init__()
        # Projection heads (NonLinearHead equivalent: hidden=input_dim)
        # Paper unimol/models/unimol.py:345-360 — NonLinearHead(in, out, 'relu')
        # uses hidden=in by default: Linear(in,in) -> ReLU -> Linear(in,out).
        self.mol_project = nn.Sequential(
            nn.Linear(mol_dim, mol_dim), nn.ReLU(), nn.Linear(mol_dim, embed_dim)
        )
        self.pocket_project = nn.Sequential(
            nn.Linear(pocket_dim, pocket_dim), nn.ReLU(), nn.Linear(pocket_dim, embed_dim)
        )
        self.protein_project = nn.Sequential(
            nn.Linear(protein_dim, protein_dim), nn.ReLU(), nn.Linear(protein_dim, embed_dim)
        )

        # Learnable scale parameters (log-space, clamped to exp(alpha) <= 1)
        self.mol_alpha = nn.Parameter(torch.tensor([embed_dim ** -0.5]).log())
        self.pocket_alpha = nn.Parameter(torch.tensor([embed_dim ** -0.5]).log())
        self.protein_alpha = nn.Parameter(torch.tensor([embed_dim ** -0.5]).log())

        # Learnable curvature (log-space)
        self.curv = nn.Parameter(torch.tensor([1.0]).log(), requires_grad=True)
        self._curv_minmax = {"max": math.log(10.0), "min": math.log(0.1)}

        # Temperature
        self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(13))

        # Cone hierarchy hyperparameters
        self.bounds = torch.tensor([5.0, 7.0, 9.0], dtype=torch.float32)
        self.chl_r0 = 0.5
        self.chl_dr = 0.5
        self.chl_eta0 = 0.7
        self.chl_deta = 0.2
        self.lambda_rad = 0.5
        self.lambda_ang = 0.5
        self.gamma_chl = 0.1
        self.lambda_angu = 0.10
        self.lambda_het = 0.10

    def _clamp_params(self):
        """Clamp scale and curvature parameters."""
        self.mol_alpha.data = torch.clamp(self.mol_alpha.data, max=0.0)
        self.pocket_alpha.data = torch.clamp(self.pocket_alpha.data, max=0.0)
        self.protein_alpha.data = torch.clamp(self.protein_alpha.data, max=0.0)
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)

    def _project_to_hyperboloid(self, feat, proj_head, alpha):
        """Project features to Lorentz hyperboloid."""
        u = proj_head(feat) * alpha.exp()
        with torch.autocast(u.device.type, dtype=torch.float32):
            h = L.exp_map0(u, self.curv.exp())
        return h

    def project_mol(self, mol_feat):
        self._clamp_params()
        return self._project_to_hyperboloid(mol_feat, self.mol_project, self.mol_alpha)

    def project_pocket(self, poc_feat):
        self._clamp_params()
        return self._project_to_hyperboloid(poc_feat, self.pocket_project, self.pocket_alpha)

    def project_protein(self, prot_feat):
        self._clamp_params()
        return self._project_to_hyperboloid(prot_feat, self.protein_project, self.protein_alpha)

    def _compute_hcc_pair(self, emb_poc, emb_mol, batch_list, act_list,
                          uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles,
                          logit_scale):
        """HCC loss for one pathway (space component dot product).

        Paper three_hybrid_loss.py:187 takes emb_poc[:, 1:] @ emb_mol[:, 1:].T,
        i.e. drops index 0 of the space component before similarity.
        """
        B = emb_poc.size(0)
        emb_poc = emb_poc[:, 1:]
        emb_mol = emb_mol[:, 1:]
        logits = torch.matmul(emb_poc, emb_mol.T) * logit_scale

        N_mol = emb_mol.size(0)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        if uniprot_poc is not None and uniprot_mol is not None:
            for i in range(B):
                for j in range(N_mol):
                    if uniprot_poc[i] == uniprot_mol[j]:
                        mask[i, j] = True
        if pocket_lig_smiles is not None:
            for i in range(B):
                bad = pocket_lig_smiles[i]
                for j in range(N_mol):
                    if lig_smiles[j] in bad:
                        mask[i, j] = True

        minus_inf = torch.finfo(logits.dtype).min
        sim_masked = logits.masked_fill(mask, minus_inf)

        # Pocket retrieves ligands
        loss_mol_list, loss_rank_list = [], []
        for i in range(B):
            s, e = batch_list[i]
            acts = act_list[i]
            L_i = e - s
            out_i = sim_masked[i, s:e]
            for k in range(s, e):
                row_mask = torch.full_like(sim_masked[i], minus_inf)
                row_mask[k] = 0
                lprobs = F.log_softmax(row_mask + sim_masked[i], dim=-1)
                if L_i > 1 and acts[k - s] < 5:
                    continue
                loss_mol_list.append(-lprobs[k] / math.sqrt(L_i))
            if L_i > 2:
                for k_rel in range(L_i - 1):
                    m = torch.zeros_like(out_i)
                    for idx in range(L_i):
                        if idx == k_rel:
                            continue
                        if acts[k_rel] - math.log10(3) <= acts[idx]:
                            m[idx] = minus_inf
                    lprobs_rank = F.log_softmax(m + out_i, dim=-1)
                    loss_rank_list.append(-lprobs_rank[k_rel] / (math.log(k_rel + 2) * math.sqrt(L_i)))
        loss_mol = torch.stack(loss_mol_list).sum() if loss_mol_list else torch.tensor(0.0, device=logits.device)
        loss_rank = torch.stack(loss_rank_list).sum() if loss_rank_list else torch.tensor(0.0, device=logits.device)

        # Ligand-to-pocket
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

        total = loss_pocket + loss_mol + loss_rank
        return {"loss": total, "loss_pocket": loss_pocket, "loss_mol": loss_mol,
                "loss_rank": loss_rank, "sim_masked": sim_masked}

    def compute_loss(self, mol_emb, poc_emb, prot_emb,
                     batch_list, act_list,
                     uniprot_poc=None, uniprot_mol=None,
                     pocket_lig_smiles=None, lig_smiles=None):
        kappa = self.curv.exp().detach()
        logit_scale = self.logit_scale.exp().detach()
        B = poc_emb.size(0)

        # === Cone Hierarchy Loss ===
        # Match paper three_hybrid_loss.py:73-74 — drop index 0 of space dim.
        poc_space = poc_emb[:, 1:]
        lig_space = mol_emb[:, 1:]
        poc_idx = []
        for i, (s, e) in enumerate(batch_list):
            poc_idx += [i] * (e - s)
        poc_idx = torch.tensor(poc_idx, device=poc_emb.device)

        poc_sel = poc_space[poc_idx]
        dist_mat = L.pairwise_dist(poc_sel, lig_space, curv=kappa)
        dist = dist_mat.diagonal()
        device = dist.device
        phi = L.oxy_angle(lig_space, poc_space[poc_idx], curv=kappa)
        omega = L.half_aperture(poc_space[poc_idx], curv=kappa)
        act_flat = torch.tensor(
            [x for sub in act_list for x in sub],
            device=poc_emb.device, dtype=torch.float32,
        )
        bounds = self.bounds.to(poc_emb.device)
        bucket = torch.bucketize(act_flat, bounds)
        r_k = self.chl_r0 + bucket.float() * self.chl_dr
        eta_k = self.chl_eta0 - bucket.float() * self.chl_deta
        Nl = dist.size(0)
        L_rad = F.relu(dist - r_k).sum() / math.sqrt(Nl)
        L_ang = F.relu(phi - eta_k * omega).sum() / math.sqrt(Nl)
        loss_cone = self.lambda_rad * L_rad + self.lambda_ang * L_ang

        # Angular regularization
        m_margin = 0.15
        R_ang = F.relu(phi - eta_k * omega + m_margin).sum() / math.sqrt(Nl)

        # Heterogeneous ranking regularization
        R_het = torch.zeros(1, device=device)
        cnt_het = 0
        beta = 80.5
        offset = 0
        for i_poc, (s, e) in enumerate(batch_list):
            L_i = e - s
            if L_i < 1:
                continue
            d_i = dist[offset : offset + L_i].detach()
            rank = (d_i.unsqueeze(0) < d_i.unsqueeze(1)).float().sum(1) + 1
            w = torch.exp(-beta * (rank - 1) / L_i)
            logits_row = torch.matmul(poc_space[i_poc : i_poc + 1], lig_space.T) * logit_scale
            row_probs = F.softmax(logits_row[0, s:e], dim=-1)
            pos_mask = act_flat[offset : offset + L_i] < 5
            if pos_mask.any():
                R_het += -(w[pos_mask] * row_probs[pos_mask].log()).sum() / (w[pos_mask].sum() + 1e-9)
                cnt_het += 1
            offset += L_i
        R_het = R_het / max(cnt_het, 1)
        loss_reg = self.lambda_het * R_het + self.lambda_angu * R_ang

        # === HCC for both pathways ===
        loss_dict_poc = self._compute_hcc_pair(
            poc_emb, mol_emb, batch_list, act_list,
            uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles, logit_scale,
        )
        loss_dict_prot = self._compute_hcc_pair(
            prot_emb, mol_emb, batch_list, act_list,
            uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles, logit_scale,
        )

        loss_hcc = loss_dict_poc["loss"] + loss_dict_prot["loss"]
        total_loss = loss_hcc + self.gamma_chl * loss_cone + loss_reg

        return total_loss, {
            "loss": total_loss.item(),
            "loss_hcc": loss_hcc.item(),
            "loss_cone": loss_cone.item(),
            "loss_reg": loss_reg.item(),
            "sim_masked": loss_dict_poc["sim_masked"],
        }

    def score(self, mol_reps, pocket_reps, prot_reps=None):
        """Score using full 128-d hyperbolic embedding (paper convention).

        NOTE: this method is dead code — virtual-screening evaluation goes
        through unimol/tasks/test_task.py:test_dude_target which scores via
        full-embedding dot product (test_task.py:797-803, no [:, 1:] slice).
        We keep this implementation aligned with that upstream convention so
        any future caller stays consistent with paper evaluation.
        """
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
