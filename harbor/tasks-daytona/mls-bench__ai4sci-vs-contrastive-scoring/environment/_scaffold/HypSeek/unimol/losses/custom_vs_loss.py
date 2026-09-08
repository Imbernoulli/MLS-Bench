"""Custom virtual screening loss wrapper (FIXED — do not modify).

Delegates all loss computation to model.scoring.compute_loss().
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from unicore.losses import UnicoreLoss, register_loss
from unicore import metrics
from sklearn.metrics import roc_auc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC


def calculate_bedroc(y_true, y_score, alpha=80.5):
    scores = np.expand_dims(y_score, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    scores = np.concatenate((scores, y_true), axis=1)
    scores = scores[scores[:, 0].argsort()[::-1]]
    return CalcBEDROC(scores, 1, alpha)


@register_loss("custom_vs_loss")
class CustomVSLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, fix_encoder=False):
        h_prot, h_poc, h_mol = model(
            **sample["pocket"]["net_input"],
            **sample["lig"]["net_input"],
            protein_sequences=sample["protein"],
            features_only=True,
            fix_encoder=fix_encoder,
            is_train=self.training,
        )

        B = h_poc.size(0)

        loss, log_dict = model.scoring.compute_loss(
            mol_emb=h_mol,
            poc_emb=h_poc,
            prot_emb=h_prot,
            batch_list=sample["batch_list"],
            act_list=sample["act_list"],
            uniprot_poc=sample.get("uniprot_list"),
            uniprot_mol=sample.get("lig_uniprot_list"),
            pocket_lig_smiles=sample.get("pocket_lig_smiles"),
            lig_smiles=sample["lig"]["smi_name"],
        )

        if self.training:
            logging_output = {
                "loss": loss.data,
                "sample_size": B,
            }
            # Include any extra logging from CustomScoring
            for k, v in log_dict.items():
                if k != "sim_masked" and isinstance(v, (int, float)):
                    logging_output[k] = v
            # TRAIN_METRICS
            print(f"TRAIN_METRICS loss={loss.item():.6f}", flush=True)
        else:
            sim_masked = log_dict.get("sim_masked")
            if sim_masked is None:
                sim_masked = h_poc @ h_mol.T
            sample_size = B
            targets = torch.arange(sample_size, dtype=torch.long, device=sim_masked.device)
            probs = F.softmax(sim_masked[:, :sample_size].float(), dim=-1)
            logging_output = {
                "loss": torch.tensor(0.0, device=sim_masked.device),
                "prob": probs.data,
                "target": targets,
                "smi_name": sample["lig"]["smi_name"],
                "sample_size": sample_size,
            }

        return loss, B, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid", args=None):
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        if sample_size == 0:
            return

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)

        if "train" in split:
            return

        # Validation: compute BEDROC and AUC on the similarity matrix
        valid_set = getattr(args, "valid_set", "CASF")
        if valid_set in ["FEP", "TIME", "TYK2", "OOD", "DEMO"]:
            return

        prob_list, tgt_list = [], []
        acc_sum = 0
        for log in logging_outputs:
            prob = log.get("prob")
            tgt = log.get("target")
            if prob is None or tgt is None:
                continue
            acc_sum += (prob.argmax(dim=-1) == tgt).sum().item()
            prob_list.append(prob)
            tgt_list.append(tgt)

        if len(prob_list) == 0:
            return
        if len(prob_list) > 1:
            prob_list = prob_list[:-1]
            tgt_list = tgt_list[:-1]

        probs = torch.cat(prob_list, dim=0)
        targets = torch.cat(tgt_list, dim=0)
        targets = targets[: len(probs)]

        metrics.log_scalar(f"{split}_acc", acc_sum / sample_size, sample_size, round=3)

        bedroc_list, auc_list = [], []
        for i in range(len(probs)):
            prob = probs[i]
            target = targets[i]
            label = torch.zeros_like(prob)
            label[target] = 1.0
            try:
                cur_auc = roc_auc_score(label.cpu(), prob.cpu())
                auc_list.append(cur_auc)
            except ValueError:
                pass
            bedroc = calculate_bedroc(label.cpu().numpy(), prob.cpu().numpy(), 80.5)
            bedroc_list.append(bedroc)

        if bedroc_list:
            metrics.log_scalar(f"{split}_bedroc", np.mean(bedroc_list), sample_size, round=3)
            metrics.log_scalar("valid_bedroc", np.mean(bedroc_list), sample_size, round=3)
        if auc_list:
            metrics.log_scalar(f"{split}_auc", np.mean(auc_list), sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed(is_train):
        return is_train
