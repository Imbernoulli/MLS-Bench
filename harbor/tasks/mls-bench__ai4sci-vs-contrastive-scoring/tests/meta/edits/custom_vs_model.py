"""Custom virtual screening model wrapper (FIXED — do not modify).

This model wraps UniMol + ESM2 backbone encoders and delegates
projection + embedding logic to the editable CustomScoring module.
Backbones are FINE-TUNED jointly with the scoring head to match the
HypSeek training protocol (paper uses joint end-to-end training).
"""

import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.data import Dictionary
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from transformers import AutoTokenizer, AutoModelForMaskedLM
from unimol.models.unimol import NonLinearHead, UniMolModel, base_architecture
from unimol.custom_scoring import CustomScoring


@register_model("custom_vs_model")
class CustomVSModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--mol-pooler-dropout", type=float, metavar="D",
                            help="dropout probability in the masked_lm pooler layers")
        parser.add_argument("--pocket-pooler-dropout", type=float, metavar="D",
                            help="dropout probability in the masked_lm pooler layers")
        parser.add_argument("--pocket-encoder-layers", type=int,
                            help="pocket encoder layers")
        parser.add_argument("--recycling", type=int, default=1,
                            help="recycling nums of decoder")
        parser.add_argument("--aperture-eta", type=float, default=1.2)
        parser.add_argument("--curv-init", type=float, default=1.0,
                            help="initial curv")
        parser.add_argument("--learn-curv", action="store_true",
                            help="learnable curvature (passed to CustomScoring)")

    def __init__(self, args, mol_dictionary: Dictionary, pocket_dictionary: Dictionary):
        super().__init__()
        custom_vs_architecture(args)
        self.args = args

        # === Backbone encoders (TRAINABLE — fine-tuned jointly) ===
        # NOTE: HypSeek paper trains backbones end-to-end. Freezing them
        # collapses absolute metrics ~4x below paper values and breaks the
        # relative ordering of hyperbolic vs. Euclidean baselines, because
        # the projection heads alone cannot adapt features to the target
        # (hyperbolic / Euclidean) geometry.
        self.mol_model = UniMolModel(args.mol, mol_dictionary)
        self.pocket_model = UniMolModel(args.pocket, pocket_dictionary)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/esm2_t12_35M_UR50D", use_fast=False
        )
        self.protein_model = AutoModelForMaskedLM.from_pretrained(
            "facebook/esm2_t12_35M_UR50D"
        )

        # === Custom scoring module (TRAINABLE) ===
        self.scoring = CustomScoring(
            mol_dim=args.mol.encoder_embed_dim,
            pocket_dim=args.pocket.encoder_embed_dim,
            protein_dim=self.protein_model.config.hidden_size,
            embed_dim=128,
        )

    @classmethod
    def build_model(cls, args, task):
        return cls(args, task.dictionary, task.pocket_dictionary)

    # --- Backbone feature extraction (no gradients) ---

    def get_dist_features(self, dist, et, flag):
        model = self.mol_model if flag == "mol" else self.pocket_model
        n_node = dist.size(-1)
        gbf_feature = model.gbf(dist, et)
        gbf_result = model.gbf_proj(gbf_feature)
        graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        return graph_attn_bias

    def _encode_mol(self, mol_src_tokens, mol_src_distance, mol_src_edge_type):
        padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        x = self.mol_model.embed_tokens(mol_src_tokens)
        attn_bias = self.get_dist_features(mol_src_distance, mol_src_edge_type, "mol")
        outputs = self.mol_model.encoder(x, padding_mask=padding_mask, attn_mask=attn_bias)
        return outputs[0][:, 0, :]  # CLS token [B, 512]

    def _encode_pocket(self, pocket_src_tokens, pocket_src_distance, pocket_src_edge_type):
        padding_mask = pocket_src_tokens.eq(self.pocket_model.padding_idx)
        x = self.pocket_model.embed_tokens(pocket_src_tokens)
        attn_bias = self.get_dist_features(pocket_src_distance, pocket_src_edge_type, "pocket")
        outputs = self.pocket_model.encoder(x, padding_mask=padding_mask, attn_mask=attn_bias)
        return outputs[0][:, 0, :]  # CLS token [B, 512]

    def _encode_protein(self, protein_sequences):
        inputs = self.tokenizer(
            protein_sequences, return_tensors="pt",
            padding="longest", truncation=True, max_length=512,
        )
        device = next(self.protein_model.parameters()).device
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        prot_outputs = self.protein_model(**inputs, output_hidden_states=True)
        prot_hidden = prot_outputs.hidden_states[-1]
        return prot_hidden[:, 0, :]  # CLS token [B, 480]

    # --- Forward methods (delegate to CustomScoring) ---

    def forward(
        self,
        mol_src_tokens, mol_src_distance, mol_src_edge_type,
        pocket_src_tokens, pocket_src_distance, pocket_src_edge_type,
        protein_sequences,
        encode=False, masked_tokens=None, features_only=True, is_train=True,
        **kwargs,
    ):
        mol_feat = self._encode_mol(mol_src_tokens, mol_src_distance, mol_src_edge_type)
        poc_feat = self._encode_pocket(pocket_src_tokens, pocket_src_distance, pocket_src_edge_type)
        prot_feat = self._encode_protein(protein_sequences)

        mol_emb = self.scoring.project_mol(mol_feat)
        poc_emb = self.scoring.project_pocket(poc_feat)
        prot_emb = self.scoring.project_protein(prot_feat)

        return prot_emb, poc_emb, mol_emb

    def mol_forward(self, mol_src_tokens, mol_src_distance, mol_src_edge_type, **kwargs):
        mol_feat = self._encode_mol(mol_src_tokens, mol_src_distance, mol_src_edge_type)
        return self.scoring.project_mol(mol_feat)

    def pocket_forward(self, pocket_src_tokens, pocket_src_distance,
                       pocket_src_edge_type, **kwargs):
        poc_feat = self._encode_pocket(pocket_src_tokens, pocket_src_distance, pocket_src_edge_type)
        return self.scoring.project_pocket(poc_feat)

    def protein_forward(self, protein_sequences, **kwargs):
        prot_feat = self._encode_protein(protein_sequences)
        return self.scoring.project_protein(prot_feat)

    def set_num_updates(self, num_updates):
        self._num_updates = num_updates

    def get_num_updates(self):
        return getattr(self, "_num_updates", 0)


@register_model_architecture("custom_vs_model", "custom_vs_model")
def custom_vs_architecture(args):
    parser = argparse.ArgumentParser()
    args.mol = parser.parse_args([])
    args.pocket = parser.parse_args([])

    args.mol.encoder_layers = getattr(args, "mol_encoder_layers", 15)
    args.mol.encoder_embed_dim = getattr(args, "mol_encoder_embed_dim", 512)
    args.mol.encoder_ffn_embed_dim = getattr(args, "mol_encoder_ffn_embed_dim", 2048)
    args.mol.encoder_attention_heads = getattr(args, "mol_encoder_attention_heads", 64)
    args.mol.dropout = getattr(args, "mol_dropout", 0.1)
    args.mol.emb_dropout = getattr(args, "mol_emb_dropout", 0.1)
    args.mol.attention_dropout = getattr(args, "mol_attention_dropout", 0.1)
    args.mol.activation_dropout = getattr(args, "mol_activation_dropout", 0.0)
    args.mol.pooler_dropout = getattr(args, "mol_pooler_dropout", 0.0)
    args.mol.max_seq_len = getattr(args, "mol_max_seq_len", 512)
    args.mol.activation_fn = getattr(args, "mol_activation_fn", "gelu")
    args.mol.pooler_activation_fn = getattr(args, "mol_pooler_activation_fn", "tanh")
    args.mol.post_ln = getattr(args, "mol_post_ln", False)
    args.mol.masked_token_loss = -1.0
    args.mol.masked_coord_loss = -1.0
    args.mol.masked_dist_loss = -1.0
    args.mol.x_norm_loss = -1.0
    args.mol.delta_pair_repr_norm_loss = -1.0

    args.pocket.encoder_layers = getattr(args, "pocket_encoder_layers", 15)
    args.pocket.encoder_embed_dim = getattr(args, "pocket_encoder_embed_dim", 512)
    args.pocket.encoder_ffn_embed_dim = getattr(args, "pocket_encoder_ffn_embed_dim", 2048)
    args.pocket.encoder_attention_heads = getattr(args, "pocket_encoder_attention_heads", 64)
    args.pocket.dropout = getattr(args, "pocket_dropout", 0.1)
    args.pocket.emb_dropout = getattr(args, "pocket_emb_dropout", 0.1)
    args.pocket.attention_dropout = getattr(args, "pocket_attention_dropout", 0.1)
    args.pocket.activation_dropout = getattr(args, "pocket_activation_dropout", 0.0)
    args.pocket.pooler_dropout = getattr(args, "pocket_pooler_dropout", 0.0)
    args.pocket.max_seq_len = getattr(args, "pocket_max_seq_len", 512)
    args.pocket.activation_fn = getattr(args, "pocket_activation_fn", "gelu")
    args.pocket.pooler_activation_fn = getattr(args, "pocket_pooler_activation_fn", "tanh")
    args.pocket.post_ln = getattr(args, "pocket_post_ln", False)
    args.pocket.masked_token_loss = -1.0
    args.pocket.masked_coord_loss = -1.0
    args.pocket.masked_dist_loss = -1.0
    args.pocket.x_norm_loss = -1.0
    args.pocket.delta_pair_repr_norm_loss = -1.0

    base_architecture(args)
