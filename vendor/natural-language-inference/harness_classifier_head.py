#!/usr/bin/env python3
"""Full-scale three-setting harness for the classifier-head surface."""
from __future__ import annotations

import argparse
import gc
import time

import common

MLP_HIDDEN = 512


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--seed", type=int, default=common.SEED_DEFAULT)
    args = parser.parse_args()
    if args.seed != common.SEED_DEFAULT:
        raise SystemExit(f"full NLI protocol requires seed {common.SEED_DEFAULT}")

    device = common.setup(args.seed)
    started = time.time()
    common.emit_protocol(
        args.seed, task_id="nli-classifier-head", surface="classifier_head",
        max_length=common.MAX_LEN,
    )
    train_rows = common.load_training_data()
    build = common.load_surface(args.solution, "build_classifier")
    config = common.require_surface_config(
        build(), {"head"}, surface="build_classifier"
    )
    mode = common.require_surface_choice(
        config["head"], "head", {"linear", "mlp", "majority"},
        surface="build_classifier",
    )
    print(f"NLI_POLICY mode={mode}", flush=True)
    if mode == "majority":
        completed = common.evaluate_majority_all_domains(train_rows)
        common.emit_final_completion(
            seed=args.seed, elapsed=time.time() - started, completed_rows=completed
        )
        return

    tokenizer = common.build_tokenizer()
    train_features, train_labels = common.encode_pairs_siamese(train_rows, tokenizer)
    del train_rows
    gc.collect()
    model = common.SiameseNLI(
        common.NUM_LABELS, "infersent", device,
        head_hidden=MLP_HIDDEN if mode == "mlp" else 0,
    )
    common.emit_model_proof(model)
    common.train_model(
        model, tokenizer, train_features, train_labels,
        collate_fn=common.collate_siamese, encoder_lr=common.ENCODER_LR,
        head_lr=common.HEAD_LR, seed=args.seed, verbose_tag=mode,
        sequence_length=common.MAX_LEN,
    )
    completed = common.evaluate_all_domains(
        model, tokenizer,
        encode_rows=lambda rows: common.encode_pairs_siamese(rows, tokenizer),
        collate_fn=common.collate_siamese,
    )
    common.emit_final_completion(
        seed=args.seed, elapsed=time.time() - started, completed_rows=completed
    )


if __name__ == "__main__":
    main()
