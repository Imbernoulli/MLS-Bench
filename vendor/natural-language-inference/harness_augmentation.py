#!/usr/bin/env python3
"""Full-scale three-setting harness for the train-time augmentation surface."""
from __future__ import annotations

import argparse
import gc
import time

import common


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
        args.seed, task_id="nli-augmentation", surface="augmentation"
    )
    train_rows = common.load_training_data()
    build = common.load_surface(args.solution, "build_augment")
    config = common.require_surface_config(
        build(), {"augment"}, surface="build_augment"
    )
    mode = common.require_surface_choice(
        config["augment"], "augment", {"none", "swap", "negation", "majority"},
        surface="build_augment",
    )
    print(f"NLI_POLICY mode={mode}", flush=True)
    if mode == "majority":
        completed = common.evaluate_majority_all_domains(train_rows)
        common.emit_final_completion(
            seed=args.seed, elapsed=time.time() - started, completed_rows=completed
        )
        return

    transformed_rows = common.augment_train(train_rows, mode)
    if len(transformed_rows) != len(train_rows):
        raise RuntimeError("NLI augmentation changed the fixed update inventory")
    tokenizer = common.build_tokenizer()
    train_features, train_labels = common.encode_pairs_cross(
        transformed_rows, tokenizer
    )
    del transformed_rows, train_rows
    gc.collect()
    model = common.CrossEncoderNLI(common.NUM_LABELS, device)
    common.emit_model_proof(model)
    common.train_model(
        model, tokenizer, train_features, train_labels,
        collate_fn=common.collate_cross, encoder_lr=common.ENCODER_LR,
        head_lr=common.HEAD_LR, seed=args.seed, verbose_tag=mode,
    )
    completed = common.evaluate_all_domains(
        model, tokenizer,
        encode_rows=lambda rows: common.encode_pairs_cross(rows, tokenizer),
        collate_fn=common.collate_cross,
    )
    common.emit_final_completion(
        seed=args.seed, elapsed=time.time() - started, completed_rows=completed
    )


if __name__ == "__main__":
    main()
