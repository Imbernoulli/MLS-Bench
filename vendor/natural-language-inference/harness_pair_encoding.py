#!/usr/bin/env python3
"""Full-scale three-setting harness for the pair-encoding surface."""
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
    train_rows = common.load_training_data()
    build = common.load_surface(args.solution, "build_encoding")
    config = common.require_surface_config(
        build(), {"encoding"}, surface="build_encoding"
    )
    mode = common.require_surface_choice(
        config["encoding"], "encoding", {"cross", "siamese", "majority"},
        surface="build_encoding",
    )
    max_length = common.MAX_LEN if mode == "siamese" else common.MAX_SEQUENCE_LENGTH
    common.emit_protocol(
        args.seed, task_id="nli-pair-encoding", surface="pair_encoding",
        max_length=max_length,
    )
    print(f"NLI_POLICY mode={mode}", flush=True)
    if mode == "majority":
        completed = common.evaluate_majority_all_domains(train_rows)
        common.emit_final_completion(
            seed=args.seed, elapsed=time.time() - started, completed_rows=completed
        )
        return

    tokenizer = common.build_tokenizer()
    if mode == "cross":
        train_features, train_labels = common.encode_pairs_cross(train_rows, tokenizer)
        model = common.CrossEncoderNLI(common.NUM_LABELS, device)
        collate_fn = common.collate_cross
        encode_eval = lambda rows: common.encode_pairs_cross(rows, tokenizer)
    else:
        train_features, train_labels = common.encode_pairs_siamese(train_rows, tokenizer)
        model = common.SiameseNLI(common.NUM_LABELS, "infersent", device)
        collate_fn = common.collate_siamese
        encode_eval = lambda rows: common.encode_pairs_siamese(rows, tokenizer)
    del train_rows
    gc.collect()

    common.emit_model_proof(model)
    common.train_model(
        model, tokenizer, train_features, train_labels, collate_fn=collate_fn,
        encoder_lr=common.ENCODER_LR, head_lr=common.HEAD_LR, seed=args.seed,
        verbose_tag=mode, sequence_length=max_length,
    )
    completed = common.evaluate_all_domains(
        model, tokenizer, encode_rows=encode_eval, collate_fn=collate_fn
    )
    common.emit_final_completion(
        seed=args.seed, elapsed=time.time() - started, completed_rows=completed
    )


if __name__ == "__main__":
    main()
