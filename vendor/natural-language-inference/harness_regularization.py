#!/usr/bin/env python3
"""Full-scale three-setting harness for the regularization surface."""
from __future__ import annotations

import argparse
import gc
import time

import common

LEVELS = {
    "standard": (0.1, 0.01),
    "none": (0.0, 0.0),
    "heavy": (0.7, 0.3),
}


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
        args.seed, task_id="nli-regularization", surface="regularization"
    )
    train_rows = common.load_training_data()
    build = common.load_surface(args.solution, "build_reg")
    config = common.require_surface_config(build(), {"reg"}, surface="build_reg")
    mode = common.require_surface_choice(
        config["reg"], "reg", set(LEVELS) | {"majority"}, surface="build_reg"
    )
    print(f"NLI_POLICY mode={mode}", flush=True)
    if mode == "majority":
        completed = common.evaluate_majority_all_domains(train_rows)
        common.emit_final_completion(
            seed=args.seed, elapsed=time.time() - started, completed_rows=completed
        )
        return

    dropout, weight_decay = LEVELS[mode]
    print(
        f"NLI_REGULARIZATION dropout={dropout:.8g} "
        f"weight_decay={weight_decay:.8g}",
        flush=True,
    )
    tokenizer = common.build_tokenizer()
    train_features, train_labels = common.encode_pairs_cross(train_rows, tokenizer)
    del train_rows
    gc.collect()
    model = common.CrossEncoderNLI(common.NUM_LABELS, device, dropout=dropout)
    common.emit_model_proof(model)
    common.train_model(
        model, tokenizer, train_features, train_labels,
        collate_fn=common.collate_cross, encoder_lr=common.ENCODER_LR,
        head_lr=common.HEAD_LR, seed=args.seed, verbose_tag=mode,
        weight_decay=weight_decay,
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
