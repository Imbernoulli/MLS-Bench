#!/usr/bin/env python3
"""Full-scale three-setting harness for the truncation-length surface."""
from __future__ import annotations

import argparse
import gc
import time

import common

MIN_LEN, MAX_CAP = 8, 128


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
    build = common.load_surface(args.solution, "build_truncation")
    config = build()
    if isinstance(config, dict) and config.get("mode") == "majority":
        common.require_surface_config(config, {"mode"}, surface="build_truncation")
        mode = "majority"
        max_length = common.MAX_SEQUENCE_LENGTH
    else:
        config = common.require_surface_config(
            config, {"max_len"}, surface="build_truncation"
        )
        max_length = config["max_len"]
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            print("SURFACE_ERROR build_truncation max_len must be an integer", flush=True)
            raise TypeError("max_len must be an integer")
        if not MIN_LEN <= max_length <= MAX_CAP:
            print(
                f"SURFACE_ERROR build_truncation max_len outside "
                f"[{MIN_LEN}, {MAX_CAP}]: {max_length}", flush=True
            )
            raise ValueError("max_len outside allowed range")
        mode = f"len{max_length}"

    common.emit_protocol(
        args.seed,
        task_id="nli-truncation",
        surface="truncation",
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
    train_features, train_labels = common.encode_pairs_cross(
        train_rows, tokenizer, max_length=max_length
    )
    del train_rows
    gc.collect()
    model = common.CrossEncoderNLI(common.NUM_LABELS, device)
    common.emit_model_proof(model)
    common.train_model(
        model, tokenizer, train_features, train_labels,
        collate_fn=common.collate_cross, encoder_lr=common.ENCODER_LR,
        head_lr=common.HEAD_LR, seed=args.seed, verbose_tag=mode,
        sequence_length=max_length,
    )
    completed = common.evaluate_all_domains(
        model, tokenizer,
        encode_rows=lambda rows: common.encode_pairs_cross(
            rows, tokenizer, max_length=max_length
        ),
        collate_fn=common.collate_cross,
    )
    common.emit_final_completion(
        seed=args.seed, elapsed=time.time() - started, completed_rows=completed
    )


if __name__ == "__main__":
    main()
