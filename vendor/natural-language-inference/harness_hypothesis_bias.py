#!/usr/bin/env python3
"""Full-scale three-setting harness for the premise-availability surface."""
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
        args.seed, task_id="nli-hypothesis-bias", surface="hypothesis_bias"
    )
    train_rows = common.load_training_data()
    build = common.load_surface(args.solution, "build_bias")
    config = build()
    if isinstance(config, dict) and config.get("mode") == "majority":
        common.require_surface_config(config, {"mode"}, surface="build_bias")
        mode = "majority"
        use_premise = False
    else:
        config = common.require_surface_config(
            config, {"use_premise"}, surface="build_bias"
        )
        use_premise = config["use_premise"]
        if not isinstance(use_premise, bool):
            print("SURFACE_ERROR build_bias use_premise must be bool", flush=True)
            raise TypeError("use_premise must be bool")
        mode = "premise" if use_premise else "hyponly"
    print(f"NLI_POLICY mode={mode}", flush=True)
    if mode == "majority":
        completed = common.evaluate_majority_all_domains(train_rows)
        common.emit_final_completion(
            seed=args.seed, elapsed=time.time() - started, completed_rows=completed
        )
        return

    mask_premise = not use_premise
    tokenizer = common.build_tokenizer()
    train_features, train_labels = common.encode_pairs_cross(
        train_rows, tokenizer, mask_premise=mask_premise
    )
    del train_rows
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
        encode_rows=lambda rows: common.encode_pairs_cross(
            rows, tokenizer, mask_premise=mask_premise
        ),
        collate_fn=common.collate_cross,
    )
    common.emit_final_completion(
        seed=args.seed, elapsed=time.time() - started, completed_rows=completed
    )


if __name__ == "__main__":
    main()
