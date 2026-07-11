#!/usr/bin/env python3
"""Full-scale three-setting harness for the class-weighting surface."""
from __future__ import annotations

import argparse
import gc
import math
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
        args.seed, task_id="nli-class-weighting", surface="class_weighting"
    )
    train_rows = common.load_training_data()
    build = common.load_surface(args.solution, "build_weighting")
    config = common.require_surface_config(build(), {"weights"}, surface="build_weighting")
    raw_weights = config["weights"]
    if not isinstance(raw_weights, (list, tuple)) or len(raw_weights) != common.NUM_LABELS:
        print("SURFACE_ERROR build_weighting: weights must contain three values", flush=True)
        raise TypeError("weights must contain one value per NLI class")
    weights = []
    for value in raw_weights:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or not 0.25 <= float(value) <= 2.0):
            print("SURFACE_ERROR build_weighting: invalid class weight", flush=True)
            raise ValueError("class weights must be finite values in [0.25, 2.0]")
        weights.append(float(value))
    if not math.isclose(sum(weights), 3.0, rel_tol=0.0, abs_tol=1e-12):
        print("SURFACE_ERROR build_weighting: class weights must have mean one", flush=True)
        raise ValueError("class weights must have arithmetic mean one")
    mode = "class_weighting"
    print(f"NLI_POLICY mode={mode}", flush=True)
    print(
        "NLI_CLASS_WEIGHTS "
        f"entailment={weights[0]:.12g} neutral={weights[1]:.12g} "
        f"contradiction={weights[2]:.12g}",
        flush=True,
    )
    loss_fn = common.build_loss("ce", class_weights=weights, device=device)
    tokenizer = common.build_tokenizer()
    train_features, train_labels = common.encode_pairs_cross(train_rows, tokenizer)
    del train_rows
    gc.collect()
    model = common.CrossEncoderNLI(common.NUM_LABELS, device)
    common.emit_model_proof(model)
    common.train_model(
        model, tokenizer, train_features, train_labels,
        collate_fn=common.collate_cross, encoder_lr=common.ENCODER_LR,
        head_lr=common.HEAD_LR, seed=args.seed, verbose_tag=mode,
        loss_fn=loss_fn,
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
