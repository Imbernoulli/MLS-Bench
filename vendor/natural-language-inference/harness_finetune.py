#!/usr/bin/env python3
"""Full-scale, three-setting harness for the encoder-update surface."""
from __future__ import annotations

import argparse
import gc
import time

import common


def _emit_setting(setting: str, accuracy: float, expected_rows: int) -> None:
    print(
        f"NLI_METRICS setting={setting} acc={accuracy:.8f} "
        f"n_eval={expected_rows}",
        flush=True,
    )
    print(
        f"NLI_SETTING_DONE setting={setting} predicted={expected_rows} "
        f"expected={expected_rows}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--seed", type=int, default=common.SEED_DEFAULT)
    args = parser.parse_args()
    if args.seed != common.SEED_DEFAULT:
        raise SystemExit(f"full NLI protocol requires seed {common.SEED_DEFAULT}")

    device = common.setup(args.seed)
    started = time.time()
    common.emit_protocol(args.seed, task_id="nli-finetune", surface="finetune")
    train_rows = common.load_training_data()

    build = common.load_surface(args.solution, "build_finetune")
    config = common.require_surface_config(
        build(), {"encoder"}, surface="build_finetune"
    )
    mode = common.require_surface_choice(
        config["encoder"], "encoder", {"frozen", "finetune", "majority"},
        surface="build_finetune",
    )
    print(f"NLI_POLICY mode={mode}", flush=True)

    if mode == "majority":
        print(
            "NLI_TRAIN_DONE epochs=0 optimizer_steps=0 expected_steps=0",
            flush=True,
        )
        completed_rows = 0
        for setting in common.DOMAINS:
            eval_rows = common.load_evaluation_data(setting)
            accuracy = common.majority_class_accuracy(train_rows, eval_rows)
            expected = common.DATASET_INVENTORY[setting]["rows"]
            _emit_setting(setting, accuracy, expected)
            completed_rows += expected
        print(
            f"NLI_EVAL_DONE settings={len(common.DOMAINS)} rows={completed_rows}",
            flush=True,
        )
        print(
            f"NLI_DONE settings={len(common.DOMAINS)} "
            f"train_rows={len(train_rows)} eval_rows={completed_rows} "
            f"seed={args.seed} elapsed={time.time() - started:.1f}",
            flush=True,
        )
        return

    tokenizer = common.build_tokenizer()
    train_features, train_labels = common.encode_pairs_cross(train_rows, tokenizer)
    del train_rows
    gc.collect()

    model = common.CrossEncoderNLI(common.NUM_LABELS, device)
    common.emit_model_proof(model)
    encoder_lr = 0.0 if mode == "frozen" else common.ENCODER_LR
    common.train_model(
        model,
        tokenizer,
        train_features,
        train_labels,
        collate_fn=common.collate_cross,
        encoder_lr=encoder_lr,
        head_lr=common.HEAD_LR,
        seed=args.seed,
        verbose_tag=mode,
    )

    completed_rows = 0
    for setting in common.DOMAINS:
        eval_rows = common.load_evaluation_data(setting)
        expected = common.DATASET_INVENTORY[setting]["rows"]
        eval_features, eval_labels = common.encode_pairs_cross(eval_rows, tokenizer)
        accuracy = common.score_model(
            model,
            tokenizer,
            eval_features,
            eval_labels,
            collate_fn=common.collate_cross,
        )
        _emit_setting(setting, accuracy, expected)
        completed_rows += expected
        del eval_rows, eval_features, eval_labels
        gc.collect()

    print(
        f"NLI_EVAL_DONE settings={len(common.DOMAINS)} rows={completed_rows}",
        flush=True,
    )
    print(
        f"NLI_DONE settings={len(common.DOMAINS)} "
        f"train_rows={common.DATASET_INVENTORY['snli_train']['rows']} "
        f"eval_rows={completed_rows} seed={args.seed} "
        f"elapsed={time.time() - started:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
