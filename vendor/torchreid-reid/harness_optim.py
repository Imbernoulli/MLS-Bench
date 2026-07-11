#!/usr/bin/env python3
"""Fixed person re-identification evaluation harness.

The selected solution surface is loaded through its public contract. Active-surface
load, runtime, shape, type, completeness, or numerical failures invalidate the run.
Other pipeline components remain fixed for a given task.
"""
from __future__ import annotations

import argparse
import time

import common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-instances", type=int, default=4)
    args = ap.parse_args()

    common.set_seeds(args.seed)
    t0 = time.perf_counter()

    import torch
    from torch.utils.data import DataLoader
    from torchreid.data.sampler import RandomIdentitySampler
    from torchreid.losses import CrossEntropyLoss, TripletLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    common.emit_fullscale_protocol(
        task_id=args.task_id, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
    )

    items, pid2label, num_ids = common.load_train_items()
    rl_items = [(p, pid2label[pid], c) for (p, pid, c) in items]
    print(f"REID_DATA train_ids={num_ids} train_imgs={len(items)}", flush=True)

    ds = common.ReidImageDataset(items, common.base_train_transform(), pid2label)
    sampler = RandomIdentitySampler(rl_items, args.batch_size, args.num_instances)
    loader = common.fullscale_loader(ds, sampler, batch_size=args.batch_size)

    model = common.build_backbone(num_ids, loss="triplet").to(device)

    if args.task_id == "reid-lr-schedule":
        opt = common.build_optimizer(model)
        lr_at_step = common.load_surface(
            args.solution, "build_lr_schedule"
        )(common.EXPECTED_TOTAL_STEPS)
        if not callable(lr_at_step):
            print("REID_SURFACE_FALLBACK name=schedule reason=not_callable", flush=True)
            raise TypeError("build_lr_schedule must return a callable")
        epoch_scheduler = None
        surface_name = getattr(lr_at_step, "name", "schedule")
    elif args.task_id == "reid-optimizer":
        opt = common.load_surface(args.solution, "build_optimizer")(model.parameters())
        lr_at_step = None
        epoch_scheduler = None
        surface_name = type(opt).__name__
    else:
        raise ValueError(f"unsupported optimizer harness task: {args.task_id!r}")
    if not isinstance(opt, torch.optim.Optimizer):
        print("REID_SURFACE_FALLBACK name=optimizer reason=bad_contract", flush=True)
        raise TypeError("optimizer surface returned an invalid optimizer")
    if args.task_id == "reid-optimizer":
        epoch_scheduler = common.default_epoch_scheduler(opt)
    expected_params = {id(p) for p in model.parameters() if p.requires_grad}
    actual_params = {id(p) for pg in opt.param_groups for p in pg.get("params", [])}
    if actual_params != expected_params:
        print("REID_SURFACE_FALLBACK name=optimizer reason=wrong_parameters", flush=True)
        raise ValueError("optimizer must contain all trainable model parameters")
    print(f"REID_OPTIM {surface_name}", flush=True)

    xent = CrossEntropyLoss(num_classes=num_ids, use_gpu=(device == "cuda"))
    triplet = TripletLoss(margin=0.3)

    def loss_step(imgs, labels):
        logits, feats = model(imgs)
        loss = xent(logits, labels) + triplet(feats, labels)
        return logits, feats, loss

    total_steps, train_samples = common.run_fullscale_training(
        model, loader, opt, loss_step, epochs=args.epochs,
        epoch_scheduler=epoch_scheduler,
        lr_at_step=lr_at_step,
    )
    common.finish_fullscale_evaluation(
        model, t0, task_id=args.task_id, total_steps=total_steps,
        train_samples=train_samples,
    )


if __name__ == "__main__":
    main()
