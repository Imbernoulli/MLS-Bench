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

    # Agent-controlled train-time augmentation.
    train_tf = common.load_surface(args.solution, "build_train_transform")(
        common.IMG_H, common.IMG_W, common.IMAGENET_MEAN, common.IMAGENET_STD
    )
    if not callable(train_tf):
        print("REID_SURFACE_FALLBACK name=augmentation reason=not_callable", flush=True)
        raise TypeError("build_train_transform must return a callable transform")
    print(f"REID_AUG {getattr(train_tf, 'name', type(train_tf).__name__)}", flush=True)

    ds = common.ReidImageDataset(items, train_tf, pid2label)
    sampler = RandomIdentitySampler(rl_items, args.batch_size, args.num_instances)
    loader = common.fullscale_loader(ds, sampler, batch_size=args.batch_size)

    model = common.build_backbone(num_ids, loss="triplet").to(device)
    xent = CrossEntropyLoss(num_classes=num_ids, use_gpu=(device == "cuda"))
    triplet = TripletLoss(margin=0.3)
    opt = common.build_optimizer(model)
    scheduler = common.default_epoch_scheduler(opt)

    def loss_step(imgs, labels):
        logits, feats = model(imgs)
        loss = xent(logits, labels) + triplet(feats, labels)
        return logits, feats, loss

    total_steps, train_samples = common.run_fullscale_training(
        model, loader, opt, loss_step, epochs=args.epochs,
        epoch_scheduler=scheduler,
    )
    common.finish_fullscale_evaluation(
        model, t0, task_id=args.task_id, total_steps=total_steps,
        train_samples=train_samples,
    )


if __name__ == "__main__":
    main()
