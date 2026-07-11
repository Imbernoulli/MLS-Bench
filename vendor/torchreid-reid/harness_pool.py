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

    if (args.seed, args.epochs, args.batch_size, args.num_instances) != (42, 60, 64, 4):
        raise ValueError("fixed protocol requires seed=42 epochs=60 batch=64 instances=4")

    common.set_seeds(args.seed)
    t0 = time.perf_counter()

    import torch
    from torch.utils.data import DataLoader
    from torchreid.data.sampler import RandomIdentitySampler
    from torchreid.losses import CrossEntropyLoss, TripletLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("full-scale re-ID verification requires one CUDA GPU")

    common.emit_fullscale_protocol(
        task_id=args.task_id, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, num_instances=args.num_instances,
    )

    items, pid2label, num_ids = common.load_train_items()
    rl_items = [(p, pid2label[pid], c) for (p, pid, c) in items]
    print(f"REID_DATA train_ids={num_ids} train_imgs={len(items)}", flush=True)

    ds = common.ReidImageDataset(items, common.base_train_transform(), pid2label)
    sampler = RandomIdentitySampler(rl_items, args.batch_size, args.num_instances)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=4, pin_memory=True, persistent_workers=True,
                        drop_last=True)
    if len(loader) < 150:
        raise RuntimeError(f"full training loader is unexpectedly short: {len(loader)}")

    backbone = common.build_backbone(num_ids, loss="triplet").to(device)
    feat_dim = backbone.feature_dim

    # Agent-controlled spatial pooling module.
    pool = common.load_surface(args.solution, "build_pooling")()
    if not isinstance(pool, torch.nn.Module):
        print("REID_SURFACE_FALLBACK name=pooling reason=not_module", flush=True)
        raise TypeError("build_pooling must return nn.Module")
    pool = pool.to(device)
    print(f"REID_POOL {getattr(pool, 'name', type(pool).__name__)}", flush=True)

    model = common.PoolModel(backbone, pool, feat_dim, num_ids).to(device)

    xent = CrossEntropyLoss(num_classes=num_ids, use_gpu=(device == "cuda"))
    triplet = TripletLoss(margin=0.3)
    opt = common.build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[40, 50], gamma=0.1
    )

    total_steps = 0
    train_samples = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        epoch_steps = 0
        for imgs, labels, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, feats = model(imgs)
            loss = xent(logits, labels) + triplet(feats, labels)
            common.validate_train_outputs(logits, feats, loss, imgs.shape[0])
            common.backward_and_step(loss, opt)

            running += float(loss.item())
            epoch_steps += 1
            total_steps += 1
            train_samples += int(imgs.shape[0])
        if epoch_steps < 150:
            raise RuntimeError(f"epoch {epoch} was incomplete: {epoch_steps} steps")
        scheduler.step()
        print(
            f"REID_EPOCH epoch={epoch} steps={epoch_steps} total_steps={total_steps} "
            f"loss={running / epoch_steps:.6f} lr={opt.param_groups[0]['lr']:.8g}",
            flush=True,
        )

    if (total_steps, train_samples) != (
        common.EXPECTED_TOTAL_STEPS,
        common.EXPECTED_TRAIN_SAMPLES,
    ):
        print(
            f"REID_PROTOCOL_ERROR stage=train_budget total_steps={total_steps} "
            f"train_samples={train_samples}",
            flush=True,
        )
        raise RuntimeError("training budget does not match the full protocol")
    print(
        f"REID_TRAIN_COMPLETE epochs={args.epochs} total_steps={total_steps} "
        f"train_samples={train_samples}",
        flush=True,
    )

    common.finish_fullscale_evaluation(
        model, t0, task_id=args.task_id, total_steps=total_steps,
        train_samples=train_samples,
    )


if __name__ == "__main__":
    main()
