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

    # Agent-controlled trainable configuration of the backbone.
    configure_result = common.load_surface(args.solution, "configure_trainable")(model)
    if configure_result is not None:
        print("REID_SURFACE_FALLBACK name=configure_trainable reason=non_none", flush=True)
        raise TypeError("configure_trainable must return None")
    # The id-classifier (and any fc head) is ALWAYS trained by the harness, so a
    # "frozen backbone" still trains a classifier on the fixed features.
    for attr in ("classifier", "fc"):
        mod = getattr(model, attr, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"REID_FINETUNE trainable_params={n_train} total_params={n_total} "
          f"frac={n_train/max(1,n_total):.3f}", flush=True)

    xent = CrossEntropyLoss(num_classes=num_ids, use_gpu=(device == "cuda"))
    triplet = TripletLoss(margin=0.3)
    # optimiser over only the parameters that still require grad
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=3e-4, weight_decay=5e-4)
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
