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
    import torch.nn as nn
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

    backbone = common.build_backbone(num_ids, loss="triplet").to(device)
    feat_dim = backbone.feature_dim

    # Agent-controlled embedding head.
    head = common.load_surface(args.solution, "build_head")(feat_dim)
    if not isinstance(head, nn.Module):
        print("REID_SURFACE_FALLBACK name=head reason=not_module", flush=True)
        raise TypeError("build_head must return nn.Module")
    head = head.to(device)
    out_dim = common.head_out_dim(head, feat_dim, device)
    print(f"REID_HEAD {getattr(head, 'name', type(head).__name__)} out_dim={out_dim}",
          flush=True)

    model = common.HeadModel(backbone, head, out_dim, num_ids).to(device)

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
