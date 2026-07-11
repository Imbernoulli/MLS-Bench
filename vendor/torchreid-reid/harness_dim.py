#!/usr/bin/env python3
"""Full Market-1501 harness with a fixed projection head and editable dimension."""
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
    started_at = time.perf_counter()

    import torch
    import torch.nn as nn
    from torchreid.data.sampler import RandomIdentitySampler
    from torchreid.losses import CrossEntropyLoss, TripletLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    common.emit_fullscale_protocol(
        task_id=args.task_id, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, num_instances=args.num_instances,
    )

    items, pid2label, num_ids = common.load_train_items()
    relabeled = [(path, pid2label[pid], cam) for path, pid, cam in items]
    print(f"REID_DATA train_ids={num_ids} train_imgs={len(items)}", flush=True)
    dataset = common.ReidImageDataset(items, common.base_train_transform(), pid2label)
    sampler = RandomIdentitySampler(relabeled, args.batch_size, args.num_instances)
    loader = common.fullscale_loader(dataset, sampler, batch_size=args.batch_size)

    backbone = common.build_backbone(num_ids, loss="triplet").to(device)
    feat_dim = backbone.feature_dim
    embedding_dim = common.load_surface(
        args.solution, "build_embedding_dim"
    )(feat_dim)
    if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int):
        print("REID_SURFACE_FALLBACK name=embedding_dim reason=not_int", flush=True)
        raise TypeError("build_embedding_dim must return an integer")
    if not 16 <= embedding_dim <= feat_dim:
        print("REID_SURFACE_FALLBACK name=embedding_dim reason=out_of_range", flush=True)
        raise ValueError(f"embedding dimension must be in [16, {feat_dim}]")

    class FixedProjection(nn.Module):
        def __init__(self, dim_in: int, dim_out: int):
            super().__init__()
            self.fc = nn.Linear(dim_in, dim_out)
            self.bn = nn.BatchNorm1d(dim_out)
            self.bn.bias.requires_grad_(False)

        def forward(self, features):
            return self.bn(self.fc(features))

    head = FixedProjection(feat_dim, embedding_dim).to(device)
    print(f"REID_EMBEDDING_DIM value={embedding_dim}", flush=True)
    model = common.HeadModel(backbone, head, embedding_dim, num_ids).to(device)

    xent = CrossEntropyLoss(num_classes=num_ids, use_gpu=True)
    triplet = TripletLoss(margin=0.3)
    optimizer = common.build_optimizer(model)
    scheduler = common.default_epoch_scheduler(optimizer)

    def loss_step(images, labels):
        logits, features = model(images)
        loss = xent(logits, labels) + triplet(features, labels)
        return logits, features, loss

    total_steps, train_samples = common.run_fullscale_training(
        model, loader, optimizer, loss_step, epochs=args.epochs,
        epoch_scheduler=scheduler,
    )
    common.finish_fullscale_evaluation(
        model, started_at, task_id=args.task_id, total_steps=total_steps,
        train_samples=train_samples,
    )


if __name__ == "__main__":
    main()
