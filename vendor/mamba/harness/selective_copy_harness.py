#!/usr/bin/env python3
"""Paper-scale selective-copying harness for mamba-selective-scan.

Trains the two-layer D=64 selective SSM from Mamba Appendix E.1 and reports
token-level accuracy over the copied positions. The agent supplies a literal SSM
parameterization mode; trusted code determines how (dt, B, C) are produced from
the post-conv features, deciding whether the recurrence is SELECTIVE
(input-dependent / S6) or TIME-INVARIANT (input-independent / S4-LTI). Only an
input-dependent parameterization can gate the randomly-scattered data tokens into
state, so this evaluates the parameterization design under the declared protocol.

Emits protocol-bound ``POOL_LOADED`` and ``MAMBA_TRAIN_COMPLETE`` proofs followed
by one ``MAMBA_COPY_METRICS`` record. All three bind the full run configuration
and parameter count.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    format_metric_line,
    get_device,
    load_surface_hook_and_choice,
    train_and_eval,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--label", default="paper_e1")
    ap.add_argument("--L", type=int, default=4096)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--A", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--d_state", type=int, default=16)
    ap.add_argument("--n_layer", type=int, default=2)
    ap.add_argument("--steps", type=int, default=400000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--protocol", default="mamba_selective_copy_paper_e1_v1")
    ap.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = get_device()
    parameterize, surface_choice = load_surface_hook_and_choice(
        Path(args.solution), "parameterize"
    )

    out = train_and_eval(
        parameterize, init_fn=None,
        steps=args.steps, L=args.L, M=args.M, A=args.A, d_model=args.d_model,
        d_state=args.d_state, n_layer=args.n_layer, batch=args.batch, lr=args.lr,
        seed=args.seed, device=device, proof_task="mamba-selective-scan",
        proof_label=args.label, surface_choice=surface_choice,
        protocol=args.protocol,
        optimizer_name=args.optimizer, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, eval_batches=args.eval_batches)

    print(f"COPY_DEBUG acc={out['acc']:.4f} final_loss={out['final_loss']:.4f} "
          f"wall_s={out['wall_s']:.1f}", flush=True)
    print(
        format_metric_line(
            "MAMBA_COPY_METRICS", protocol=args.protocol,
            task="mamba-selective-scan",
            label=args.label, surface_choice=surface_choice,
            L=args.L, M=args.M, A=args.A,
            d_model=args.d_model, d_state=args.d_state, n_layer=args.n_layer,
            steps=args.steps, batch=args.batch, lr=args.lr,
            optimizer=args.optimizer, weight_decay=args.weight_decay,
            grad_clip=args.grad_clip, eval_batches=args.eval_batches,
            seed=args.seed, result=out,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
