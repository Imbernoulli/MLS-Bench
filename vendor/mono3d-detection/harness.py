#!/usr/bin/env python3
"""Monocular 3D object detection harness (fixed pipeline) for the mono3d-* tasks.

Recover a full 3D bounding box (metric depth, metric dimensions, yaw) for an object seen in
a SINGLE image. The data is REAL KITTI 3D Object Detection (see common.py for the download/
staging story): real photos, real LiDAR-derived 3D boxes (Car/Pedestrian/Cyclist), and a
single fixed representative pinhole intrinsic K (the dataset-mean calibration; see common.py
docstring for the measured-variance justification). The model is handed the amodal 2D box
(as a geometry feature vector) + a REAL appearance crop from the image. Because the boxes are
real annotations (not rendered), 3D IoU / AP3D are computed against real, occasionally noisy,
LiDAR-derived geometry -- this is the standard KITTI eval convention.

ONE harness hosts MANY research questions over the SAME fixed dataset and the SAME fixed
appearance/geometry encoder (`common.RegionEncoder`). Everything — data, splits, encoder,
optimizer, epochs, batch, seed, scoring — is FIXED. The only degree of freedom is the ONE
design surface selected by --task; every OTHER component is pinned to its fixed STRONG default
so the surface under study is isolated. If the agent surface errors, the harness falls back to
the fixed strong default for that surface. The editable surfaces (RQs):

  depth        DEPTH PARAMETERIZATION — metric Z is destroyed by projection. WEAK = regress Z
               directly; STRONG = projective geometry Z=f*H/h2d (Deep3DBox height-guided depth)
               + a learned residual. `build_depth_head(emb_dim) -> (head, decode(raw,ctx)->Z)`.

  orient       ORIENTATION ENCODING — yaw lives on a circle. WEAK = scalar regression (unstable
               at +-pi); MEDIUM = (cos,sin)+atan2; STRONG = Deep3DBox MultiBin.
               `build_orient_head(emb_dim) -> (head, decode(raw)->yaw, loss(raw,yaw_gt))`.

  dims         DIMENSION PRIOR / ANCHORS — metric (l,h,w). WEAK = regress dims directly (no
               prior); STRONG = residual on the log class-mean anchor (statistical shape prior,
               Deep3DBox/MonoDLE). `build_dims_head(emb_dim, log_mean, cls_dims) -> (head,
               decode(raw,ctx)->dims[B,3])`; ctx has "cls_onehot" (soft class posterior).

  yaw_frame    ALLOCENTRIC vs EGOCENTRIC YAW — appearance encodes the LOCAL (observation-ray)
               orientation, not the global yaw; the two differ by the ray angle. WEAK = predict
               global (egocentric) yaw directly; STRONG = predict allocentric (ray-relative) yaw
               and add the ray angle back (Deep3DBox/M3D-RPN allocentric convention).
               `build_yawframe_head(emb_dim) -> (head, decode(raw,ctx)->yaw)`; ctx has "ray"
               (the observation-ray azimuth per object, atan2(x, z)).

  loss3d       3D-BOX REGRESSION LOSS — how depth/dims/yaw errors are combined. WEAK = a single
               coupled L2 on the raw concatenated 7-DoF vector (all terms same scale, depth in
               metres dominates); STRONG = decoupled/disentangled per-component loss (log-depth,
               log-dims, angular yaw), the MonoDLE/disentangling-losses recipe.
               `build_loss3d(emb_dim) -> loss(pred, gt) -> scalar`; pred/gt dicts of tensors.

  uncertainty  MULTI-TASK UNCERTAINTY WEIGHTING — how the depth/dims/orient losses are balanced.
               WEAK = fixed EQUAL weights; STRONG = learned homoscedastic (Kendall) uncertainty
               weights sigma_k with the log-sigma regularizer. `build_task_weighting() ->
               (params: nn.Module|None, weight(losses:dict)->scalar)`.

  feature      FEATURE REPRESENTATION / FUSION — which cues the head sees. WEAK = appearance-only
               (drop the geometry feature vector -> no direct 2D-box-height cue); STRONG = fuse
               appearance + geometry features. `build_feature_fusion(feat_dim, crop_hw) ->
               (module: nn.Module, forward(feat, crop)->emb[B,EMB_DIM])`.

  backbone     HEAD CAPACITY (depth/width) — WEAK = a shallow/narrow head; STRONG = a deeper,
               wider head with residual connections. `build_backbone(emb_dim) -> nn.Module`
               mapping emb -> emb (a refinement block before the fixed task heads).

  normalization DEPTH TARGET / OUTPUT NORMALIZATION — how the geometry-depth residual is
               parameterized. WEAK = raw additive residual in metres (badly scaled across the
               6-40m range); STRONG = multiplicative log-space residual (scale-invariant).
               `build_depth_norm() -> apply(geom_Z, raw) -> Z`.

  lr           HEAD LEARNING-RATE MULTIPLIER — WEAK = a tiny LR (head barely trains, residual
               ~0, ~raw geometry); STRONG = a well-tuned LR. `build_lr_mult() -> float` (the
               multiplier on the base LR for the depth head's parameters only).

  depth_cue    WHICH 2D CUE DRIVES THE GEOMETRY DEPTH — WEAK = box WIDTH (Z=f*W/w2d; width is
               yaw-confounded); STRONG = box HEIGHT (Z=f*H/h2d; yaw-invariant). Reuses
               `build_depth_head`; ctx exposes h2d/pred_H and w2d/pred_W.

  height_source WHERE THE METRIC HEIGHT H COMES FROM — WEAK = a GLOBAL CONSTANT H0=1.5; STRONG =
               the PER-OBJECT predicted height (class prior). Reuses `build_depth_head`.

  projection   3D-CENTER BACK-PROJECTION — WEAK = ON-AXIS assumption (x=y=0); STRONG = full
               pinhole inverse projection x=(u-cx)Z/f, y=(v-cy)Z/f. `build_backproject() ->
               backproject(loc_z, box2d, cx, cy, focal) -> (x, y)`.

Metric line (one per run):
    MONO3D_METRICS task=<T> setting=<L> ap25=<A25> ap50=<A50> miou=<I> \
        depth_err=<Dz> yaw_err=<Yd> dim_err=<De> steps=<n>
where ap25 = AP3D at 3D-IoU>=0.25 over the TEST split (HIGHER better, the primary score),
ap50 the same at 0.5, miou the mean 3D IoU (HIGHER), depth_err the median abs center-depth
error in metres (LOWER), yaw_err the mean abs yaw error in degrees (LOWER), dim_err the mean
abs dimension error in metres (LOWER). A degenerate mean/constant-box predictor scores AP3D~0.
The orient / yaw_frame / loss3d surfaces exist and run, but did not yield a monotone weak->strong
order at this scale (verified empirically) and are NOT shipped as tasks.

Settings: every shipped task is scored over KITTI's OWN OFFICIAL easy/moderate/hard difficulty
tiers (bbox height / occlusion / truncation thresholds; see common.kitti_difficulty), applied
EXCLUSIVELY (each object belongs to exactly one tier, not the cumulative official-eval-server
convention) -- this replaces the old synthetic near/mid/far depth-distance regimes.
"""
from __future__ import annotations

import argparse
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common

# ---------------------------------------------------------------------------- #
# Fixed training hyper-parameters (NOT exposed to the agent).
# ---------------------------------------------------------------------------- #
DEFAULT_STEPS = 1200          # short schedule (minute-scale on 1 GPU)
BATCH_SIZE = 128
LR = 2e-3
N_ORIENT_BINS = 4             # fixed MultiBin bin count
DIM_HEAD_HID = 64

# NOTE: the old synthetic harness had two independent settings axes -- distance regimes for
# most tasks, yaw/orientation bands for the orient/yaw_frame surfaces (neither of the latter is
# shipped). On real KITTI data there is a single settings axis, the dataset's own official
# easy/moderate/hard difficulty tiers (common.DIFFICULTY_SETTINGS), used by every task.


# ---------------------------------------------------------------------------- #
# Fixed shared model: RegionEncoder (frozen design) + dims head + task heads.
# By default EVERY head is the fixed STRONG default; the ONE surface under study
# (--task) is replaced by the agent's design. The dims head default is the
# log-residual regressor (a statistical shape prior).
# ---------------------------------------------------------------------------- #
def default_dims_head(emb_dim, log_mean, cls_dims):
    """STRONG default dims surface: residual on log(class-mean dims) -> positive metric (l,h,w)."""
    net = nn.Sequential(nn.Linear(emb_dim, DIM_HEAD_HID), nn.ReLU(),
                        nn.Linear(DIM_HEAD_HID, 3))
    lm = log_mean.detach().clone()

    class _Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.register_buffer("log_mean", lm)

        def forward(self, emb):
            return self.net(emb)

    head = _Head()

    def decode(raw, ctx):
        return torch.exp(head.log_mean.unsqueeze(0) + 0.3 * raw)

    return head, decode


class DimsHead(nn.Module):
    """Back-compat fixed dims head wrapper (log-residual on class mean)."""

    def __init__(self, emb_dim, log_mean):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim, DIM_HEAD_HID), nn.ReLU(),
                                 nn.Linear(DIM_HEAD_HID, 3))
        self.register_buffer("log_mean", log_mean)

    def forward(self, emb):
        return torch.exp(self.log_mean.unsqueeze(0) + 0.3 * self.net(emb))


# -- fixed STRONG defaults (used for every task NOT under study, or on fallback) -- #
def default_depth_head(emb_dim):
    """STRONG default depth surface: geometry-from-height Z = f*H / h2d with a small
    learned multiplicative residual (Deep3DBox / height-guided depth)."""
    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def decode(raw, ctx):
        geom = ctx["focal"] * ctx["pred_H"].reshape(-1) / ctx["h2d"].reshape(-1).clamp(min=1.0)
        return geom * torch.exp(0.1 * raw[:, 0].clamp(-3, 3))

    return head, decode


def default_orient_head(emb_dim):
    """STRONG default orient surface: Deep3DBox MultiBin (classify bin + (cos,sin) residual)."""
    n_bins = N_ORIENT_BINS
    head = nn.Sequential(nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, n_bins * 3))
    centers = torch.tensor([(-math.pi + 2 * math.pi * (i + 0.5) / n_bins) for i in range(n_bins)])

    def decode(raw):
        B = raw.shape[0]
        logit = raw[:, :n_bins]
        res = raw[:, n_bins:].reshape(B, n_bins, 2)
        b = torch.argmax(logit, dim=1)
        c = centers.to(raw.device)[b]
        r = res[torch.arange(B, device=raw.device), b]
        return c + torch.atan2(r[:, 1], r[:, 0])

    def loss(raw, yaw_gt):
        B = raw.shape[0]
        cen = centers.to(raw.device)
        logit = raw[:, :n_bins]
        res = raw[:, n_bins:].reshape(B, n_bins, 2)
        diff = torch.atan2(torch.sin(yaw_gt.unsqueeze(1) - cen.unsqueeze(0)),
                           torch.cos(yaw_gt.unsqueeze(1) - cen.unsqueeze(0)))
        tgt = torch.argmin(diff.abs(), dim=1)
        ce = F.cross_entropy(logit, tgt)
        delta = diff[torch.arange(B, device=raw.device), tgt]
        r = res[torch.arange(B, device=raw.device), tgt]
        rl = ((r[:, 0] - torch.cos(delta)) ** 2 + (r[:, 1] - torch.sin(delta)) ** 2).mean()
        return ce + rl

    return head, decode, loss


def default_yawframe_head(emb_dim):
    """STRONG default yaw-frame surface: predict ALLOCENTRIC (ray-relative) yaw with a MultiBin
    head (supervised on the ray-relative target), then add the observation-ray azimuth back to
    get the global (egocentric) yaw. The allocentric angle is what appearance actually
    determines; adding the ray angle recovers the global pose (Deep3DBox / M3D-RPN)."""
    head, mb_decode, mb_loss = default_orient_head(emb_dim)

    def decode(raw, ctx):
        alloc = mb_decode(raw)
        return alloc + ctx["ray"].reshape(-1)

    def loss(raw, yaw_gt, ctx):
        # supervise the head on the ALLOCENTRIC target (global yaw minus the ray azimuth)
        return mb_loss(raw, yaw_gt - ctx["ray"].reshape(-1))

    return head, decode, loss


def default_task_weighting():
    """STRONG default: learned homoscedastic (Kendall) uncertainty weighting."""
    log_sigma = nn.Parameter(torch.zeros(3))

    class _W(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_sigma = log_sigma

    mod = _W()

    def weight(losses):
        ls = mod.log_sigma
        keys = ["depth", "orient", "dims"]
        total = 0.0
        for i, k in enumerate(keys):
            total = total + torch.exp(-ls[i]) * losses[k] + ls[i]
        return total

    return mod, weight


def default_feature_fusion(feat_dim, crop_hw):
    """STRONG default: the fixed RegionEncoder fusing appearance crop + geometry features."""
    enc = common.RegionEncoder(feat_dim, crop_hw)

    def forward(feat, crop):
        return enc(feat, crop)

    return enc, forward


def default_backbone(emb_dim):
    """Default backbone: IDENTITY (no refinement). The shared encoder embedding is passed
    straight to the task heads, so every task NOT studying the backbone is unaffected. The
    `backbone` RQ swaps in a deeper/wider residual refinement block (STRONG) vs a shallow one."""
    class _Id(nn.Module):
        def forward(self, x):
            return x

    return _Id()


def default_depth_norm():
    """STRONG default: multiplicative log-space residual on the geometry depth (scale-invariant)."""
    def apply(geom_Z, raw):
        return geom_Z * torch.exp(0.1 * raw[:, 0].clamp(-3, 3))

    return apply


def default_lr_mult():
    return 1.0


def default_loss3d(emb_dim):
    """STRONG default: DECOUPLED per-component loss (log-depth, log-dims, angular yaw)."""
    def loss(pred, gt):
        ld = F.smooth_l1_loss(torch.log(pred["Z"].clamp(min=0.5)), torch.log(gt["Z"]))
        ldim = F.smooth_l1_loss(torch.log(pred["dims"].clamp(min=0.05)), torch.log(gt["dims"]))
        dyaw = torch.atan2(torch.sin(pred["yaw"] - gt["yaw"]), torch.cos(pred["yaw"] - gt["yaw"]))
        lyaw = (dyaw ** 2).mean()
        return ld + 0.5 * ldim + 0.5 * lyaw

    return loss


# ---------------------------------------------------------------------------- #
# Assemble the full model given the (possibly agent-supplied) surfaces.
# ---------------------------------------------------------------------------- #
class Mono3DModel(nn.Module):
    def __init__(self, encoder, backbone, dims_head, depth_head, orient_head):
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.dims_head = dims_head
        self.depth_head = depth_head
        self.orient_head = orient_head

    def forward(self, feat, crop):
        emb = self.encoder(feat, crop)
        emb = self.backbone(emb)
        dims_raw = self.dims_head(emb)
        depth_raw = self.depth_head(emb)
        orient_raw = self.orient_head(emb)
        return emb, dims_raw, depth_raw, orient_raw


def _ctx_for(splits, name, dev, dims_pred_h):
    """Build the decode context for split `name`: focal, 2D-box pixel height h2d, principal
    point, and the predicted metric height (from the dims head — geometry depth needs H)."""
    box = splits[f"box2d_{name}"]
    h2d = (box[:, 3] - box[:, 1])
    w2d = (box[:, 2] - box[:, 0])
    loc = splits[f"loc_{name}"]
    ray = torch.atan2(loc[:, 0], loc[:, 2].clamp(min=0.5))     # observation-ray azimuth
    return {
        "focal": torch.tensor(splits["focal"], device=dev),
        "cx": torch.tensor(splits["cx"], device=dev),
        "cy": torch.tensor(splits["cy"], device=dev),
        "h2d": h2d,
        "w2d": w2d,
        "box2d": box,
        "pred_H": dims_pred_h,          # metric height column of the predicted dims
        "ray": ray,
    }


def _load(sol, symbol):
    return common.load_surface(sol, symbol)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["depth", "orient", "dims", "yaw_frame", "loss3d",
                             "uncertainty", "feature", "backbone", "normalization", "lr",
                             "depth_cue", "height_source", "projection"])
    ap.add_argument("--solution", required=True)
    ap.add_argument("--label", default="default")
    ap.add_argument("--setting", default=None,
                    help="Evaluate only the TEST objects in this setting (a distance regime, or "
                         "a yaw regime for the orientation tasks). Training is always on the full "
                         "fixed train split; only scoring is sliced.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = ap.parse_args()
    if args.setting is None:
        args.setting = args.label

    common.set_seeds(args.seed)
    t0 = time.time()
    dev = common.device()
    print(f"DATA_LOADING device={dev} torch={torch.__version__}", flush=True)

    splits = common.load_splits()
    feat_dim = splits["feat_dim"]
    crop_hw = splits["crop_hw"]
    cls_dims = torch.tensor(common._CLASS_DIMS, dtype=torch.float32, device=dev)
    log_mean = torch.log(cls_dims.mean(0))
    print(f"DATA_LOADED n_train={splits['feat_train'].shape[0]} "
          f"n_val={splits['feat_val'].shape[0]} n_test={splits['feat_test'].shape[0]} "
          f"feat_dim={feat_dim} focal={splits['focal']} n_bins={N_ORIENT_BINS}", flush=True)

    emb_dim = common.EMB_DIM

    # ---- fixed strong defaults for EVERY surface ---------------------------
    common.set_seeds(args.seed)
    enc_mod, encoder_fwd = default_feature_fusion(feat_dim, crop_hw)
    backbone = default_backbone(emb_dim)
    dims_head, dims_decode = default_dims_head(emb_dim, log_mean, cls_dims)
    depth_head, depth_decode = default_depth_head(emb_dim)
    orient_head, orient_decode, orient_loss = default_orient_head(emb_dim)
    task_w_mod, task_weight = default_task_weighting()
    depth_norm = default_depth_norm()
    lr_mult = default_lr_mult()
    loss3d = default_loss3d(emb_dim)
    backproject_fn = common.backproject_xy      # default: full pinhole (u,v)->(x,y) at depth Z
    # For the depth/orient/yaw_frame decode we distinguish whether decode takes ctx.
    orient_decode_takes_ctx = False
    yawframe_loss = None

    # ---- swap in the agent surface for the ONE task under study -------------
    def _try(symbol, builder):
        try:
            fn = _load(args.solution, symbol)
            out = builder(fn)
            print(f"SURFACE_OK {symbol}", flush=True)
            return out
        except Exception:
            print(f"SURFACE_ERROR {symbol} -> fixed strong default", flush=True)
            traceback.print_exc()
            return None

    if args.task in ("depth", "depth_cue", "height_source"):
        # all three edit the DEPTH head (the geometry decode). depth = which parameterization;
        # depth_cue = which 2D cue (box height vs width) drives the geometry; height_source =
        # where the metric height H comes from (class prior vs constant). Same plumbing.
        r = _try("build_depth_head", lambda fn: fn(emb_dim))
        if r is not None:
            head, decode = r
            assert isinstance(head, nn.Module) and callable(decode)
            depth_head, depth_decode = head, decode
    elif args.task == "orient":
        r = _try("build_orient_head", lambda fn: fn(emb_dim))
        if r is not None:
            head, decode, lf = r
            assert isinstance(head, nn.Module) and callable(decode) and callable(lf)
            orient_head, orient_decode, orient_loss = head, decode, lf
    elif args.task == "yaw_frame":
        r = _try("build_yawframe_head", lambda fn: fn(emb_dim))
        if r is not None:
            head, decode, lf = r
            assert isinstance(head, nn.Module) and callable(decode) and callable(lf)
            # The yaw_frame surface owns BOTH the decode (which frame it reconstructs the global
            # yaw in) AND the loss (which frame it supervises the head in), so the two are
            # consistent. The encoding is the fixed strong MultiBin.
            orient_head = head
            orient_decode = decode
            orient_decode_takes_ctx = True
            yawframe_loss = lf
        else:
            # fixed strong default (allocentric)
            head, decode, lf = default_yawframe_head(emb_dim)
            orient_head = head.to(dev)
            orient_decode = decode
            orient_decode_takes_ctx = True
            yawframe_loss = lf
    elif args.task == "dims":
        r = _try("build_dims_head", lambda fn: fn(emb_dim, log_mean, cls_dims))
        if r is not None:
            head, decode = r
            assert isinstance(head, nn.Module) and callable(decode)
            dims_head, dims_decode = head, decode
    elif args.task == "loss3d":
        r = _try("build_loss3d", lambda fn: fn(emb_dim))
        if r is not None:
            assert callable(r)
            loss3d = r
    elif args.task == "uncertainty":
        r = _try("build_task_weighting", lambda fn: fn())
        if r is not None:
            mod, w = r
            assert callable(w)
            task_w_mod = mod if isinstance(mod, nn.Module) else default_task_weighting()[0]
            task_weight = w
    elif args.task == "feature":
        r = _try("build_feature_fusion", lambda fn: fn(feat_dim, crop_hw))
        if r is not None:
            mod, fwd = r
            assert isinstance(mod, nn.Module) and callable(fwd)
            enc_mod, encoder_fwd = mod, fwd
    elif args.task == "backbone":
        r = _try("build_backbone", lambda fn: fn(emb_dim))
        if r is not None:
            assert isinstance(r, nn.Module)
            backbone = r
    elif args.task == "normalization":
        r = _try("build_depth_norm", lambda fn: fn())
        if r is not None:
            assert callable(r)
            # override the depth decode's residual application while keeping geometry base
            _agent_apply = r

            def _depth_decode_norm(raw, ctx):
                geom = ctx["focal"] * ctx["pred_H"].reshape(-1) / ctx["h2d"].reshape(-1).clamp(min=1.0)
                return _agent_apply(geom, raw)

            depth_decode = _depth_decode_norm
    elif args.task == "lr":
        r = _try("build_lr_mult", lambda fn: fn())
        if r is not None:
            lr_mult = float(r)
    elif args.task == "projection":
        r = _try("build_backproject", lambda fn: fn())
        if r is not None:
            assert callable(r)
            backproject_fn = r

    # ---- wrap the (possibly custom) encoder forward into an nn.Module -------
    class _EncWrap(nn.Module):
        def __init__(self, mod, fwd):
            super().__init__()
            self.mod = mod
            self._fwd = fwd

        def forward(self, feat, crop):
            return self._fwd(feat, crop)

    encoder = _EncWrap(enc_mod, encoder_fwd).to(dev)

    depth_head = depth_head.to(dev)
    orient_head = orient_head.to(dev)
    dims_head = dims_head.to(dev)
    backbone = backbone.to(dev)
    task_w_mod = task_w_mod.to(dev)
    model = Mono3DModel(encoder, backbone, dims_head, depth_head, orient_head).to(dev)

    # collect params: everything trains; for --task lr the DEPTH head (the residual head under
    # study) gets the lr_mult. For every other task lr_mult=1.0 so the split is a numerical no-op.
    agent_params = []
    other_params = []
    agent_head = {"depth": depth_head, "orient": orient_head, "yaw_frame": orient_head,
                  "dims": dims_head, "feature": encoder, "backbone": backbone,
                  "lr": depth_head, "normalization": depth_head}.get(args.task)
    ah_ids = set(id(p) for p in agent_head.parameters()) if isinstance(agent_head, nn.Module) else set()
    for p in list(model.parameters()) + list(task_w_mod.parameters()):
        if not p.requires_grad:
            continue
        (agent_params if id(p) in ah_ids else other_params).append(p)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MODEL_BUILT params={n_params} task={args.task} lr_mult={lr_mult:.4g}", flush=True)

    param_groups = [{"params": other_params, "lr": LR}]
    if agent_params:
        param_groups.append({"params": agent_params, "lr": LR * lr_mult})
    opt = torch.optim.AdamW(param_groups, lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, [g["lr"] for g in param_groups], args.steps + 10,
        pct_start=0.1, cycle_momentum=False, anneal_strategy="cos")

    feat_tr = splits["feat_train"]; crop_tr = splits["crop_train"]
    dims_tr = splits["dims_train"]; yaw_tr = splits["yaw_train"]; loc_tr = splits["loc_train"]
    n = feat_tr.shape[0]
    ctx_tr = _ctx_for(splits, "train", dev, None)

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    model.train()
    step = 0
    while step < args.steps:
        order = torch.randperm(n, generator=g).tolist()
        for i in range(0, n, BATCH_SIZE):
            idx = torch.tensor(order[i:i + BATCH_SIZE], device=dev)
            emb, dims_raw, depth_raw, orient_raw = model(feat_tr[idx], crop_tr[idx])
            opt.zero_grad()

            # dims decode (agent or fixed strong)
            try:
                dims_pred = dims_decode(dims_raw, {"cls_onehot": None})
            except Exception as e:
                if step == 0:
                    print(f"SURFACE_ERROR dims_decode raised: {e}", flush=True)
                dims_pred = torch.exp(log_mean.unsqueeze(0) + 0.3 * dims_raw[:, :3])
            dims_pred = dims_pred.clamp(min=0.05)

            # depth decode
            bctx = {
                "focal": ctx_tr["focal"], "cx": ctx_tr["cx"], "cy": ctx_tr["cy"],
                "h2d": ctx_tr["h2d"][idx], "w2d": ctx_tr["w2d"][idx],
                "box2d": ctx_tr["box2d"][idx],
                "pred_H": dims_pred[:, 1], "pred_W": dims_pred[:, 2],
                "pred_dims": dims_pred, "ray": ctx_tr["ray"][idx],
            }
            try:
                Zp = depth_decode(depth_raw, bctx)
            except Exception as e:
                if step == 0:
                    print(f"SURFACE_ERROR depth_decode raised: {e}", flush=True)
                Zp = default_depth_head(emb_dim)[1](depth_raw[:, :1], bctx)
            Zp = Zp.reshape(-1).clamp(min=0.5)

            # yaw decode/loss
            try:
                if orient_decode_takes_ctx:
                    yaw_pred_tr = orient_decode(orient_raw, bctx)
                else:
                    yaw_pred_tr = orient_decode(orient_raw)
            except Exception:
                yaw_pred_tr = default_orient_head(emb_dim)[1](orient_raw)
            try:
                if yawframe_loss is not None:
                    loss_orient = yawframe_loss(orient_raw, yaw_tr[idx], bctx)
                else:
                    loss_orient = orient_loss(orient_raw, yaw_tr[idx])
            except Exception as e:
                if step == 0:
                    print(f"SURFACE_ERROR orient loss raised: {e}", flush=True)
                loss_orient = default_orient_head(emb_dim)[2](orient_raw, yaw_tr[idx])

            # component losses
            loss_depth = F.smooth_l1_loss(torch.log(Zp), torch.log(loc_tr[idx][:, 2]))
            loss_dim = F.smooth_l1_loss(dims_pred, dims_tr[idx])

            if args.task == "loss3d":
                # the agent's coupled/decoupled 3D loss REPLACES the fixed combination
                pred = {"Z": Zp, "dims": dims_pred, "yaw": yaw_pred_tr.reshape(-1)}
                gt = {"Z": loc_tr[idx][:, 2], "dims": dims_tr[idx], "yaw": yaw_tr[idx]}
                try:
                    loss = loss3d(pred, gt)
                except Exception as e:
                    if step == 0:
                        print(f"SURFACE_ERROR loss3d raised: {e}", flush=True)
                    loss = default_loss3d(emb_dim)(pred, gt)
                loss = loss + 0.1 * loss_orient   # keep the bin classifier supervised
            elif args.task == "uncertainty":
                try:
                    loss = task_weight({"depth": loss_depth, "orient": loss_orient, "dims": loss_dim})
                except Exception as e:
                    if step == 0:
                        print(f"SURFACE_ERROR task_weight raised: {e}", flush=True)
                    loss = loss_depth + 0.5 * loss_orient + 0.5 * loss_dim
            else:
                loss = loss_depth + 0.5 * loss_orient + 0.5 * loss_dim

            if torch.isfinite(loss) and loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            elif step == 0:
                print("SURFACE_ERROR non-finite/non-diff loss — skipping", flush=True)
            sched.step()
            step += 1
            if step % 300 == 0:
                print(f"step={step} loss={float(loss):.4f} "
                      f"(z={float(loss_depth):.3f} o={float(loss_orient):.3f} "
                      f"d={float(loss_dim):.3f})", flush=True)
            if step >= args.steps:
                break

    # ---- evaluation on TEST -------------------------------------------------
    model.eval()
    with torch.no_grad():
        emb, dims_raw, depth_raw, orient_raw = model(splits["feat_test"], splits["crop_test"])
        try:
            dims_pred = dims_decode(dims_raw, {"cls_onehot": None}).clamp(min=0.05)
        except Exception:
            dims_pred = torch.exp(log_mean.unsqueeze(0) + 0.3 * dims_raw[:, :3]).clamp(min=0.05)
        bctx = _ctx_for(splits, "test", dev, dims_pred[:, 1])
        bctx["pred_W"] = dims_pred[:, 2]
        bctx["pred_dims"] = dims_pred
        try:
            Zp = depth_decode(depth_raw, bctx).reshape(-1).clamp(min=0.5)
        except Exception:
            Zp = default_depth_head(emb_dim)[1](depth_raw[:, :1], bctx).reshape(-1).clamp(min=0.5)
        try:
            if orient_decode_takes_ctx:
                yaw_pred = orient_decode(orient_raw, bctx).reshape(-1)
            else:
                yaw_pred = orient_decode(orient_raw).reshape(-1)
        except Exception:
            yaw_pred = default_orient_head(emb_dim)[1](orient_raw).reshape(-1)

    # Recover full center (x,y,z): back-project the 2D-box center at predicted depth Z.
    box_test = splits["box2d_test"]
    try:
        x_pred, y_pred = backproject_fn(Zp, box_test, splits["cx"], splits["cy"], splits["focal"])
    except Exception:
        print("SURFACE_ERROR build_backproject raised -> full pinhole backprojection", flush=True)
        x_pred, y_pred = common.backproject_xy(Zp, box_test, splits["cx"], splits["cy"], splits["focal"])
    pred_loc = torch.stack([x_pred, y_pred, Zp], dim=1).detach().cpu().numpy()
    pred_dims = dims_pred.detach().cpu().numpy()
    pred_yaw = yaw_pred.detach().cpu().numpy()

    gt_dims = splits["dims_test"].detach().cpu().numpy()
    gt_loc = splits["loc_test"].detach().cpu().numpy()
    gt_yaw = splits["yaw_test"].detach().cpu().numpy()
    gt_difficulty = splits["difficulty_test"].detach().cpu().numpy()

    # ---- restrict scoring to this setting's slice of the TEST split ----------
    # Real KITTI has a single settings axis: the dataset's own official easy/moderate/hard
    # difficulty tiers (see common.DIFFICULTY_SETTINGS / common.kitti_difficulty). This
    # replaces the old synthetic near/mid/far distance-regime and yaw-band axes.
    valid = common.DIFFICULTY_SETTINGS
    if args.setting in valid:
        mask = common.setting_mask(args.task, args.setting, gt_difficulty)
    else:
        print(f"SETTING_WARN unknown setting={args.setting!r} for task={args.task} -> full test",
              flush=True)
        mask = np.ones(gt_dims.shape[0], dtype=bool)
    n_sel = int(mask.sum())
    print(f"SETTING_SLICE task={args.task} setting={args.setting} n={n_sel}/{gt_dims.shape[0]}",
          flush=True)
    pred_dims, pred_loc, pred_yaw = pred_dims[mask], pred_loc[mask], pred_yaw[mask]
    gt_dims, gt_loc, gt_yaw = gt_dims[mask], gt_loc[mask], gt_yaw[mask]

    m = common.score_predictions(pred_dims, pred_loc, pred_yaw, gt_dims, gt_loc, gt_yaw)
    dt = time.time() - t0
    print(f"MONO3D_METRICS task={args.task} setting={args.label} "
          f"ap25={m['ap25']:.6f} ap50={m['ap50']:.6f} miou={m['miou']:.6f} "
          f"depth_err={m['med_depth_err']:.6f} yaw_err={m['mean_yaw_err_deg']:.6f} "
          f"dim_err={float(np.mean(np.abs(pred_dims - gt_dims))):.6f} steps={args.steps} "
          f"elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
