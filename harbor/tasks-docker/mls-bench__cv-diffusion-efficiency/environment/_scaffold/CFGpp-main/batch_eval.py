"""Batch evaluation script for cv-diffusion-efficiency task.

Generates 10K images from COCO captions using multi-GPU DDP,
then computes FID against pre-computed COCO val2014 inception stats
and average CLIP score.
"""

import argparse
import os
import torch
import torch.distributed as dist
import clip
import numpy as np
from pathlib import Path
from PIL import Image
from munch import munchify
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pytorch_fid.inception import InceptionV3

from latent_diffusion import get_solver
from latent_sdxl import get_solver as get_solver_sdxl
from utils.log_util import set_seed

NUM_IMAGES = 10000
PROMPT_FILE = "examples/assets/coco_v2.txt"
COCO_STATS = "/data/coco_val2014/inception_stats.npz"


def setup_ddp():
    """Initialize DDP. Returns (rank, world_size). Falls back to single-GPU."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        return rank, world_size
    return 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_prompts(prompt_file, num_images):
    """Load COCO captions from file."""
    with open(prompt_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[:num_images]


def compute_fid(gen_dir, ref_stats_path, device, batch_size=50):
    """Compute FID between generated images and pre-computed reference stats."""
    ref = np.load(ref_stats_path)
    mu_ref, sigma_ref = ref["mu"], ref["sigma"]

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device).eval()

    class ImageDataset(Dataset):
        def __init__(self, img_dir):
            self.files = sorted([
                f for f in img_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            ])
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
            ])

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            img = Image.open(self.files[idx]).convert("RGB")
            return self.transform(img)

    dataset = ImageDataset(gen_dir)
    if len(dataset) < 2:
        raise RuntimeError(
            f"compute_fid: only {len(dataset)} generated image(s) under {gen_dir}; "
            "image generation likely failed. Refusing to compute a degenerate FID."
        )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_acts = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            pred = model(batch)[0].squeeze(-1).squeeze(-1).cpu().numpy()
            all_acts.append(pred)

    all_acts = np.concatenate(all_acts, axis=0)
    mu_gen = np.mean(all_acts, axis=0)
    sigma_gen = np.cov(all_acts, rowvar=False)

    from scipy.linalg import sqrtm
    diff = mu_gen - mu_ref
    covmean, _ = sqrtm(sigma_gen @ sigma_ref, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma_gen + sigma_ref - 2 * covmean))
    return fid


# ── NFE accounting (fixed; do not modify) ────────────────────────────
# The sampler is scored under a fixed budget of --NFE denoiser evaluations per
# image, so the budget is measured here rather than taken on trust. After the
# solver is built, its UNet is swapped for a counting facade that holds the
# network only in a closure: every evaluation the sampler can make, whether
# through predict_noise() or a direct self.unet(...) call, goes through the
# counter, which lives in this file and nowhere the sampler can reach.
#
# One NFE is one UNet forward over at most two latent rows, i.e. one
# denoiser evaluation of the (single) image being generated: the batched
# unconditional+conditional pair of classifier-free guidance counts as 1, and
# so does a single-branch (conditional-only or unconditional-only) call. A
# call over more rows counts ceil(rows / 2), so batching extra latents or
# timesteps into one call spends budget like separate calls would.
def _holds_network(value, net, depth=0):
    """True if `value` is (or wraps, or directly contains) the raw network."""
    if value is net:
        return True
    if getattr(value, "__self__", None) is net:  # bound method, e.g. unet.forward
        return True
    if isinstance(value, torch.nn.Module):
        return any(m is net for m in value.modules())
    if depth == 0 and isinstance(value, (list, tuple, set, frozenset)):
        return any(_holds_network(v, net, 1) for v in value)
    if depth == 0 and isinstance(value, dict):
        return any(_holds_network(v, net, 1) for v in value.values())
    return False


def install_nfe_counter(solver):
    """Replace solver.unet with a counting facade; return the call counter."""
    net = solver.unet
    if not isinstance(net, torch.nn.Module):
        raise RuntimeError("NFE accounting: solver.unet is not the diffusion UNet "
                           f"(got {type(net).__name__}); the solver must use the "
                           "UNet loaded by the fixed StableDiffusion/SDXL base class.")
    for name, value in vars(solver).items():
        if name != "unet" and _holds_network(value, net):
            raise RuntimeError(f"NFE accounting: solver attribute {name!r} holds a second "
                               "handle on the UNet; every denoiser evaluation must go "
                               "through self.unet / self.predict_noise().")
    nfe = [0]
    config = net.config
    add_embedding = getattr(net, "add_embedding", None)

    class CountedUNet:
        """self.unet as the sampler sees it: callable like the UNet."""
        __slots__ = ()

        def __call__(self, sample, *args, **kwargs):
            rows = int(sample.shape[0]) if sample.dim() > 3 else 1
            nfe[0] += max(1, (rows + 1) // 2)
            return net(sample, *args, **kwargs)

        @property
        def config(self):
            return config

        @property
        def add_embedding(self):  # read by SDXL._get_add_time_ids
            return add_embedding

    solver.unet = CountedUNet()
    return nfe


def main():
    rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{rank}")

    parser = argparse.ArgumentParser(description="Batch evaluation for diffusion efficiency")
    parser.add_argument("--model", choices=["sd15", "sd20", "sdxl"], required=True)
    parser.add_argument("--method", type=str, default="ddim_cfg++")
    parser.add_argument("--cfg_guidance", type=float, default=0.6)
    parser.add_argument("--NFE", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--num_images", type=int, default=NUM_IMAGES)
    args = parser.parse_args()

    # Clear stale images before generating: compute_fid globs EVERY .png in
    # workdir, so leftover shards from a prior run (a crashed/partial run, or a
    # run with a different GPU count -> different shard indices) would silently
    # contaminate the FID set and skew the score. Rank 0 wipes; others wait.
    if rank == 0:
        import shutil
        if args.workdir.exists():
            shutil.rmtree(args.workdir)
        args.workdir.mkdir(parents=True, exist_ok=True)
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
    args.workdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed + rank)
    solver_config = munchify({"num_sampling": args.NFE})

    # Load prompts — all ranks load full list, each takes its shard
    all_prompts = load_prompts(PROMPT_FILE, args.num_images)
    my_indices = list(range(rank, len(all_prompts), world_size))
    my_prompts = [all_prompts[i] for i in my_indices]
    if rank == 0:
        print(f"[{args.model}] Total {len(all_prompts)} prompts, "
              f"{world_size} GPUs, ~{len(my_prompts)} per GPU", flush=True)

    # Load diffusion model on this rank's device
    if rank == 0:
        print(f"[{args.model}] Loading model...", flush=True)
    if args.model == "sdxl":
        solver = get_solver_sdxl(
            args.method, solver_config=solver_config, device=device)
    else:
        model_keys = {
            "sd15": "runwayml/stable-diffusion-v1-5",
            "sd20": "Manojb/stable-diffusion-2-base",
        }
        solver = get_solver(
            args.method, solver_config=solver_config,
            model_key=model_keys[args.model], device=device)
    if rank == 0:
        print(f"[{args.model}] Model loaded on {world_size} GPUs.", flush=True)
    nfe = install_nfe_counter(solver)
    nfe_max = 0

    # Load CLIP model
    clip_model, preprocess = clip.load("ViT-B/32", device=device,
                                       download_root="/opt/model_weights")

    # Generate images and compute CLIP scores
    clip_scores = []
    for count, (global_idx, prompt) in enumerate(zip(my_indices, my_prompts)):
        nfe[0] = 0
        if args.model == "sdxl":
            result = solver.sample(
                prompt1=["", prompt],
                prompt2=["", prompt],
                cfg_guidance=args.cfg_guidance,
                target_size=(1024, 1024))
        else:
            result = solver.sample(
                prompt=["", prompt],
                cfg_guidance=args.cfg_guidance)
        if nfe[0] > args.NFE:
            raise RuntimeError(
                f"NFE_BUDGET_EXCEEDED: the sampler spent {nfe[0]} denoiser evaluations "
                f"on image {global_idx} (budget NFE={args.NFE}; the batched cond+uncond "
                f"CFG pair counts as 1). The run is rejected.")
        if nfe[0] == 0:
            raise RuntimeError(
                f"NFE accounting: the sampler made no denoiser evaluation for image "
                f"{global_idx}; images must be generated by the diffusion UNet.")
        nfe_max = max(nfe_max, nfe[0])

        img_path = args.workdir / f"{str(global_idx).zfill(5)}.png"
        save_image(result, img_path, normalize=True)

        # CLIP score
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        text = clip.tokenize([prompt], truncate=True).to(device)
        with torch.no_grad():
            img_feat = clip_model.encode_image(image)
            txt_feat = clip_model.encode_text(text)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            cs = (img_feat @ txt_feat.T).item()
        clip_scores.append(cs)

        if rank == 0 and ((count + 1) % 100 == 0 or count == 0):
            print(f"[{args.model}] rank0: {count+1}/{len(my_prompts)} generated, "
                  f"running_clip={sum(clip_scores)/len(clip_scores):.4f}", flush=True)

    # Gather CLIP scores from all ranks
    local_clip_sum = torch.tensor(sum(clip_scores), device=device)
    local_clip_cnt = torch.tensor(len(clip_scores), device=device)
    local_nfe_max = torch.tensor(nfe_max, device=device)
    if world_size > 1:
        dist.all_reduce(local_clip_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_clip_cnt, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_nfe_max, op=dist.ReduceOp.MAX)
        dist.barrier()

    # Only rank 0 computes FID and prints final metrics
    if rank == 0:
        del solver
        torch.cuda.empty_cache()

        print(f"[{args.model}] Computing FID...", flush=True)
        fid = compute_fid(args.workdir, COCO_STATS, device)

        avg_clip = (local_clip_sum / local_clip_cnt).item()
        print(f"GENERATION_METRICS model={args.model} method={args.method} "
              f"cfg_guidance={args.cfg_guidance} NFE={args.NFE} "
              f"nfe_used={int(local_nfe_max.item())} seed={args.seed} "
              f"fid={fid:.4f} clip_score={avg_clip:.4f}",
              flush=True)

    cleanup_ddp()


if __name__ == "__main__":
    main()
