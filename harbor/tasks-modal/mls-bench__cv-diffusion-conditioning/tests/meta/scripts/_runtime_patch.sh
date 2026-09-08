#!/bin/bash
# Verifier-only runtime patch for cv-diffusion-conditioning.
#
# The task image bakes /workspace/diffusers-main/custom_train.py from
# edits/custom_template.py at render time, so template fixes do NOT reach
# already-built images. This rewrites the compute_fid scratch-dir handling in
# the agent's workspace copy at eval time. The rewritten region is OUTSIDE the
# declared editable range (config.json edit = lines 195-227,
# prepare_conditioning + ClassConditioner only), so on any guard-passing
# submission it is byte-identical to the pristine template.
#
# Bug being fixed: per-rank FID samples went to the shared
# tempfile.gettempdir()/fid_gen_<pid> and rank 0 merged EVERY /tmp/fid_gen_*
# dir it could glob. A crashed earlier label (labels run sequentially in ONE
# container and share /tmp; native Apptainer /tmp is the host's) leaves up to
# 8 dirs x 50k PNGs that later labels' FID silently ingests. Fix: scope
# everything under this run's $OUTPUT_DIR/_fid_tmp, wipe the base at the start
# of every compute_fid call, glob only under the base, clean up at the end.
#
# Detection AND application are EOF-anchored against the file's protected
# tail: all three rewrite sites sit below the editable range, in the
# protected region that runs to EOF, so the patch compares the file's tail
# against the two byte-exact known states of that region (pre-fix render vs
# fixed template) and swaps stale tail -> fixed tail in a single write.
# Content the agent puts inside the editable range can never influence the
# decision (a whole-file '_fid_tmp' marker search could previously be forged
# from the editable region, silently keeping the contaminating /tmp glob);
# a tail matching neither known state (unexpected variant / tampered
# workspace) is left untouched — all-or-nothing, as before.

_fid_patch_lock="${MLSBENCH_FID_PATCH_LOCK:-.mlsbench_fid_patch.lock}"
{
if command -v flock >/dev/null 2>&1; then
    flock 9
fi

python3 - <<'PY'
from pathlib import Path

# Eval scripts run with cwd = the package root (/workspace/diffusers-main).
path = Path("custom_train.py")
if not path.exists():
    raise SystemExit(0)
text = path.read_text()

OLD_GEN = (
    '    gen_dir = os.path.join(tempfile.gettempdir(), f"fid_gen_{os.getpid()}")\n'
    "    os.makedirs(gen_dir, exist_ok=True)\n"
)
NEW_GEN = (
    "    # All FID scratch dirs live under this run's OUTPUT_DIR so leftovers from\n"
    "    # a crashed or concurrent eval sharing /tmp can never leak into this FID.\n"
    "    fid_base = os.path.join(os.environ.get('OUTPUT_DIR', '/tmp/output'), '_fid_tmp')\n"
    "    if rank == 0:\n"
    "        if os.path.exists(fid_base):\n"
    "            shutil.rmtree(fid_base)\n"
    "        os.makedirs(fid_base)\n"
    "    if world_size > 1:\n"
    "        dist.barrier()\n"
    '    gen_dir = os.path.join(fid_base, f"fid_gen_{os.getpid()}")\n'
    "    os.makedirs(gen_dir, exist_ok=True)\n"
)

OLD_MERGE = (
    "    # Rank 0 gathers all images and computes FID\n"
    "    score = 0.0\n"
    "    if rank == 0:\n"
    "        # Merge images from all ranks into one dir\n"
    '        merged_dir = os.path.join(tempfile.gettempdir(), "fid_merged")\n'
    "        if os.path.exists(merged_dir):\n"
    "            shutil.rmtree(merged_dir)\n"
    "        os.makedirs(merged_dir)\n"
    "\n"
    "        # Copy from all per-rank dirs\n"
    "        for f in sorted(os.listdir(gen_dir)):\n"
    "            shutil.copy2(os.path.join(gen_dir, f), os.path.join(merged_dir, f))\n"
    "\n"
    "        if world_size > 1:\n"
    "            # Other ranks wrote to /tmp on the same node\n"
    "            import glob\n"
    '            for other_dir in glob.glob(os.path.join(tempfile.gettempdir(), "fid_gen_*")):\n'
    "                if other_dir == gen_dir:\n"
    "                    continue\n"
    "                for f in os.listdir(other_dir):\n"
    "                    shutil.copy2(os.path.join(other_dir, f), os.path.join(merged_dir, f))\n"
)
NEW_MERGE = (
    "    # Rank 0 gathers all images and computes FID\n"
    "    score = 0.0\n"
    "    if rank == 0:\n"
    "        # Merge images from all ranks into one dir\n"
    '        merged_dir = os.path.join(fid_base, "fid_merged")\n'
    "        if os.path.exists(merged_dir):\n"
    "            shutil.rmtree(merged_dir)\n"
    "        os.makedirs(merged_dir)\n"
    "\n"
    "        # Copy from all per-rank dirs\n"
    "        for f in sorted(os.listdir(gen_dir)):\n"
    "            shutil.copy2(os.path.join(gen_dir, f), os.path.join(merged_dir, f))\n"
    "\n"
    "        if world_size > 1:\n"
    "            # Other ranks wrote their own subdirs under this run's fid_base\n"
    "            import glob\n"
    '            for other_dir in glob.glob(os.path.join(fid_base, "fid_gen_*")):\n'
    "                if other_dir == gen_dir:\n"
    "                    continue\n"
    "                for f in os.listdir(other_dir):\n"
    "                    shutil.copy2(os.path.join(other_dir, f), os.path.join(merged_dir, f))\n"
)

OLD_CLEAN = (
    "    # Clean up per-rank dir\n"
    "    shutil.rmtree(gen_dir, ignore_errors=True)\n"
    "\n"
    "    if world_size > 1:\n"
    "        dist.barrier()\n"
    "\n"
    "    model.train()\n"
    "    return score\n"
)
NEW_CLEAN = (
    "    # Clean up per-rank dir\n"
    "    shutil.rmtree(gen_dir, ignore_errors=True)\n"
    "\n"
    "    if world_size > 1:\n"
    "        dist.barrier()\n"
    "\n"
    "    # Remove this run's FID scratch base (merged_dir + per-rank dirs are gone)\n"
    "    if rank == 0:\n"
    "        shutil.rmtree(fid_base, ignore_errors=True)\n"
    "\n"
    "    model.train()\n"
    "    return score\n"
)

BLOCKS = [(OLD_GEN, NEW_GEN), (OLD_MERGE, NEW_MERGE), (OLD_CLEAN, NEW_CLEAN)]

# tests/meta/config.json declares ONE editable range for this file: lines
# 195-227, prepare_conditioning + ClassConditioner (hardcoded here — keep in
# sync with config.json). All three rewrite sites sit BELOW that range, in
# the protected region that runs to EOF, so on any workspace whose protected
# bytes are intact the file's tail IS that protected region. STALE_TAIL is
# the byte-exact protected tail (pristine lines 228..EOF) of the pre-fix
# render; FIXED_TAIL — derived from it by the three anchored replacements —
# is the same tail in the fixed template. Comparing the file's tail against
# these two constants means nothing the agent writes inside the editable
# range (which always lies ABOVE the tail) can suppress, trigger, or
# misplace the patch.
STALE_TAIL = '''

# ============================================================================
# Sampling — uses diffusers DDIMScheduler
# ============================================================================

@torch.no_grad()
def sample_images(model, num_samples, device, num_classes=10, num_steps=1000,
                  sample_steps=50, img_size=32, channels=3):
    """Generate class-conditional images via DDIM sampling (diffusers)."""
    model.eval()
    scheduler = DDIMScheduler(
        num_train_timesteps=num_steps,
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=True,
        set_alpha_to_one=False,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(sample_steps)

    x = torch.randn(num_samples, channels, img_size, img_size, device=device)
    class_labels = torch.randint(0, num_classes, (num_samples,), device=device)

    for t in scheduler.timesteps:
        t_batch = t.expand(num_samples).to(device)
        with torch.amp.autocast(device_type='cuda'):
            noise_pred = model(x, t_batch, class_labels)
        x = scheduler.step(noise_pred, t, x).prev_sample

    model.train()
    return x.clamp(-1, 1)


# ============================================================================
# FID computation (using clean-fid)
# ============================================================================

def compute_fid(model, device, num_samples=2048, num_classes=10, num_steps=1000,
                sample_steps=50, img_size=32, batch_size=128,
                rank=0, world_size=1):
    """Compute FID against CIFAR-10 train set using clean-fid.

    All ranks sample in parallel; rank 0 computes FID on the merged results.
    """
    import shutil
    import tempfile

    model.eval()

    # Each rank samples its share
    my_samples = num_samples // world_size
    if rank < (num_samples % world_size):
        my_samples += 1
    start_idx = rank * (num_samples // world_size) + min(rank, num_samples % world_size)

    gen_dir = os.path.join(tempfile.gettempdir(), f"fid_gen_{os.getpid()}")
    os.makedirs(gen_dir, exist_ok=True)

    generated = 0
    idx = start_idx
    while generated < my_samples:
        bs = min(batch_size, my_samples - generated)
        imgs = sample_images(model, bs, device, num_classes, num_steps,
                             sample_steps, img_size)
        imgs_uint8 = ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu()
        for j in range(bs):
            img_np = imgs_uint8[j].permute(1, 2, 0).numpy()
            Image.fromarray(img_np).save(os.path.join(gen_dir, f'{idx:05d}.png'))
            idx += 1
        generated += bs
        if rank == 0:
            print(f"  Sampling: {generated}/{my_samples}", flush=True)

    # Sync all ranks before FID computation
    if world_size > 1:
        dist.barrier()

    # Rank 0 gathers all images and computes FID
    score = 0.0
    if rank == 0:
        # Merge images from all ranks into one dir
        merged_dir = os.path.join(tempfile.gettempdir(), "fid_merged")
        if os.path.exists(merged_dir):
            shutil.rmtree(merged_dir)
        os.makedirs(merged_dir)

        # Copy from all per-rank dirs
        for f in sorted(os.listdir(gen_dir)):
            shutil.copy2(os.path.join(gen_dir, f), os.path.join(merged_dir, f))

        if world_size > 1:
            # Other ranks wrote to /tmp on the same node
            import glob
            for other_dir in glob.glob(os.path.join(tempfile.gettempdir(), "fid_gen_*")):
                if other_dir == gen_dir:
                    continue
                for f in os.listdir(other_dir):
                    shutil.copy2(os.path.join(other_dir, f), os.path.join(merged_dir, f))

        from cleanfid import fid as cleanfid
        import cleanfid.features as _feat

        cache_dir = "/data/cleanfid"
        os.makedirs(cache_dir, exist_ok=True)

        inception_path = os.path.join(cache_dir, "inception-2015-12-05.pt")
        stats_path = os.path.join(cache_dir, "cifar10_clean_train_32.npz")
        missing = [p for p in (inception_path, stats_path) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing clean-fid cache files prepared by `mlsbench data diffusers-main`: "
                + ", ".join(missing)
            )

        _orig_build = _feat.build_feature_extractor
        def _patched_build(mode, device=device, use_dataparallel=True):
            from cleanfid.inception_torchscript import InceptionV3W
            m = InceptionV3W(cache_dir, download=False,
                             resize_inside=(mode == "legacy_tensorflow")).to(device)
            m.eval()
            if use_dataparallel:
                m = torch.nn.DataParallel(m)
            return lambda x: m(x)
        _feat.build_feature_extractor = _patched_build

        _orig_ref = _feat.get_reference_statistics
        def _patched_ref(name, res, mode="clean", model_name="inception_v3",
                         seed=0, split="train", metric="FID"):
            fpath = os.path.join(cache_dir, f"{name}_{mode}_{split}_{res}.npz".lower())
            stats = np.load(fpath)
            return stats["mu"], stats["sigma"]
        _feat.get_reference_statistics = _patched_ref
        import cleanfid.fid as _fid_mod
        _fid_mod.get_reference_statistics = _patched_ref
        _orig_fid_build = _fid_mod.build_feature_extractor
        _fid_mod.build_feature_extractor = _patched_build

        print(f"  Computing FID on {len(os.listdir(merged_dir))} images...", flush=True)
        score = cleanfid.compute_fid(
            merged_dir, dataset_name="cifar10", dataset_res=32,
            dataset_split="train", device=device, batch_size=batch_size, verbose=False,
        )

        shutil.rmtree(merged_dir)
        _feat.build_feature_extractor = _orig_build
        _feat.get_reference_statistics = _orig_ref
        _fid_mod.get_reference_statistics = _orig_ref
        _fid_mod.build_feature_extractor = _orig_fid_build

    # Clean up per-rank dir
    shutil.rmtree(gen_dir, ignore_errors=True)

    if world_size > 1:
        dist.barrier()

    model.train()
    return score


def save_sample_images(model, device, output_dir, step, num_images=16,
                       num_classes=10, num_steps=1000, sample_steps=50, tag=""):
    """Save a grid of sample images for visual inspection."""
    imgs = sample_images(model, num_images, device, num_classes, num_steps, sample_steps)
    imgs = ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu()

    # Make a grid: 4x4
    nrow = int(math.sqrt(num_images))
    grid_h = nrow * 32
    grid_w = nrow * 32
    grid = Image.new('RGB', (grid_w, grid_h))
    for i in range(num_images):
        img_np = imgs[i].permute(1, 2, 0).numpy()
        img = Image.fromarray(img_np)
        row, col = divmod(i, nrow)
        grid.paste(img, (col * 32, row * 32))

    suffix = f"_{tag}" if tag else ""
    path = os.path.join(output_dir, f'samples_step{step}{suffix}.png')
    grid.save(path)
    print(f"Saved sample images to {path}", flush=True)


# ============================================================================
# Training Script
# ============================================================================

if __name__ == '__main__':
    seed = int(os.environ.get('SEED', 42))
    data_dir = os.environ.get('DATA_DIR', '/data/cifar10')
    output_dir = os.environ.get('OUTPUT_DIR', '/tmp/output')
    max_steps = int(os.environ.get('MAX_STEPS', 10000))
    eval_interval = int(os.environ.get('EVAL_INTERVAL', 10000))
    batch_size = int(os.environ.get('BATCH_SIZE', 128))
    lr = float(os.environ.get('LR', 2e-4))
    num_fid_samples = int(os.environ.get('NUM_FID_SAMPLES', 2048))
    num_classes = int(os.environ.get('NUM_CLASSES', 10))
    diffusion_steps = int(os.environ.get('DIFFUSION_STEPS', 1000))
    sample_steps = int(os.environ.get('SAMPLE_STEPS', 50))
    ema_rate = float(os.environ.get('EMA_RATE', 0.9999))

    # ── DDP setup ──────────────────────────────────────────────────────────
    use_ddp = 'RANK' in os.environ
    if use_ddp:
        import datetime as _dt
        dist.init_process_group(backend='nccl', timeout=_dt.timedelta(hours=20))
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        is_main = (rank == 0)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank = 0
        world_size = 1
        is_main = True

    torch.manual_seed(seed + rank)
    os.makedirs(output_dir, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = datasets.CIFAR10(data_dir, train=True, transform=transform, download=False)
    if use_ddp:
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=4, pin_memory=True, drop_last=True,
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )
    data_iter = iter(loader)

    # ── Noise scheduler (diffusers) ────────────────────────────────────────
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=diffusion_steps,
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        prediction_type="epsilon",
        clip_sample=True,
        variance_type="fixed_large",
    )

    # ── Model ───────────────────────────────────────────────────────────────
    net = ConditionalUNet(num_classes=num_classes).to(device)

    # Parameter budget check (1.05x largest baseline: Cross-Attention)
    # Build a temporary reference model with the cross-attn baseline to compute budget.
    def _budget_prepare_conditioning(time_emb, class_emb):
        return time_emb
    class _BudgetClassConditioner(nn.Module):
        def __init__(self, channels, cond_dim):
            super().__init__()
            self.cross_attn = CrossAttentionLayer(channels, cond_dim, num_heads=4)
        def forward(self, h, class_emb):
            return self.cross_attn(h, class_emb)
    _orig_prepare = globals()['prepare_conditioning']
    _orig_conditioner = globals()['ClassConditioner']
    globals()['prepare_conditioning'] = _budget_prepare_conditioning
    globals()['ClassConditioner'] = _BudgetClassConditioner
    _ref_net = ConditionalUNet(num_classes=num_classes)
    _max_budget = int(sum(p.numel() for p in _ref_net.parameters()) * 1.05)
    del _ref_net
    globals()['prepare_conditioning'] = _orig_prepare
    globals()['ClassConditioner'] = _orig_conditioner
    _total_params = sum(p.numel() for p in net.parameters())
    print(f"Parameter count: {_total_params:,} / {_max_budget:,}", flush=True)

    # EMA model for evaluation (on main process only)
    ema_net = copy.deepcopy(net)
    ema_net.requires_grad_(False)

    if use_ddp:
        net = DDP(net, device_ids=[local_rank])
    net_raw = net.module if use_ddp else net

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.95, 0.999), weight_decay=1e-6, eps=1e-8)
    scaler = torch.amp.GradScaler()

    num_params = sum(p.numel() for p in net_raw.parameters())
    if is_main:
        print(f"Model parameters: {num_params/1e6:.1f}M | GPUs: {world_size}", flush=True)

    # ── Training loop ────────────────────────────────────────────────────────
    best_fid = float('inf')
    t0 = time.time()
    epoch = 0

    for step in range(1, max_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            epoch += 1
            if use_ddp:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        B = x.shape[0]

        # Sample random timesteps and add noise (using diffusers scheduler)
        t = torch.randint(0, diffusion_steps, (B,), device=device).long()
        noise = torch.randn_like(x)
        x_t = noise_scheduler.add_noise(x, noise, t)

        # Predict noise
        with torch.amp.autocast(device_type='cuda'):
            pred_noise = net(x_t, t, y)
            loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Update EMA (use raw model params)
        with torch.no_grad():
            for p_ema, p in zip(ema_net.parameters(), net_raw.parameters()):
                p_ema.mul_(ema_rate).add_(p, alpha=1 - ema_rate)

        if is_main and step % 200 == 0:
            dt_elapsed = time.time() - t0
            print(f"step {step}/{max_steps} | loss {loss.item():.4f} | {dt_elapsed:.1f}s",
                  flush=True)
            t0 = time.time()

        if step % eval_interval == 0 or step == max_steps:
            if is_main:
                print(f"Eval at step {step}...", flush=True)
                save_sample_images(net_raw, device, output_dir, step,
                                   num_classes=num_classes, num_steps=diffusion_steps,
                                   sample_steps=sample_steps, tag="net")
                save_sample_images(ema_net, device, output_dir, step,
                                   num_classes=num_classes, num_steps=diffusion_steps,
                                   sample_steps=sample_steps, tag="ema")
            eval_model = ema_net if step >= 20000 else net_raw
            fid = compute_fid(eval_model, device, num_samples=num_fid_samples,
                              num_classes=num_classes, num_steps=diffusion_steps,
                              sample_steps=sample_steps,
                              rank=rank, world_size=world_size)
            if is_main:
                print(f"TRAIN_METRICS: step={step}, loss={loss.item():.4f}, fid={fid:.2f}",
                      flush=True)
                if fid < best_fid:
                    best_fid = fid

    # ── Save & final eval ────────────────────────────────────────────────────
    if is_main:
        print(f"Saving checkpoint to {output_dir}/checkpoint.pth", flush=True)
        torch.save({
            'step': max_steps,
            'model_state_dict': net_raw.state_dict(),
            'ema_model_state_dict': ema_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_fid': best_fid,
        }, os.path.join(output_dir, 'checkpoint.pth'))

        save_sample_images(net_raw, device, output_dir, max_steps,
                           num_classes=num_classes, num_steps=diffusion_steps,
                           sample_steps=sample_steps, tag="net_final")
        save_sample_images(ema_net, device, output_dir, max_steps,
                           num_classes=num_classes, num_steps=diffusion_steps,
                           sample_steps=sample_steps, tag="ema_final")

    eval_model = ema_net if max_steps >= 20000 else net_raw
    fid = compute_fid(eval_model, device, num_samples=num_fid_samples,
                      num_classes=num_classes, num_steps=diffusion_steps,
                      sample_steps=sample_steps,
                      rank=rank, world_size=world_size)
    if is_main:
        print(f"TEST_METRICS: fid={fid:.2f}, best_fid={best_fid:.2f}", flush=True)

    if use_ddp:
        dist.destroy_process_group()
'''

if any(STALE_TAIL.count(old) != 1 for old, _ in BLOCKS):
    raise SystemExit(0)  # embedded-constant sanity check — never expected
FIXED_TAIL = STALE_TAIL
for old, new in BLOCKS:
    FIXED_TAIL = FIXED_TAIL.replace(old, new, 1)

if text.endswith(FIXED_TAIL):
    raise SystemExit(0)  # fix already present in the protected tail — no-op
if not text.endswith(STALE_TAIL):
    # The protected tail matches neither the known stale render nor the fixed
    # template (unexpected variant / tampered workspace) — leave the file
    # untouched rather than risk a partial rewrite. All-or-nothing.
    raise SystemExit(0)
# Swap the whole protected tail in one write; bytes above it (including the
# agent's editable region) are byte-for-byte untouched.
path.write_text(text[:len(text) - len(STALE_TAIL)] + FIXED_TAIL)
print("[runtime-patch] custom_train.py: FID scratch dirs scoped under "
      "$OUTPUT_DIR/_fid_tmp (was shared /tmp fid_gen_* glob)", flush=True)
PY
} 9>"${_fid_patch_lock}"
