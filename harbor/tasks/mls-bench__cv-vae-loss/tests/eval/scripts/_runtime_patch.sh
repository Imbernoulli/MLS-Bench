#!/bin/bash
# Verifier-only runtime patch for mls-bench__cv-vae-loss.
#
# The template fix scopes the /dev/shm FID image dirs per
# (ENV label, SEED, pid) and reclaims them in a finally: — the fixed
# /dev/shm/_eval_{orig,recon} paths are shared across parallel seeds and
# co-located Apptainer instances, which cleared/overwrote each other's
# images mid-eval.
#
# That fix lives in tasks/cv-vae-loss/edits/custom_template.py and is baked into the
# workspace at image build time — images built before the fix still carry the
# racy block. This patch closes the gap at eval time.
#
# Why a copy instead of patching in place: custom_train.py is guarded byte-for-byte
# against tests/meta/pristine by score_task.py `guard`, and guard runs at the
# start of every verifier pass — mutating the agent's file would zero any
# verifier re-run. File creation is explicitly not a guard violation, so the
# patched code is written to _verifier_custom_train.py and the eval scripts run that copy.
#
# Anchoring: the replace matches the exact pre-fix block. If the baked file
# already carries the fix (image rebuilt from the fixed template) or an agent
# edit touched the block (the block is in a protected region, so it is present whenever guard passed), the replace is a no-op and the
# copy is byte-identical to the agent's file, i.e. identical semantics.

_mls_patch_lock=".mls_runtime_patch.lock"
{
if command -v flock >/dev/null 2>&1; then
    flock 9
fi

python3 - <<'PY'
import os
import tempfile
from pathlib import Path

src = Path('custom_train.py')
dst = Path('_verifier_custom_train.py')

OLD = (
    "    # Use RAM-backed tmpfs for image I/O (much faster than disk)\n"
    "    orig_dir = \"/dev/shm/_eval_orig\"\n"
    "    recon_dir = \"/dev/shm/_eval_recon\"\n"
    "\n"
    "    if rank == 0:\n"
    "        for d in [orig_dir, recon_dir]:\n"
    "            if os.path.exists(d):\n"
    "                shutil.rmtree(d)\n"
    "            os.makedirs(d)\n"
    "    if world_size > 1:\n"
    "        dist.barrier()\n"
    "\n"
    "    sample_pairs = []\n"
    "    idx = 0\n"
    "\n"
    "    with torch.no_grad():\n"
    "        for x, _ in test_loader:\n"
    "            x = x.to(device)\n"
    "            with torch.amp.autocast(device_type='cuda'):\n"
    "                recon, _ = model(x)\n"
    "\n"
    "            x_01 = (x * 0.5 + 0.5).float()\n"
    "            recon_01 = recon.clamp(-1, 1).float() * 0.5 + 0.5\n"
    "\n"
    "            psnr_sum += compute_psnr(recon_01, x_01).item() * x.shape[0]\n"
    "            ssim_sum += compute_ssim(recon_01, x_01).item() * x.shape[0]\n"
    "            count += x.shape[0]\n"
    "\n"
    "            if rank == 0:\n"
    "                # Collect first 10 pairs for visual comparison\n"
    "                if len(sample_pairs) < 10:\n"
    "                    n = min(10 - len(sample_pairs), x.shape[0])\n"
    "                    for j in range(n):\n"
    "                        sample_pairs.append((\n"
    "                            (x_01[j] * 255).clamp(0, 255).byte().cpu(),\n"
    "                            (recon_01[j] * 255).clamp(0, 255).byte().cpu(),\n"
    "                        ))\n"
    "\n"
    "                # Save to RAM tmpfs for FID computation\n"
    "                x_uint8 = (x_01 * 255).clamp(0, 255).byte().cpu()\n"
    "                r_uint8 = (recon_01 * 255).clamp(0, 255).byte().cpu()\n"
    "                for j in range(x.shape[0]):\n"
    "                    Image.fromarray(x_uint8[j].permute(1, 2, 0).numpy()).save(\n"
    "                        os.path.join(orig_dir, f'{idx:05d}.png'))\n"
    "                    Image.fromarray(r_uint8[j].permute(1, 2, 0).numpy()).save(\n"
    "                        os.path.join(recon_dir, f'{idx:05d}.png'))\n"
    "                    idx += 1\n"
    "\n"
    "    if world_size > 1:\n"
    "        dist.barrier()\n"
    "\n"
    "    avg_psnr = psnr_sum / max(count, 1)\n"
    "    avg_ssim = ssim_sum / max(count, 1)\n"
    "\n"
    "    rfid = None\n"
    "    if rank == 0:\n"
    "        import cleanfid.features as _feat\n"
    "\n"
    "        cache_dir = \"/data/cleanfid\"\n"
    "        os.makedirs(cache_dir, exist_ok=True)\n"
    "\n"
    "        # Patch cleanfid to load inception from image cache (no network needed)\n"
    "        _orig_build = _feat.build_feature_extractor\n"
    "        def _patched_build(mode, device=device, use_dataparallel=True):\n"
    "            from cleanfid.inception_torchscript import InceptionV3W\n"
    "            m = InceptionV3W(cache_dir, download=False,\n"
    "                             resize_inside=(mode == \"legacy_tensorflow\")).to(device)\n"
    "            m.eval()\n"
    "            if use_dataparallel:\n"
    "                m = torch.nn.DataParallel(m)\n"
    "            return lambda x: m(x)\n"
    "        _feat.build_feature_extractor = _patched_build\n"
    "\n"
    "        rfid = cleanfid.compute_fid(\n"
    "            orig_dir, recon_dir,\n"
    "            device=device, batch_size=64, verbose=False,\n"
    "        )\n"
    "\n"
    "        _feat.build_feature_extractor = _orig_build\n"
    "\n"
    "        # Save 10 sample comparisons\n"
    "        sample_dir = os.path.join(output_dir, 'samples')\n"
    "        os.makedirs(sample_dir, exist_ok=True)\n"
    "        for i, (orig_t, recon_t) in enumerate(sample_pairs):\n"
    "            o = Image.fromarray(orig_t.permute(1, 2, 0).numpy())\n"
    "            r = Image.fromarray(recon_t.permute(1, 2, 0).numpy())\n"
    "            cmp = Image.new('RGB', (o.width * 2 + 4, o.height), (128, 128, 128))\n"
    "            cmp.paste(o, (0, 0))\n"
    "            cmp.paste(r, (o.width + 4, 0))\n"
    "            cmp.save(os.path.join(sample_dir, f'cmp_{i:02d}.png'))\n"
    "\n"
    "        shutil.rmtree(orig_dir, ignore_errors=True)\n"
    "        shutil.rmtree(recon_dir, ignore_errors=True)\n"
    "\n"
    "    if world_size > 1:\n"
    "        dist.barrier()\n"
    "\n"
    "    model.train()\n"
    "    return rfid, avg_psnr, avg_ssim\n"
)

NEW = (
    "    # Use RAM-backed tmpfs for image I/O (much faster than disk).\n"
    "    # /dev/shm is shared beyond this run: parallel seeds in this container,\n"
    "    # and co-located instances under Apptainer (which binds the host\n"
    "    # /dev/shm), all see the same tmpfs \u2014 with fixed paths, concurrent\n"
    "    # evals rmtree/overwrite each other's images mid-eval. Scope the dirs\n"
    "    # per (ENV label, SEED, pid) instead.\n"
    "    def _fs_safe(value):\n"
    "        return \"\".join(c if (c.isalnum() or c in \"._-\") else \"_\" for c in str(value))\n"
    "    _eval_scope = \"{}_s{}_p{}\".format(\n"
    "        _fs_safe(os.environ.get(\"ENV\", \"default\")),\n"
    "        _fs_safe(os.environ.get(\"SEED\", \"0\")),\n"
    "        os.getpid(),\n"
    "    )\n"
    "    orig_dir = f\"/dev/shm/_eval_orig_{_eval_scope}\"\n"
    "    recon_dir = f\"/dev/shm/_eval_recon_{_eval_scope}\"\n"
    "\n"
    "    if rank == 0:\n"
    "        for d in [orig_dir, recon_dir]:\n"
    "            shutil.rmtree(d, ignore_errors=True)\n"
    "            os.makedirs(d)\n"
    "    if world_size > 1:\n"
    "        dist.barrier()\n"
    "\n"
    "    sample_pairs = []\n"
    "    idx = 0\n"
    "\n"
    "    try:\n"
    "        with torch.no_grad():\n"
    "            for x, _ in test_loader:\n"
    "                x = x.to(device)\n"
    "                with torch.amp.autocast(device_type='cuda'):\n"
    "                    recon, _ = model(x)\n"
    "\n"
    "                x_01 = (x * 0.5 + 0.5).float()\n"
    "                recon_01 = recon.clamp(-1, 1).float() * 0.5 + 0.5\n"
    "\n"
    "                psnr_sum += compute_psnr(recon_01, x_01).item() * x.shape[0]\n"
    "                ssim_sum += compute_ssim(recon_01, x_01).item() * x.shape[0]\n"
    "                count += x.shape[0]\n"
    "\n"
    "                if rank == 0:\n"
    "                    # Collect first 10 pairs for visual comparison\n"
    "                    if len(sample_pairs) < 10:\n"
    "                        n = min(10 - len(sample_pairs), x.shape[0])\n"
    "                        for j in range(n):\n"
    "                            sample_pairs.append((\n"
    "                                (x_01[j] * 255).clamp(0, 255).byte().cpu(),\n"
    "                                (recon_01[j] * 255).clamp(0, 255).byte().cpu(),\n"
    "                            ))\n"
    "\n"
    "                    # Save to RAM tmpfs for FID computation\n"
    "                    x_uint8 = (x_01 * 255).clamp(0, 255).byte().cpu()\n"
    "                    r_uint8 = (recon_01 * 255).clamp(0, 255).byte().cpu()\n"
    "                    for j in range(x.shape[0]):\n"
    "                        Image.fromarray(x_uint8[j].permute(1, 2, 0).numpy()).save(\n"
    "                            os.path.join(orig_dir, f'{idx:05d}.png'))\n"
    "                        Image.fromarray(r_uint8[j].permute(1, 2, 0).numpy()).save(\n"
    "                            os.path.join(recon_dir, f'{idx:05d}.png'))\n"
    "                        idx += 1\n"
    "\n"
    "        if world_size > 1:\n"
    "            dist.barrier()\n"
    "\n"
    "        avg_psnr = psnr_sum / max(count, 1)\n"
    "        avg_ssim = ssim_sum / max(count, 1)\n"
    "\n"
    "        rfid = None\n"
    "        if rank == 0:\n"
    "            import cleanfid.features as _feat\n"
    "\n"
    "            cache_dir = \"/data/cleanfid\"\n"
    "            os.makedirs(cache_dir, exist_ok=True)\n"
    "\n"
    "            # Patch cleanfid to load inception from image cache (no network needed)\n"
    "            _orig_build = _feat.build_feature_extractor\n"
    "            def _patched_build(mode, device=device, use_dataparallel=True):\n"
    "                from cleanfid.inception_torchscript import InceptionV3W\n"
    "                m = InceptionV3W(cache_dir, download=False,\n"
    "                                 resize_inside=(mode == \"legacy_tensorflow\")).to(device)\n"
    "                m.eval()\n"
    "                if use_dataparallel:\n"
    "                    m = torch.nn.DataParallel(m)\n"
    "                return lambda x: m(x)\n"
    "            _feat.build_feature_extractor = _patched_build\n"
    "\n"
    "            rfid = cleanfid.compute_fid(\n"
    "                orig_dir, recon_dir,\n"
    "                device=device, batch_size=64, verbose=False,\n"
    "            )\n"
    "\n"
    "            _feat.build_feature_extractor = _orig_build\n"
    "\n"
    "            # Save 10 sample comparisons\n"
    "            sample_dir = os.path.join(output_dir, 'samples')\n"
    "            os.makedirs(sample_dir, exist_ok=True)\n"
    "            for i, (orig_t, recon_t) in enumerate(sample_pairs):\n"
    "                o = Image.fromarray(orig_t.permute(1, 2, 0).numpy())\n"
    "                r = Image.fromarray(recon_t.permute(1, 2, 0).numpy())\n"
    "                cmp = Image.new('RGB', (o.width * 2 + 4, o.height), (128, 128, 128))\n"
    "                cmp.paste(o, (0, 0))\n"
    "                cmp.paste(r, (o.width + 4, 0))\n"
    "                cmp.save(os.path.join(sample_dir, f'cmp_{i:02d}.png'))\n"
    "\n"
    "    finally:\n"
    "        # Self-scoped cleanup: always reclaim this run's /dev/shm dirs,\n"
    "        # even when the eval raises, so tmpfs never leaks across runs\n"
    "        # (only rank 0 ever created them).\n"
    "        if rank == 0:\n"
    "            shutil.rmtree(orig_dir, ignore_errors=True)\n"
    "            shutil.rmtree(recon_dir, ignore_errors=True)\n"
    "\n"
    "    if world_size > 1:\n"
    "        dist.barrier()\n"
    "\n"
    "    model.train()\n"
    "    return rfid, avg_psnr, avg_ssim\n"
)

text = src.read_text()
if OLD in text:
    text = text.replace(OLD, NEW, 1)

fd, tmp = tempfile.mkstemp(dir=str(dst.parent), prefix="." + dst.name + ".", suffix=".tmp")
try:
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, dst)
finally:
    if os.path.exists(tmp):
        os.unlink(tmp)
PY
} 9>"${_mls_patch_lock}"
