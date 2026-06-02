"""Prepare data for InverseBench package.

Downloads pretrained diffusion models and test datasets for inverse problem benchmarks.
Run via: mlsbench data InverseBench

Creates:
  <data_root>/inversebench/checkpoints/{inv-scatter-5m.pt, blackhole-50k.pt, ffhq256.pt}
  <data_root>/inversebench/inv-scatter-test/
  <data_root>/inversebench/blackhole/{test/, measure/}
  <data_root>/inversebench/cache/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


CHECKPOINTS = {
    "inv-scatter-5m.pt": "https://github.com/devzhk/InverseBench/releases/download/diffusion-prior/inv-scatter-5m.pt",
    "blackhole-50k.pt": "https://github.com/devzhk/InverseBench/releases/download/diffusion-prior/blackhole-50k.pt",
    # Diffusion prior for the FFHQ256 inpainting eval. Required: the
    # ai4sci-inverse-diffusion-algo task scores an `inpainting` label whose
    # script loads problem.prior=checkpoints/ffhq256.pt. Converted from the DPS
    # repo; published on the same InverseBench diffusion-prior release.
    # NOTE: the inpainting TEST DATA (/data/ffhq256, ImageFolder id_list 0-9) is
    # NOT in the public InverseBench distribution (the OSN bucket has no ffhq256)
    # — it must be supplied separately; see the ffhq256 placeholder handling below.
    "ffhq256.pt": "https://github.com/devzhk/InverseBench/releases/download/diffusion-prior/ffhq256.pt",
}

TEST_DATA = {
    "inv-scatter-test": "https://sdsc.osn.xsede.org/ini230004-bucket01/zg89b-mpv16/inv-scatter-test.zip",
    "blackhole": "https://sdsc.osn.xsede.org/ini230004-bucket01/zg89b-mpv16/blackhole.zip",
}

# Torch Hub cached weights used by runtime evaluators (e.g. piq.LPIPS for ffhq256
# inpainting). Compute nodes have NO network access, so these must be pre-fetched
# at build time. We write them under <data_root>/inversebench/cache/torch/hub/
# which the container mounts at /workspace/InverseBench/cache/torch, and set
# TORCH_HOME to that path via pkg_configs/InverseBench/config.json env.
TORCH_HUB_CHECKPOINTS = {
    "lpips_weights.pt": "https://github.com/photosynthesis-team/photosynthesis.metrics/releases/download/v0.4.0/lpips_weights.pt",
    # piq.LPIPS loads a VGG16 backbone via torchvision, which fetches this file
    # through torch.hub. Pre-download it so offline compute nodes can construct
    # the LPIPS metric without network access.
    "vgg16-397923af.pth": "https://download.pytorch.org/models/vgg16-397923af.pth",
}


def download(url: str, dest: str) -> None:
    print(f"  Downloading {url} -> {dest}")
    subprocess.run(["wget", "-q", "-O", dest, url], check=True)


# inv-scatter forward-operator SVD cache directory name. Must match
# InverseScatter.compute_svd()'s path 'cache/inv-scatter_numT_<T>_numR_<R>'
# (numTrans=20, numRec=360 from configs/problem/inv-scatter.yaml).
_INV_SCATTER_SVD_DIRNAME = "inv-scatter_numT_20_numR_360"
_INV_SCATTER_SVD_FILES = ("U.pt", "S.pt", "Vt.pt", "matrix.pt", "matrix_inv.pt")


def precompute_inv_scatter_svd(project_root: Path, root: Path) -> None:
    """Precompute the deterministic inv-scatter SVD artifact ONCE at build time.

    Root cause this fixes: InverseScatter.compute_svd() builds a (14400, 16384)
    float64 matrix and runs torch.svd + torch.linalg.pinv on it. That float64
    factorization is recomputed on EVERY run because the cache was never
    populated, and on GPUs with crippled FP64 throughput (e.g. H20) it scales
    pathologically and can consume the whole timeout without emitting a metric.

    The artifact depends only on problem=inv-scatter (not on algorithm/seed), so
    we compute it once here and write it to the shared cache directory that the
    container mounts at /workspace/InverseBench/cache. The operator's existing
    cache-load branch then picks it up verbatim and skips the SVD entirely.

    Runs the precompute INSIDE the just-built InverseBench container so it uses
    the exact numpy<2 / scipy / torch versions and a GPU, falling back to host
    Python only if no container runtime is available. Idempotent and best-effort:
    a failure here only means runtime falls back to the (slow) recompute path, so
    we warn rather than abort data preparation.
    """
    cache_dir = root / "cache" / _INV_SCATTER_SVD_DIRNAME
    if all((cache_dir / f).exists() for f in _INV_SCATTER_SVD_FILES):
        print(f"  [SKIP] inv-scatter SVD cache already complete at {cache_dir}")
        return

    cache_root = root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    ext_pkg = project_root / "vendor" / "external_packages" / "InverseBench"
    script = (
        project_root
        / "vendor"
        / "data_scripts"
        / "InverseBench"
        / "precompute_inv_scatter_svd.py"
    )
    if not ext_pkg.exists() or not script.exists():
        print(
            f"  [WARN] cannot precompute inv-scatter SVD: missing "
            f"{ext_pkg if not ext_pkg.exists() else script}; "
            f"runtime will recompute (slow)."
        )
        return

    # Inside the container: package mounted read-only at /workspace/InverseBench,
    # writable cache at /workspace/InverseBench/cache, helper script at /precompute.
    # cwd=/workspace/InverseBench so the relative 'cache/...' path lines up with
    # the runtime layout.
    docker_image = os.environ.get("INV_SCATTER_DOCKER_IMAGE", "mlsbench/inversebench")
    sif = project_root / "vendor" / "images" / "InverseBench.sif"

    inner = (
        "cd /workspace/InverseBench && "
        "PYTHONPATH=/workspace/InverseBench "
        "INV_SCATTER_CACHE_ROOT=/workspace/InverseBench/cache "
        "python3 /precompute/precompute_inv_scatter_svd.py"
    )

    def _have_docker_image() -> bool:
        try:
            r = subprocess.run(
                ["docker", "image", "inspect", docker_image],
                capture_output=True, text=True,
            )
            return r.returncode == 0
        except FileNotFoundError:
            return False

    cmd = None
    if _have_docker_image():
        gpu_flag = ["--gpus", "all"] if _docker_gpu_available() else []
        # NOTE: the package is mounted read-WRITE (not :ro) because the cache is
        # mounted as a subdirectory inside it; docker cannot create the
        # /workspace/InverseBench/cache mountpoint under a read-only parent
        # mount. The container is --rm and the host package dir is not modified
        # by the precompute (it only writes into the cache bind).
        cmd = [
            "docker", "run", "--rm", *gpu_flag,
            "-v", f"{ext_pkg.resolve()}:/workspace/InverseBench",
            "-v", f"{cache_root.resolve()}:/workspace/InverseBench/cache",
            "-v", f"{script.parent.resolve()}:/precompute:ro",
            "--entrypoint", "bash", docker_image, "-c", inner,
        ]
        runtime = f"docker ({docker_image})"
    elif sif.exists() and _which("apptainer"):
        cmd = [
            "apptainer", "exec", "--nv",
            "--bind", f"{ext_pkg.resolve()}:/workspace/InverseBench",
            "--bind", f"{cache_root.resolve()}:/workspace/InverseBench/cache",
            "--bind", f"{script.parent.resolve()}:/precompute",
            "--pwd", "/workspace/InverseBench",
            str(sif), "bash", "-c", inner,
        ]
        runtime = f"apptainer ({sif.name})"
    else:
        # Host fallback: requires torch + scipy + numpy<2 on the host. The
        # external package must be importable; add it to PYTHONPATH.
        runtime = "host python (no container runtime found)"
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{ext_pkg.resolve()}:" + env.get("PYTHONPATH", "")
        env["INV_SCATTER_CACHE_ROOT"] = str(cache_root.resolve())
        cmd = [sys.executable, str(script)]
        print(f"  Precomputing inv-scatter SVD via {runtime} ...")
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            print(
                "  [WARN] inv-scatter SVD precompute failed on host; runtime "
                "will recompute (slow). Provide a built InverseBench image to "
                "precompute reliably."
            )
        return

    print(f"  Precomputing inv-scatter SVD via {runtime} ...")
    r = subprocess.run(cmd)
    if r.returncode != 0 or not all((cache_dir / f).exists() for f in _INV_SCATTER_SVD_FILES):
        print(
            "  [WARN] inv-scatter SVD precompute did not complete; runtime will "
            "recompute (slow)."
        )
    else:
        print(f"  inv-scatter SVD cache ready at {cache_dir}")


def _which(name: str) -> bool:
    from shutil import which
    return which(name) is not None


def _docker_gpu_available() -> bool:
    """True if docker can see a GPU (so the FP64 SVD runs fast, not on CPU)."""
    try:
        r = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all",
             "--entrypoint", "nvidia-smi", "mlsbench/inversebench", "-L"],
            capture_output=True, text=True, timeout=60,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    # The pkg_config host_path uses {project_root}/vendor/data/inversebench,
    # NOT {data_root}/inversebench. When data_root differs from project_root
    # (e.g. apptainer-verify config), populating data_root makes the readiness
    # check fail because it inspects the {project_root}-templated host_path.
    # Always write to the project-root location so docker (where the two are
    # equal) and apptainer-verify both end up with the right host path.
    project_root = Path(__file__).resolve().parents[3]
    root = project_root / "vendor" / "data" / "inversebench"
    root.mkdir(parents=True, exist_ok=True)

    # Checkpoints
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for name, url in CHECKPOINTS.items():
        dest = ckpt_dir / name
        if dest.exists():
            print(f"  [SKIP] {name} already exists")
        else:
            download(url, str(dest))

    # Test data
    for name, url in TEST_DATA.items():
        dest_dir = root / name
        if dest_dir.exists() and any(dest_dir.iterdir()):
            print(f"  [SKIP] {name} already exists")
        else:
            zip_path = f"/tmp/{name}.zip"
            download(url, zip_path)
            subprocess.run(["unzip", "-oq", zip_path, "-d", str(root)], check=True)
            os.remove(zip_path)
            print(f"  Extracted {name}")

    # Cache dir
    (root / "cache").mkdir(parents=True, exist_ok=True)

    # The pkg_config data_bind references several extra directories
    # (fwi-test, ffhq256, pip_packages_v2, devito_packages). For the
    # inv-scatter smoke verify only `inv-scatter-test` and the diffusion
    # checkpoints are *required*, but apptainer's --bind hard-fails when
    # ANY listed source directory does not exist. Mirror readable copies
    # from the shared location when available, and otherwise create empty
    # placeholder directories so the bind succeeds (the workload won't
    # touch them).
    _shared_root = Path(
        "/scratch/gpfs/CHIJ/st3812/projects/MLS-Bench/vendor/data/inversebench"
    )
    for sub in ("fwi-test", "ffhq256", "pip_packages_v2", "devito_packages"):
        dst = root / sub
        if dst.exists() or dst.is_symlink():
            continue
        src = _shared_root / sub
        try:
            if src.exists() and os.access(str(src), os.R_OK):
                # Symlink the readable shared mirror so subsequent runs see
                # the contents; falls back to empty dir if that fails.
                dst.symlink_to(src)
                print(f"  symlinked {sub} -> {src}")
                continue
        except (PermissionError, OSError):
            pass
        dst.mkdir(parents=True, exist_ok=True)
        print(f"  created empty placeholder {sub}/ (shared copy unreadable)")

    # Torch Hub checkpoints (for runtime evaluators with no network access).
    # TORCH_HOME=/workspace/InverseBench/cache/torch in the container, so torch.hub
    # resolves weights at /workspace/InverseBench/cache/torch/hub/checkpoints/.
    hub_dir = root / "cache" / "torch" / "hub" / "checkpoints"
    hub_dir.mkdir(parents=True, exist_ok=True)
    for name, url in TORCH_HUB_CHECKPOINTS.items():
        dest = hub_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [SKIP] torch hub {name} already exists")
        else:
            download(url, str(dest))

    # Precompute the deterministic inv-scatter forward-operator SVD once so the
    # operator loads it instead of recomputing the float64 SVD/pinv on every
    # run (see precompute_inv_scatter_svd docstring for root cause).
    precompute_inv_scatter_svd(project_root, root)

    # Verify
    checks = [
        ckpt_dir / "inv-scatter-5m.pt",
        ckpt_dir / "blackhole-50k.pt",
        ckpt_dir / "ffhq256.pt",
        root / "inv-scatter-test",
        root / "blackhole" / "test",
        root / "blackhole" / "measure",
        hub_dir / "lpips_weights.pt",
        hub_dir / "vgg16-397923af.pth",
    ]
    missing = [str(p) for p in checks if not p.exists()]
    if missing:
        print(f"ERROR: Missing: {missing}", file=sys.stderr)
        sys.exit(1)

    # The FFHQ256 inpainting eval data (root/ffhq256, ImageFolder id_list 0-9) is
    # NOT publicly distributed by InverseBench and is handled as an optional
    # placeholder above. Warn loudly if it is empty so an image built without it
    # doesn't silently ship a broken `inpainting` eval (FileNotFoundError / no
    # images at /data/ffhq256).
    ffhq_dir = root / "ffhq256"
    if not any(ffhq_dir.iterdir()) if ffhq_dir.exists() else True:
        print(
            "WARNING: ffhq256/ test images are absent — the `inpainting` eval of "
            "ai4sci-inverse-diffusion-algo will fail. Supply the FFHQ256 images "
            "(ImageFolder, ids 0-9) before building the Harbor base image.",
            file=sys.stderr,
        )
    print("All InverseBench data verified.")


if __name__ == "__main__":
    main()
