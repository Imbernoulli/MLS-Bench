#!/usr/bin/env python3
"""Prepare data for ProteinWorkshop (ai4bio-protein-structure-repr).

Drives the co-located preprocessing script
``preprocess_protein_workshop.py`` inside the ProteinWorkshop
Apptainer image (which has graphein + torch_geometric + proteinworkshop
installed). The script downloads PDB files, converts to PyG graphs, and
aggregates {train, val, test}.pt for each of the three datasets used by
the task.

Output:
    <data_root>/ProteinWorkshop/ECReaction/processed/{train,val,test}.pt
    <data_root>/ProteinWorkshop/GeneOntology/processed/{train,val,test}.pt
    <data_root>/ProteinWorkshop/FoldClassification/processed/{train,val,test}.pt

Sources:
  - Split text files + raw PDBs auto-pulled by proteinworkshop's own
    dataset downloaders (e.g. EnzymeCommissionReactionDataset.download()).
    Backed by Zenodo dataset 10.5281/zenodo.8282470 plus rcsb.org PDB.
  - SCOPe ASTRAL pdbstyle-sel-gs-bib-95-1.75.tgz (1.34 GB) for Fold:
    upstream is scop.berkeley.edu, which is unreachable from our network.
    We mirror via the Internet Archive Wayback Machine snapshot from
    2024-08-08 and pre-stage the archive on the host so the in-container
    download_structures() finds it on disk and skips the network call.

Run via:
    mlsbench data ProteinWorkshop
"""

import argparse
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


DATASETS = ["ECReaction", "GeneOntology", "FoldClassification"]
SPLITS = ["train", "val", "test"]


# ---- SCOPe pdbstyle pre-stage ------------------------------------------------
# scop.berkeley.edu times out from compute nodes, so the in-container
# wget.download(scop_url, ...) inside FoldClassificationDataModule.download_structures
# never returns. Pre-download the archive on the host (which has a working
# WAN route to archive.org), drop it next to where the container expects it,
# and the container side will skip the download branch.
SCOP_ARCHIVE = "pdbstyle-sel-gs-bib-95-1.75.tgz"
SCOP_EXPECTED_BYTES = 1336065122  # exact upstream Content-Length from 2009 release
SCOP_MIRRORS = [
    # Wayback Machine snapshot of the original Berkeley URL. The ``id_``
    # suffix returns the raw archived bytes (gzip), bypassing the wayback
    # toolbar/replay HTML wrapper that the bare URL serves on cache miss.
    # Verified 2024-08-08 capture: 1336065122 bytes, application/x-gzip,
    # original Last-Modified 2009-06-03 (upstream is frozen).
    "https://web.archive.org/web/20240808143140id_/"
    "https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-95-1.75.tgz",
    # Bare URL as a backstop; works only when wayback's edge cache already
    # holds the binary, but cheap to try.
    "https://web.archive.org/web/20240808143140/"
    "https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-95-1.75.tgz",
]


def _download_with_retries(url: str, dest: Path, attempts: int = 5) -> bool:
    """Stream a URL to ``dest`` with bounded exponential backoff.

    Mirrors the retry pattern in vendor/data_scripts/ClimSim/prepare_data.py
    (5 attempts, exponential backoff capped at 60s). Writes to a .part file
    and atomically renames on success so a partial download can never satisfy
    the "file exists, skip download" check.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"  [SCOPe] Attempt {attempt}/{attempts}: {url}", flush=True)
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "mlsbench-prep/1.0",
                    # Force raw bytes; wayback may otherwise transcode.
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                expected = resp.headers.get("Content-Length")
                # Reject text/html error pages that wayback occasionally
                # serves on cache miss. The real archive is application/x-gzip.
                if "html" in ctype:
                    raise RuntimeError(
                        f"unexpected content-type {ctype!r}; "
                        "wayback served replay HTML, not the archive"
                    )
                with open(tmp, "wb") as fh:
                    shutil.copyfileobj(resp, fh, length=8 * 1024 * 1024)
            size = tmp.stat().st_size
            if expected is not None and int(expected) != size:
                raise RuntimeError(
                    f"size mismatch: got {size} bytes, expected {expected}"
                )
            if size < 1_000_000_000:
                # Anything substantially smaller than the 1.34 GB upstream file
                # is almost certainly an HTML error page or wayback redirect.
                raise RuntimeError(
                    f"downloaded file too small ({size} bytes); refusing to use"
                )
            tmp.replace(dest)
            print(f"  [SCOPe] Saved {size} bytes -> {dest}", flush=True)
            return True
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if tmp.exists():
                tmp.unlink()
            wait = min(60, 5 * attempt)
            print(
                f"  [SCOPe] attempt {attempt}/{attempts} failed: {exc}; "
                f"retry in {wait}s",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(wait)
    print(f"  [SCOPe] giving up on {url}: {last_err}", flush=True)
    return False


def prestage_scop_archive(workshop_dir: Path) -> bool:
    """Ensure the SCOPe pdbstyle tarball is on disk before container runs.

    Returns True if the archive is present (already-cached or freshly
    downloaded), False if every mirror failed.
    """
    fold_dir = workshop_dir / "FoldClassification"
    fold_dir.mkdir(parents=True, exist_ok=True)
    dest = fold_dir / SCOP_ARCHIVE
    extracted = fold_dir / "pdbstyle-1.75"

    if extracted.exists() and any(extracted.rglob("*.ent")):
        print(f"  [SCOPe] {extracted} already extracted, skip prestage")
        return True
    if dest.exists() and dest.stat().st_size == SCOP_EXPECTED_BYTES:
        print(f"  [SCOPe] {dest} already present ({dest.stat().st_size} bytes)")
        return True
    if dest.exists():
        # Truncated / wrong-size leftover from a previous failed run.
        print(
            f"  [SCOPe] {dest} present but {dest.stat().st_size} bytes "
            f"(expected {SCOP_EXPECTED_BYTES}); re-downloading"
        )
        dest.unlink()

    print(f"  [SCOPe] Pre-staging {SCOP_ARCHIVE} into {fold_dir}", flush=True)
    for url in SCOP_MIRRORS:
        if _download_with_retries(url, dest):
            return True
    return False


# ---- preprocessing driver ----------------------------------------------------
def have_processed(workshop_dir: Path) -> bool:
    for ds in DATASETS:
        for sp in SPLITS:
            if not (workshop_dir / ds / "processed" / f"{sp}.pt").exists():
                return False
    return True


def run_inside_apptainer(workshop_dir: Path, project_root: Path, script: Path, pkg_dir: Path) -> bool:
    sif = project_root / "vendor" / "images" / "ProteinWorkshop.sif"
    if not sif.exists():
        return False
    cmd = [
        "apptainer", "exec", "--nv",
        "--bind", f"{workshop_dir}:/data/ProteinWorkshop",
        "--bind", f"{script}:/data/ProteinWorkshop/preprocess_protein_workshop.py",
        "--bind", f"{pkg_dir}:/workspace/ProteinWorkshop",
        "--env", "PROTEIN_WORKSHOP_DATA_DIR=/data/ProteinWorkshop",
        "--env", "PYTHONPATH=/workspace/ProteinWorkshop",
        str(sif),
        "python", "/data/ProteinWorkshop/preprocess_protein_workshop.py", "--task", "all",
    ]
    print(f"  Running ProteinWorkshop preprocessing inside {sif.name}...", flush=True)
    res = subprocess.run(cmd)
    return res.returncode == 0


def run_inside_docker(workshop_dir: Path, project_root: Path, script: Path, pkg_dir: Path) -> bool:
    inspect = subprocess.run(
        ["docker", "image", "inspect", "mlsbench/proteinworkshop:latest"],
        capture_output=True,
    )
    if inspect.returncode != 0:
        return False
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--entrypoint", "",
        "-v", f"{workshop_dir}:/data/ProteinWorkshop",
        "-v", f"{script}:/data/ProteinWorkshop/preprocess_protein_workshop.py",
        "-v", f"{pkg_dir}:/workspace/ProteinWorkshop",
        "-e", "PROTEIN_WORKSHOP_DATA_DIR=/data/ProteinWorkshop",
        "-e", "PYTHONPATH=/workspace/ProteinWorkshop",
        "-w", "/workspace",
        "mlsbench/proteinworkshop:latest",
        "python", "/data/ProteinWorkshop/preprocess_protein_workshop.py", "--task", "all",
    ]
    print("  Running ProteinWorkshop preprocessing inside docker mlsbench/proteinworkshop:latest...", flush=True)
    res = subprocess.run(cmd)
    return res.returncode == 0


def run_inside_image(workshop_dir: Path) -> bool:
    project_root = Path(__file__).resolve().parents[3]
    script = Path(__file__).resolve().parent / "preprocess_protein_workshop.py"
    if not script.exists():
        print(f"ERROR: missing {script}", file=sys.stderr)
        return False
    pkg_dir = project_root / "vendor" / "external_packages" / "ProteinWorkshop"
    if run_inside_apptainer(workshop_dir, project_root, script, pkg_dir):
        return True
    return run_inside_docker(workshop_dir, project_root, script, pkg_dir)


def verify(workshop_dir: Path) -> None:
    missing = []
    for ds in DATASETS:
        for sp in SPLITS:
            p = workshop_dir / ds / "processed" / f"{sp}.pt"
            if not p.exists():
                missing.append(str(p.relative_to(workshop_dir)))
    if missing:
        print("ERROR: missing artifacts:", missing, file=sys.stderr)
        sys.exit(1)
    print("All ProteinWorkshop data verified.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()

    workshop_dir = Path(args.data_root) / "ProteinWorkshop"
    workshop_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Preparing ProteinWorkshop data at {workshop_dir} ===")

    if have_processed(workshop_dir):
        print("  [SKIP] all processed/*.pt files already present")
        return

    # Stage the SCOPe archive on the host BEFORE container runs. Upstream
    # scop.berkeley.edu is unreachable; without this, FoldClassification
    # preprocessing hangs forever inside the container.
    if not prestage_scop_archive(workshop_dir):
        print(
            "\n  ERROR: every mirror for pdbstyle-sel-gs-bib-95-1.75.tgz "
            "failed to download.\n"
            "  Manually drop the archive at\n"
            f"    {workshop_dir / 'FoldClassification' / SCOP_ARCHIVE}\n"
            "  and re-run.",
            file=sys.stderr,
        )
        sys.exit(3)

    if not run_inside_image(workshop_dir):
        print(
            "\n  ProteinWorkshop Apptainer image not built (or preprocessing failed).\n"
            "  Run `mlsbench build ProteinWorkshop`, then re-run\n"
            f"  `python vendor/data_scripts/ProteinWorkshop/prepare_data.py --data-root {args.data_root}`.\n"
            "  Preprocessing pulls split files + PDB structures automatically through\n"
            "  proteinworkshop's dataset downloaders.",
            file=sys.stderr,
        )
        sys.exit(2)

    verify(workshop_dir)


if __name__ == "__main__":
    main()
