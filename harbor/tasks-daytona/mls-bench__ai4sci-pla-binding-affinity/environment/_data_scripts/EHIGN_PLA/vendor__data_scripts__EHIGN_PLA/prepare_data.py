#!/usr/bin/env python3
"""Prepare data for EHIGN_PLA (ai4sci-pla-binding-affinity).

Downloads the upstream preprocessed graph bundle (Google Drive id
``1oGUP4z7htNXyxTqx95HNSDLsaoxa3fX7``) referenced from the EHIGN_PLA
README, sources the per-split CSVs from the cloned upstream repo
(``vendor/external_packages/EHIGN_PLA/data/<split>.csv``), and
aggregates the per-complex DGL heterographs into the list-of-dict
torch.save bundles the task expects:

    <data_root>/EHIGN_PLA/{train,valid,test2013,test2016,test2019}_data.pt
    <data_root>/EHIGN_PLA/{train,valid,test2013,test2016,test2019}.csv

The GDrive bundle layout (recovered from the original setup session,
see comment in the inline conversion script below) is

    <split>/<pdbid>/Graph_EHIGN-<pdbid>.dgl

where each ``.dgl`` is ``torch.save((heterograph, label))`` from
upstream ``EHIGN_PLA/graph_constructor.py:mols2graphs``. The
heterograph has node types ``ligand`` / ``pocket`` and edge types
``intra_l`` / ``intra_p`` / ``inter_l2p`` / ``inter_p2l`` with node
features under ``g.ndata['h']`` and edge features under ``g.edata['e']``.

Aggregation runs in the already-built runtime for the package:
Apptainer image, Docker image, or the current local/Conda Python. This
keeps DGL out of the host environment for container users while still
supporting local runtime builds.

Run via:
    mlsbench data EHIGN_PLA
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


GDRIVE_BUNDLE_ID = "1oGUP4z7htNXyxTqx95HNSDLsaoxa3fX7"

EXPECTED_PT = [
    "train_data.pt",
    "valid_data.pt",
    "test2013_data.pt",
    "test2016_data.pt",
    "test2019_data.pt",
]
EXPECTED_CSV = [
    "train.csv",
    "valid.csv",
    "test2013.csv",
    "test2016.csv",
    "test2019.csv",
]


def gdrive_download(file_id: str, dest: Path, retries: int = 5) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  [SKIP] {dest} already downloaded")
        return
    try:
        import gdown  # type: ignore

        gdown.download(id=file_id, output=str(dest), quiet=False)
        if dest.exists() and dest.stat().st_size > 10_000:
            return
    except ImportError:
        pass
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            page = urllib.request.urlopen(
                "https://drive.google.com/uc?export=download&id=" + file_id
            ).read().decode(errors="ignore")
            if "Quota exceeded" in page:
                wait = 60 * attempt
                print(f"  Quota exceeded, retry in {wait}s ({attempt}/{retries})")
                time.sleep(wait)
                continue
            m = re.search(r'uuid.*?value="([^"]+)', page)
            if not m:
                raise RuntimeError("No uuid in confirm page")
            url = (
                "https://drive.usercontent.google.com/download"
                f"?id={file_id}&export=download&confirm=t&uuid={m.group(1)}"
            )
            urllib.request.urlretrieve(url, str(dest))
            if dest.exists() and dest.stat().st_size > 10_000:
                print(f"  Downloaded {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  Error: {e}, retry in {60 * attempt}s")
            time.sleep(60 * attempt)
    raise RuntimeError(f"Failed to fetch GDrive id={file_id}: {last_err}")


def fetch_bundle(staging: Path) -> Path:
    staging.mkdir(parents=True, exist_ok=True)
    archive = staging / "EHIGN_PLA_graphs.zip"
    extracted = staging / "extracted"
    if extracted.exists() and any(extracted.iterdir()):
        return extracted
    gdrive_download(GDRIVE_BUNDLE_ID, archive)

    if archive.exists() and archive.stat().st_size > 10_000:
        extracted.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(str(archive)) as zf:
                zf.extractall(str(extracted))
            archive.unlink()
        except zipfile.BadZipFile:
            print(f"  Bundle {archive} is not a zip; trying tarball")
            import tarfile

            with tarfile.open(str(archive)) as tf:
                tf.extractall(str(extracted))
            archive.unlink()
    return extracted


def copy_split_csvs(target: Path) -> None:
    """Copy upstream split CSVs from the cloned EHIGN_PLA repo into target/.

    The upstream repo (cloned via mlsbench fetch into vendor/external_packages)
    ships data/{train,valid,test2013,test2016,test2019}.csv with columns
    pdbid + -logKd/Ki.
    """
    project_root = Path(__file__).resolve().parents[3]
    repo_data = project_root / "vendor" / "external_packages" / "EHIGN_PLA" / "data"
    if not repo_data.is_dir():
        raise RuntimeError(
            f"Upstream EHIGN_PLA repo not found at {repo_data}. "
            "Run `mlsbench fetch --name EHIGN_PLA` first."
        )
    for csv_name in EXPECTED_CSV:
        src = repo_data / csv_name
        dst = target / csv_name
        if not src.exists():
            raise RuntimeError(f"Missing split CSV in upstream repo: {src}")
        if not dst.exists():
            shutil.copy2(src, dst)
    print(f"  Copied 5 split CSVs from {repo_data}")


def have_outputs(target: Path) -> bool:
    return all((target / f).exists() for f in EXPECTED_PT + EXPECTED_CSV)


def conversion_inline_py() -> str:
    """Return the inline Python aggregating per-complex .dgl into .pt lists.

    Runs where DGL is available. Container backends bind the staging
    directory at /staging and the output directory at /target; local
    execution passes EHIGN_STAGING/EHIGN_TARGET in the environment.
    Bundle layout (recovered from the original setup session):
    <staging>/<split>/<pdbid>/Graph_EHIGN-<pdbid>.dgl.
    """
    return r"""
import os, sys, torch
import pandas as pd

# Some PyTorch builds carry a half-stubbed dgl.graphbolt that fails to
# import because of a missing C++ extension. Real EHIGN_PLA only needs
# core dgl, so neutralise graphbolt before the dgl import.
try:
    import dgl.graphbolt  # noqa: F401
except Exception:
    import importlib, types
    sys.modules['dgl.graphbolt'] = types.ModuleType('dgl.graphbolt')

import dgl  # noqa: F401

# Bundle was pickled when DGL still exposed ``DGLHeteroGraph`` at
# ``dgl.heterograph.DGLHeteroGraph``. In modern DGL (>=0.5) the unified
# graph class is ``dgl.DGLGraph`` (always heterograph-capable) and the
# old attribute is gone, so torch.load fails with
# ``Can't get attribute 'DGLHeteroGraph'``. Restore the alias both via
# the live module reference *and* via sys.modules so pickle's import-by-
# name path finds it whichever route python takes.
import dgl.heterograph as _dgl_hetero
if not hasattr(_dgl_hetero, 'DGLHeteroGraph'):
    _dgl_hetero.DGLHeteroGraph = dgl.DGLGraph
sys.modules['dgl.heterograph'].DGLHeteroGraph = dgl.DGLGraph
print('SHIM_OK', 'DGLHeteroGraph' in dir(sys.modules['dgl.heterograph']), flush=True)

STAGING = os.environ.get('EHIGN_STAGING', '/staging')
TARGET = os.environ.get('EHIGN_TARGET', '/target')
SPLITS = ['train', 'valid', 'test2013', 'test2016', 'test2019']

# Locate the directory that actually contains <split>/<pdbid>/Graph_EHIGN-*.dgl
# (the GDrive bundle wraps content in one extra dir).
def find_split_root():
    for root, dirs, _ in os.walk(STAGING):
        names = set(dirs)
        if all(s in names for s in SPLITS):
            return root
    return None

split_root = find_split_root()
if split_root is None:
    print('FATAL: bundle does not have all 5 split dirs under', STAGING, file=sys.stderr)
    sys.exit(1)
print('Bundle split root:', split_root)


def heterograph_to_dict(g, label):
    # Node features: g.ndata['h'] returns a dict keyed by node type for
    # heterographs.
    h = g.ndata['h']
    e = g.edata['e']
    lig_x = h['ligand'].float()
    poc_x = h['pocket'].float()

    def edge_pair(etype):
        u, v = g.edges(etype=etype)
        return torch.stack([u, v], dim=0).long()

    lig_edge_index = edge_pair('intra_l')
    poc_edge_index = edge_pair('intra_p')
    l2p_edge_index = edge_pair('inter_l2p')
    p2l_edge_index = edge_pair('inter_p2l')

    lig_edge_attr = e[('ligand', 'intra_l', 'ligand')].float()
    poc_edge_attr = e[('pocket', 'intra_p', 'pocket')].float()
    l2p_edge_attr = e[('ligand', 'inter_l2p', 'pocket')].float()
    p2l_edge_attr = e[('pocket', 'inter_p2l', 'ligand')].float()

    return {
        'lig_x': lig_x,
        'lig_edge_index': lig_edge_index,
        'lig_edge_attr': lig_edge_attr,
        'poc_x': poc_x,
        'poc_edge_index': poc_edge_index,
        'poc_edge_attr': poc_edge_attr,
        'l2p_edge_index': l2p_edge_index,
        'l2p_edge_attr': l2p_edge_attr,
        'p2l_edge_index': p2l_edge_index,
        'p2l_edge_attr': p2l_edge_attr,
        'label': torch.tensor([float(label)], dtype=torch.float32),
        'num_lig_atoms': int(lig_x.shape[0]),
        'num_poc_atoms': int(poc_x.shape[0]),
    }


for split in SPLITS:
    out_pt = os.path.join(TARGET, f'{split}_data.pt')
    # Treat zero-byte aggregates from previous failed runs as missing so
    # re-running the script repairs them rather than silently skipping.
    if os.path.exists(out_pt) and os.path.getsize(out_pt) > 1024:
        print(f'SKIP {out_pt}'); continue
    csv_path = os.path.join(TARGET, f'{split}.csv')
    if not os.path.exists(csv_path):
        print(f'FATAL: missing {csv_path}', file=sys.stderr); sys.exit(2)
    df = pd.read_csv(csv_path)

    items = []
    missing = 0
    for _, row in df.iterrows():
        pid = str(row['pdbid'])
        label = float(row['-logKd/Ki'])
        # Primary upstream layout, plus a few historical fallbacks.
        candidates = [
            os.path.join(split_root, split, pid, f'Graph_EHIGN-{pid}.dgl'),
            os.path.join(split_root, split, pid, f'{pid}.dgl'),
            os.path.join(split_root, split, f'{pid}.dgl'),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            missing += 1
            if missing <= 5:
                print(f'  MISSING graph for {pid}; tried {candidates[0]}')
            continue
        try:
            g, _ = torch.load(path, weights_only=False)
        except Exception as ex:
            missing += 1
            if missing <= 5:
                print(f'  load fail {path}: {ex}')
            continue
        try:
            items.append(heterograph_to_dict(g, label))
        except Exception as ex:
            missing += 1
            if missing <= 5:
                print(f'  extract fail {pid}: {ex}')
    print(f'{split}: aggregated {len(items)}, missing {missing} of {len(df)}')
    torch.save(items, out_pt)
    print(f'  wrote {out_pt}')
"""


def run_conversion(target: Path, staging_extracted: Path) -> bool:
    project_root = Path(__file__).resolve().parents[3]
    inline = conversion_inline_py()

    # Prefer the runtime image the user already built. This supports all
    # three MLS-Bench runtimes without requiring Apptainer on Docker/local
    # installations.
    sif = project_root / "vendor" / "images" / "EHIGN_PLA.sif"
    if sif.exists() and shutil.which("apptainer"):
        cmd = [
            "apptainer", "exec",
            "--bind", f"{staging_extracted}:/staging:ro",
            "--bind", f"{target}:/target",
            str(sif),
            "python", "-c", inline,
        ]
        print("  Running EHIGN_PLA DGL->pt aggregation via Apptainer...", flush=True)
        res = subprocess.run(cmd)
        if res.returncode == 0:
            return True

    docker_tag = "mlsbench/ehign_pla:latest"
    if shutil.which("docker"):
        inspect = subprocess.run(
            ["docker", "image", "inspect", docker_tag],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if inspect.returncode == 0:
            cmd = [
                "docker", "run", "--rm", "--entrypoint", "",
                "-v", f"{staging_extracted}:/staging:ro",
                "-v", f"{target}:/target",
                docker_tag,
                "python", "-c", inline,
            ]
            print("  Running EHIGN_PLA DGL->pt aggregation via Docker...", flush=True)
            res = subprocess.run(cmd)
            if res.returncode == 0:
                return True

    env = os.environ.copy()
    env["EHIGN_STAGING"] = str(staging_extracted)
    env["EHIGN_TARGET"] = str(target)
    print("  Running EHIGN_PLA DGL->pt aggregation via local Python...", flush=True)
    res = subprocess.run([sys.executable, "-c", inline], env=env)
    return res.returncode == 0


def verify(target: Path) -> None:
    missing = [f for f in EXPECTED_PT + EXPECTED_CSV if not (target / f).exists()]
    if missing:
        print("ERROR: missing artifacts:", missing, file=sys.stderr)
        sys.exit(1)
    # Existence is not enough — the conversion can write empty lists if the
    # bundle layout drifts. Fail loud when any split has zero items.
    import torch as _torch  # local import; user's host might not always have torch

    empty = []
    for f in EXPECTED_PT:
        try:
            data = _torch.load(target / f, weights_only=False)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: cannot load {f}: {e}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(data, list) or len(data) == 0:
            empty.append(f)
    if empty:
        print(
            "ERROR: empty .pt aggregates: " + ", ".join(empty)
            + ". Bundle layout likely changed; inspect the staging extracted dir.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("All EHIGN_PLA data verified.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()

    target = Path(args.data_root) / "EHIGN_PLA"
    target.mkdir(parents=True, exist_ok=True)
    print(f"=== Preparing EHIGN_PLA data at {target} ===")

    if have_outputs(target):
        print("  [SKIP] all .pt/.csv files already present")
        verify(target)
        return

    copy_split_csvs(target)

    staging = target / "_staging"
    bundle = fetch_bundle(staging)
    print(f"  Upstream bundle extracted at: {bundle}")

    ok = run_conversion(target, bundle)
    if not ok:
        print(
            "\n  EHIGN_PLA DGL aggregation failed in Apptainer, Docker, and local Python.\n"
            "  Run `mlsbench build EHIGN_PLA` for your configured runtime, then re-run\n"
            f"  `python vendor/data_scripts/EHIGN_PLA/prepare_data.py --data-root {args.data_root}`.",
            file=sys.stderr,
        )
        sys.exit(2)

    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    verify(target)


if __name__ == "__main__":
    main()
