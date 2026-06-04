#!/usr/bin/env python3
"""Preprocess ProteinWorkshop datasets: download PDB files, convert to PyG graphs,
and aggregate into train/val/test.pt for offline use on compute nodes.

Run inside the ProteinWorkshop container on a login node (with internet):
    bash scripts/run_preprocess_protein_workshop.sh [ec|go|fold|all]
"""

import os
import sys
import ssl
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Disable SSL verification globally (container lacks CA certs)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/workspace/ProteinWorkshop')
os.environ.setdefault('PROTEIN_WORKSHOP_DATA_DIR', '/data/ProteinWorkshop')

DATA_DIR = Path(os.environ['PROTEIN_WORKSHOP_DATA_DIR'])

NUM_AMINO_ACIDS = 20

TASK_META = {
    'ec_reaction': {'num_classes': 384, 'task_type': 'multilabel'},
    'go_bp': {'num_classes': 1943, 'task_type': 'multilabel'},
    'fold_fold': {'num_classes': 1195, 'task_type': 'multiclass'},
}


def monkeypatch_all():
    """Apply all necessary monkeypatches."""
    # 1. Patch obsolete PDB mapping to avoid network call
    from graphein.protein.utils import download_pdb
    gpu_mod = sys.modules[download_pdb.__module__]
    gpu_mod.get_obsolete_mapping = lambda: {}

    from proteinworkshop.datasets.base import ProteinDataModule
    ProteinDataModule.obsolete_pdbs = property(lambda self: {})


def download_pdbs_robust(pdb_codes, out_dir, format='pdb', max_workers=16):
    """Download PDB files in parallel using threads, skipping failures."""
    from graphein.protein.utils import download_pdb

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext_map = {'pdb': '.pdb', 'mmtf': '.mmtf.gz', 'ent': '.ent'}
    ext = ext_map.get(format, f'.{format}')

    # Filter to only those not yet downloaded
    to_download = [p for p in pdb_codes if not (out_dir / f"{p}{ext}").exists()]
    already = len(pdb_codes) - len(to_download)
    print(f"  {already} already downloaded, {len(to_download)} to download")

    if not to_download:
        return

    def safe_download(pdb_code):
        try:
            download_pdb(pdb_code, out_dir=out_dir, format=format, strict=False)
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(safe_download, to_download),
            total=len(to_download),
            desc="Downloading PDB files",
            unit="file",
        ))

    success = sum(results)
    print(f"  Downloaded {success}/{len(to_download)} files")


def process_pdbs_robust(pdb_codes, chains, raw_dir, processed_dir, format='pdb', graph_labels=None):
    """Process PDB files to PyG Data objects individually, skipping failures."""
    from graphein.protein.tensor.io import protein_to_pyg

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(raw_dir)

    ext_map = {'pdb': '.pdb', 'mmtf': '.mmtf.gz', 'ent': '.ent'}
    ext = ext_map.get(format, f'.{format}')

    processed = 0
    skipped = 0
    already_done = 0

    for i, pdb in enumerate(tqdm(pdb_codes, desc="Processing PDBs")):
        try:
            chain = chains[i] if chains is not None else None
            if chain is not None:
                out_fname = f"{pdb}_{chain}.pt"
            else:
                out_fname = f"{pdb}.pt"

            out_path = processed_dir / out_fname
            if out_path.exists():
                already_done += 1
                continue

            # Find raw file
            raw_path = raw_dir / f"{pdb}{ext}"
            if not raw_path.exists():
                # Try with .gz
                raw_path_gz = raw_dir / f"{pdb}{ext}.gz"
                if raw_path_gz.exists():
                    raw_path = raw_path_gz
                else:
                    skipped += 1
                    continue

            chain_sel = chain if chain is not None else "all"
            graph = protein_to_pyg(
                path=str(raw_path),
                chain_selection=chain_sel,
                keep_insertions=True,
                store_het=False,
            )

            if chain is not None:
                graph.id = f"{pdb}_{chain}"
            else:
                graph.id = pdb

            if graph_labels is not None:
                graph.graph_y = graph_labels[i]

            torch.save(graph, out_path)
            processed += 1

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  Warning: error processing {pdb}: {e}")

    print(f"  Processed: {processed}, Already done: {already_done}, Skipped: {skipped}")


def convert_label(y, task_type, num_classes):
    """Convert label to expected format (binary vector for multilabel, scalar for multiclass)."""
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    if y.dim() > 1:
        y = y.squeeze(0)

    if task_type == 'multilabel':
        binary = torch.zeros(num_classes, dtype=torch.float)
        if y.dim() == 0:
            idx = int(y.item())
            if 0 <= idx < num_classes:
                binary[idx] = 1.0
        else:
            for idx in y.long():
                if 0 <= idx < num_classes:
                    binary[idx] = 1.0
        return binary
    else:
        if y.dim() > 0:
            y = y[0] if y.numel() == 1 else y
        return y.long()


def aggregate_to_splits(task_name, pdb_codes, chains, labels, processed_dir, output_path):
    """Load individual .pt files and aggregate into a single .pt file."""
    from torch_geometric.data import Data

    meta = TASK_META[task_name]
    num_classes = meta['num_classes']
    task_type = meta['task_type']

    data_list = []
    skipped = 0

    for i, pdb in enumerate(pdb_codes):
        try:
            if chains is not None:
                fname = f"{pdb}_{chains[i]}.pt"
            else:
                fname = f"{pdb}.pt"

            fpath = Path(processed_dir) / fname
            if not fpath.exists():
                skipped += 1
                continue

            graph = torch.load(fpath, weights_only=False)

            # Extract alpha-carbon positions
            if hasattr(graph, 'coords') and graph.coords is not None:
                coords = graph.coords
                if coords.dim() == 3:
                    pos = coords[:, 1, :] if coords.size(1) >= 2 else coords[:, 0, :]
                elif coords.dim() == 2:
                    pos = coords
                else:
                    skipped += 1; continue
            elif hasattr(graph, 'pos') and graph.pos is not None:
                pos = graph.pos
            else:
                skipped += 1; continue

            if pos is None or pos.size(0) < 4:
                skipped += 1; continue

            if torch.isnan(pos).any():
                valid = ~torch.isnan(pos).any(dim=1)
                if valid.sum() < 4:
                    skipped += 1; continue
                pos = pos[valid]

            # Amino acid indices
            if hasattr(graph, 'residue_type') and graph.residue_type is not None:
                aa_idx = graph.residue_type.long()
                if aa_idx.dim() > 1:
                    aa_idx = aa_idx.squeeze()
            else:
                aa_idx = torch.zeros(pos.size(0), dtype=torch.long)

            L = pos.size(0)
            if aa_idx.size(0) != L:
                aa_idx = aa_idx[:L] if aa_idx.size(0) > L else F.pad(aa_idx, (0, L - aa_idx.size(0)))

            y = convert_label(labels[i], task_type, num_classes)

            data_list.append(Data(
                pos=pos.float(),
                aa_idx=aa_idx.clamp(0, NUM_AMINO_ACIDS - 1),
                y=y,
                num_nodes=L,
            ))

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  Warning aggregating {pdb}: {e}")

    print(f"  Aggregated: {len(data_list)}, Skipped: {skipped}")
    torch.save(data_list, output_path)
    print(f"  Saved to {output_path}")
    return data_list


def preprocess_ec_reaction():
    """Preprocess EC Reaction dataset."""
    print("\n=== Preprocessing EC Reaction ===")
    from proteinworkshop.datasets.ec_reaction import EnzymeCommissionReactionDataset

    ec_dir = DATA_DIR / 'ECReaction'
    pdb_dir = DATA_DIR / 'pdb'
    processed_dir = ec_dir / 'processed'

    dm = EnzymeCommissionReactionDataset(
        path=str(ec_dir), pdb_dir=str(pdb_dir), format='pdb',
        batch_size=1, num_workers=0, pin_memory=False,
        dataset_fraction=1.0, shuffle_labels=False,
    )
    dm.download()

    for split_name, split_key in [('train', 'training'), ('val', 'validation'), ('test', 'testing')]:
        output_path = processed_dir / f'{split_name}.pt'
        if output_path.exists():
            print(f"  {split_name}.pt already exists, skipping")
            continue

        print(f"\n  Processing {split_name} split...")
        df = dm.parse_dataset(split_key)
        pdb_codes = list(df.pdb)
        chain_list = list(df.chain)
        label_list = [torch.tensor(a) for a in list(df.label)]

        # 1. Download
        unique_pdbs = list(set(pdb_codes))
        download_pdbs_robust(unique_pdbs, pdb_dir, format='pdb')

        # 2. Process
        process_pdbs_robust(pdb_codes, chain_list, pdb_dir, processed_dir,
                           format='pdb', graph_labels=label_list)

        # 3. Aggregate
        aggregate_to_splits('ec_reaction', pdb_codes, chain_list, label_list,
                          processed_dir, output_path)


def preprocess_go_bp():
    """Preprocess Gene Ontology BP dataset."""
    print("\n=== Preprocessing GO-BP ===")
    from proteinworkshop.datasets.go import GeneOntologyDataset

    go_dir = DATA_DIR / 'GeneOntology'
    pdb_dir = DATA_DIR / 'pdb'
    processed_dir = go_dir / 'processed'

    dm = GeneOntologyDataset(
        path=str(go_dir), pdb_dir=str(pdb_dir), format='pdb',
        batch_size=1, num_workers=0, pin_memory=False,
        dataset_fraction=1.0, shuffle_labels=False, split='BP',
    )
    dm.download()

    for split_name, split_key in [('train', 'training'), ('val', 'validation'), ('test', 'test_0.95')]:
        output_path = processed_dir / f'{split_name}.pt'
        if output_path.exists():
            print(f"  {split_name}.pt already exists, skipping")
            continue

        print(f"\n  Processing {split_name} split...")
        df = dm.parse_dataset(split_key)
        pdb_codes = list(df.pdb)
        chain_list = list(df.chain)
        label_list = list(df.label)  # Already tensors from parse_labels

        unique_pdbs = list(set(pdb_codes))
        download_pdbs_robust(unique_pdbs, pdb_dir, format='pdb')
        process_pdbs_robust(pdb_codes, chain_list, pdb_dir, processed_dir,
                           format='pdb', graph_labels=label_list)
        aggregate_to_splits('go_bp', pdb_codes, chain_list, label_list,
                          processed_dir, output_path)


def preprocess_fold():
    """Preprocess Fold Classification dataset."""
    print("\n=== Preprocessing Fold Classification ===")
    from proteinworkshop.datasets.fold_classification import FoldClassificationDataModule

    fold_dir = DATA_DIR / 'FoldClassification'
    processed_dir = fold_dir / 'processed'
    structure_dir = fold_dir / 'pdbstyle-1.75'

    dm = FoldClassificationDataModule(
        path=str(fold_dir), batch_size=1, num_workers=0,
        pin_memory=False, dataset_fraction=1.0,
        shuffle_labels=False, split='fold',
    )
    dm.download()

    for split_name, split_key in [('train', 'training'), ('val', 'validation'), ('test', 'test_fold')]:
        output_path = processed_dir / f'{split_name}.pt'
        if output_path.exists():
            print(f"  {split_name}.pt already exists, skipping")
            continue

        print(f"\n  Processing {split_name} split...")
        df = dm.parse_dataset(split_key)
        pdb_codes = list(df.id)
        label_list = [torch.tensor(a) for a in list(df.label)]

        # Fold uses .ent files from SCOPe, no download needed (already extracted)
        process_pdbs_robust(pdb_codes, None, structure_dir, processed_dir,
                           format='ent', graph_labels=label_list)
        aggregate_to_splits('fold_fold', pdb_codes, None, label_list,
                          processed_dir, output_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'ec', 'go', 'fold'],
                        help='Which task to preprocess')
    args = parser.parse_args()

    monkeypatch_all()

    if args.task in ('all', 'ec'):
        try:
            preprocess_ec_reaction()
        except Exception as e:
            print(f"ERROR preprocessing EC: {e}")
            traceback.print_exc()

    if args.task in ('all', 'go'):
        try:
            preprocess_go_bp()
        except Exception as e:
            print(f"ERROR preprocessing GO: {e}")
            traceback.print_exc()

    if args.task in ('all', 'fold'):
        try:
            preprocess_fold()
        except Exception as e:
            print(f"ERROR preprocessing Fold: {e}")
            traceback.print_exc()

    print("\nDone!")
