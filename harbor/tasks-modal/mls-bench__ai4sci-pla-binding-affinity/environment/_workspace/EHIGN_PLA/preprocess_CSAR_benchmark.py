# %%
import os
import pymol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# %%
data_dir = '/data2/yzd/docking/CSAR_NRC_HiQ_Set/Structures'
set1_dir = os.path.join(data_dir, 'set1')
set2_dir = os.path.join(data_dir, 'set2')

set_id = 'set1'
set_dir = set1_dir
for data_id in os.listdir(set_dir):
    if os.path.exists(os.path.join(set_dir, data_id, f'Pocket_5A.pdb')):
        os.remove(os.path.join(set_dir, data_id, f'Pocket_5A.pdb'))
    if os.path.exists(os.path.join(set_dir, data_id, f'Ligand.pdb')):
        os.remove(os.path.join(set_dir, data_id, f'Ligand.pdb'))
    if os.path.exists(os.path.join(set_dir, data_id, f'Ligand.mol2')):
        os.remove(os.path.join(set_dir, data_id, f'Ligand.mol2'))
    if os.path.exists(os.path.join(set_dir, data_id, f'Ligand.sdf')):
        os.remove(os.path.join(set_dir, data_id, f'Ligand.sdf'))

    complex_path = os.path.join(set_dir, data_id, f'{set_id}_{data_id}_complex_min.mol2')
    pymol.cmd.load(complex_path)
    pymol.cmd.remove('resn HOH')
    pymol.cmd.select('Ligand', f'resn INH')
    pymol.cmd.save(os.path.join(set_dir, data_id, f'Ligand.sdf'), 'Ligand')
    pymol.cmd.deselect()
    pymol.cmd.remove('hydrogens')
    pymol.cmd.select('Pocket', f'byres (resn INH) around 5')
    pymol.cmd.save(os.path.join(set_dir, data_id, f'Pocket_5A.pdb'), 'Pocket')
    pymol.cmd.deselect()
    pymol.cmd.delete('all')

    ligand_sdf_path = os.path.join(set_dir, data_id, f'Ligand.sdf')
    ligand_pdb_path = ligand_sdf_path.replace(".sdf", ".pdb")
    os.system(f'obabel {ligand_sdf_path} -O {ligand_pdb_path} -d')