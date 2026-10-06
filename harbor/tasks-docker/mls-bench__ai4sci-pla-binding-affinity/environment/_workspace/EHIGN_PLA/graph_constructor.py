# %%
import os
import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance_matrix
from utils import cal_dist, area_triangle, angle
import multiprocessing
from itertools import repeat
import networkx as nx
import torch 
from torch.utils.data import DataLoader
import dgl
from rdkit import Chem
from rdkit import RDLogger
from rdkit import Chem
import warnings
RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')

# %%
def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):

    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def edge_features(mol, graph):
    geom = mol.GetConformers()[0].GetPositions()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        angles_ijk = []
        areas_ijk = []
        dists_ik = []
        for neighbor in mol.GetAtomWithIdx(j).GetNeighbors():
            k = neighbor.GetIdx() 
            if mol.GetBondBetweenAtoms(j, k) is not None and i != k:
                vector1 = geom[j] - geom[i]
                vector2 = geom[k] - geom[i]

                angles_ijk.append(angle(vector1, vector2))
                areas_ijk.append(area_triangle(vector1, vector2))
                dists_ik.append(cal_dist(geom[i], geom[k]))

        angles_ijk = np.array(angles_ijk) if angles_ijk != [] else np.array([0.])
        areas_ijk = np.array(areas_ijk) if areas_ijk != [] else np.array([0.])
        dists_ik = np.array(dists_ik) if dists_ik != [] else np.array([0.])
        dist_ij1 = cal_dist(geom[i], geom[j], ord=1)
        dist_ij2 = cal_dist(geom[i], geom[j], ord=2)
        # length = 11
        geom_feats = [
            angles_ijk.max()*0.1,
            angles_ijk.sum()*0.01,
            angles_ijk.mean()*0.1,
            areas_ijk.max()*0.1,
            areas_ijk.sum()*0.01,
            areas_ijk.mean()*0.1,
            dists_ik.max()*0.1,
            dists_ik.sum()*0.01,
            dists_ik.mean()*0.1,
            dist_ij1*0.1,
            dist_ij2*0.1,
        ]

        bond_type = bond.GetBondType()
        basic_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]

        graph.add_edge(i, j, feats=torch.tensor(basic_feats+geom_feats).float())

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    edge_features(mol, graph)

    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v, feats in graph.edges(data=True)]).T
    edge_attr = torch.stack([feats['feats'] for u, v, feats in graph.edges(data=True)])

    return x, edge_index, edge_attr

def geom_feat(pos_i, pos_j, pos_k, angles_ijk, areas_ijk, dists_ik):
    vector1 = pos_j - pos_i
    vector2 = pos_k - pos_i
    angles_ijk.append(angle(vector1, vector2))
    areas_ijk.append(area_triangle(vector1, vector2))
    dists_ik.append(cal_dist(pos_i, pos_k))

def geom_feats(pos_i, pos_j, angles_ijk, areas_ijk, dists_ik):
    angles_ijk = np.array(angles_ijk) if angles_ijk != [] else np.array([0.])
    areas_ijk = np.array(areas_ijk) if areas_ijk != [] else np.array([0.])
    dists_ik = np.array(dists_ik) if dists_ik != [] else np.array([0.])
    dist_ij1 = cal_dist(pos_i, pos_j, ord=1)
    dist_ij2 = cal_dist(pos_i, pos_j, ord=2)
    # length = 11
    geom = [
        angles_ijk.max()*0.1,
        angles_ijk.sum()*0.01,
        angles_ijk.mean()*0.1,
        areas_ijk.max()*0.1,
        areas_ijk.sum()*0.01,
        areas_ijk.mean()*0.1,
        dists_ik.max()*0.1,
        dists_ik.sum()*0.01,
        dists_ik.mean()*0.1,
        dist_ij1*0.1,
        dist_ij2*0.1,
    ]

    return geom

def inter_graph(ligand, pocket, dis_threshold = 5.):
    graph_l2p = nx.DiGraph()
    graph_p2l = nx.DiGraph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    for i, j in zip(node_idx[0], node_idx[1]):
        ks = node_idx[0][node_idx[1] == j]
        angles_ijk = []
        areas_ijk = []
        dists_ik = []
        for k in ks:
            if k != i:
                geom_feat(pos_l[i], pos_p[j], pos_l[k], angles_ijk, areas_ijk, dists_ik)
        geom = geom_feats(pos_l[i], pos_p[j], angles_ijk, areas_ijk, dists_ik)
        bond_feats = torch.FloatTensor(geom)
        graph_l2p.add_edge(i, j, feats=bond_feats)
        ks = node_idx[1][node_idx[0] == i]
        angles_ijk = []
        areas_ijk = []
        dists_ik = []
        for k in ks:
            if k != j:
                geom_feat(pos_p[j], pos_l[i], pos_p[k], angles_ijk, areas_ijk, dists_ik)     
        geom = geom_feats(pos_p[j], pos_l[i], angles_ijk, areas_ijk, dists_ik)
        bond_feats = torch.FloatTensor(geom)
        graph_p2l.add_edge(j, i, feats=bond_feats)
    
    edge_index_l2p = torch.stack([torch.LongTensor((u, v)) for u, v, feats in graph_l2p.edges(data=True)]).T
    edge_attr_l2p = torch.stack([feats['feats'] for u, v, feats in graph_l2p.edges(data=True)])

    edge_index_p2l = torch.stack([torch.LongTensor((u, v)) for u, v, feats in graph_p2l.edges(data=True)]).T
    edge_attr_p2l = torch.stack([feats['feats'] for u, v, feats in graph_p2l.edges(data=True)])

    return (edge_index_l2p, edge_attr_l2p), (edge_index_p2l, edge_attr_p2l)

# %%
def mols2graphs(complex_path, label, save_path, dis_threshold=5.0):
    try:
        with open(complex_path, 'rb') as f:
            ligand, pocket = pickle.load(f)

        atom_num_l = ligand.GetNumAtoms()
        atom_num_p = pocket.GetNumAtoms()

        x_l, edge_index_l, edge_attr_l = mol2graph(ligand)
        x_p, edge_index_p, edge_attr_p = mol2graph(pocket)
        (edge_index_l2p, edge_attr_l2p), (edge_index_p2l, edge_attr_p2l) = inter_graph(ligand, pocket, dis_threshold=dis_threshold)
        
        graph_data = {
            ('ligand', 'intra_l', 'ligand') : (edge_index_l[0], edge_index_l[1]),
            ('pocket', 'intra_p', 'pocket') : (edge_index_p[0], edge_index_p[1]),
            ('ligand', 'inter_l2p', 'pocket') : (edge_index_l2p[0], edge_index_l2p[1]),
            ('pocket', 'inter_p2l', 'ligand') : (edge_index_p2l[0], edge_index_p2l[1])
        }
        g = dgl.heterograph(graph_data, num_nodes_dict={"ligand":atom_num_l, "pocket":atom_num_p})
        g.nodes['ligand'].data['h'] = x_l
        g.nodes['pocket'].data['h'] = x_p
        g.edges['intra_l'].data['e'] = edge_attr_l
        g.edges['intra_p'].data['e'] = edge_attr_p
        g.edges['inter_l2p'].data['e'] = edge_attr_l2p
        g.edges['inter_p2l'].data['e'] = edge_attr_p2l

        if torch.any(torch.isnan(edge_attr_l)) or torch.any(torch.isnan(edge_attr_p)):
            status = False
            print(save_path)
        else:
            status = True
    except:
        g = None
        status = False

    if status:
        torch.save((g, torch.FloatTensor([label])), save_path)

# %%
def collate_fn(data_batch):
    """
    used for dataset generated from GraphDatasetV2MulPro class
    :param data_batch:
    :return:
    """
    # print(data_batch)
    g, label = map(list, zip(*data_batch))
    bg = dgl.batch(g)
    y = torch.cat(label, dim=0)

    return bg, y

class GraphDataset(object):
    """
    This class is used for generating graph objects using multi process
    """

    def __init__(self, data_dir, data_df, dis_threshold=5.0, graph_type='Graph_EHIGN', num_process=48, create=True):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process
        self._pre_process()

    def _pre_process(self):

        data_dir = self.data_dir
        data_df = self.data_df
        graph_type = self.graph_type
        dis_thresholds = repeat(self.dis_threshold, len(data_df))

        complex_path_list = []
        complex_id_list = []
        pKa_list = []
        graph_path_list = []
        for i, row in data_df.iterrows():
            cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
            complex_dir = os.path.join(data_dir, cid)
            graph_path = os.path.join(complex_dir, f"{graph_type}-{cid}.dgl")
            complex_path = os.path.join(complex_dir, f"{cid}.rdkit")

            complex_path_list.append(complex_path)
            complex_id_list.append(cid)
            pKa_list.append(pKa)
            graph_path_list.append(graph_path)

        if self.create:
            print('Generate complex graph...')
            # multi-thread processing
            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(mols2graphs,
                            zip(complex_path_list, pKa_list, graph_path_list, dis_thresholds))
            pool.close()
            pool.join()

        self.graph_paths = graph_path_list
        self.complex_ids = complex_id_list

    def __getitem__(self, idx):
        return torch.load(self.graph_paths[idx])

    def __len__(self):
        return len(self.data_df)

if __name__ == '__main__':
    data_root = './data'
    toy_dir = os.path.join(data_root, 'toy_set')
    toy_df = pd.read_csv(os.path.join(data_root, "toy_examples.csv"))
    toy_set = GraphDataset(toy_dir, toy_df, graph_type='Graph_EHIGN', dis_threshold=5., create=True)
    toy_loader = DataLoader(toy_set, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=1)

# %%
