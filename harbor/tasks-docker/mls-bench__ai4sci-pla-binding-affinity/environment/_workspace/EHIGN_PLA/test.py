
# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from graph_constructor import GraphDataset, collate_fn
from EHIGN import DTIPredictor
from utils import *

from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

import warnings
warnings.filterwarnings('ignore')


# %%
def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        bg, label = data
        bg, label = bg.to(device), label.to(device)

        with torch.no_grad():
            pred_lp, pred_pl = model(bg)
            pred = (pred_lp + pred_pl) / 2
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    pr = pearsonr(pred, label)[0]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()

    return rmse, pr

if __name__ == '__main__':
    data_root = './data'

    valid_dir = os.path.join(data_root, 'valid')
    test2013_dir = os.path.join(data_root, 'test2013')
    test2016_dir = os.path.join(data_root, 'test2016')
    test2019_dir = os.path.join(data_root, 'test2019')

    train_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(data_root, 'valid.csv'))
    test2013_df = pd.read_csv(os.path.join(data_root, 'test2013.csv'))
    test2016_df = pd.read_csv(os.path.join(data_root, 'test2016.csv'))
    test2019_df = pd.read_csv(os.path.join(data_root, 'test2019.csv'))

    valid_set = GraphDataset(valid_dir, valid_df, graph_type='Graph_EHIGN', create=False)
    test2013_set = GraphDataset(test2013_dir, test2013_df, graph_type='Graph_EHIGN', create=False)
    test2016_set = GraphDataset(test2016_dir, test2016_df, graph_type='Graph_EHIGN', create=False)
    test2019_set = GraphDataset(test2019_dir, test2019_df, graph_type='Graph_EHIGN', create=False)

    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=8)
    test2013_loader = DataLoader(test2013_set, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=8)
    test2016_loader = DataLoader(test2016_set, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=8)
    test2019_loader = DataLoader(test2019_set, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=8)

    device = torch.device('cuda:0')
    model =  DTIPredictor(node_feat_size=35, edge_feat_size=17, hidden_feat_size=256, layer_num=3).to(device)
    load_model_dict(model, './model/20230120_135757_EHIGN_repeat0/model/epoch-144, train_loss-0.5772, train_rmse-0.7598, valid_rmse-1.1799, valid_pr-0.7718.pt')

    valid_rmse, valid_pr = val(model, valid_loader, device)
    test2013_rmse, test2013_pr = val(model, test2013_loader, device)
    test2016_rmse, test2016_pr = val(model, test2016_loader, device)
    test2019_rmse, test2019_pr = val(model, test2019_loader, device)
    msg = "valid_rmse-%.4f, valid_pr-%.4f, test2013_rmse-%.4f, test2013_pr-%.4f, test2016_rmse-%.4f, test2016_pr-%.4f, test2019_rmse-%.4f, test2019_pr-%.4f," \
                % (valid_rmse, valid_pr, test2013_rmse, test2013_pr, test2016_rmse, test2016_pr, test2019_rmse, test2019_pr)
    print(msg)


# %%
