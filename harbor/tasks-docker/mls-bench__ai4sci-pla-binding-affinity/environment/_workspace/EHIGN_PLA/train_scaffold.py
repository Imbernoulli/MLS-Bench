# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from graph_constructor import GraphDataset, collate_fn
from EHIGN import DTIPredictor

import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from config.config_dict import *
from log.train_logger import *
from utils import *

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
    cfg = 'TrainConfig_scaffold'
    config = Config(cfg)
    args = config.get_config()
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    batch_size = args.get("batch_size")
    data_root = args.get('data_root')
    epochs = args.get('epochs')
    repeats = args.get('repeat')
    early_stop_epoch = args.get("early_stop_epoch")

    for repeat in range(repeats):
        args['repeat'] = repeat

        train_dir = os.path.join(data_root, 'train')

        train_df = pd.read_csv(os.path.join('./cold_start_data', 'train_scaffold.csv'))
        valid_df = pd.read_csv(os.path.join('./cold_start_data', 'valid_scaffold.csv'))
        test_df = pd.read_csv(os.path.join('./cold_start_data', 'test_scaffold.csv'))

        train_set = GraphDataset(train_dir, train_df, graph_type=graph_type, create=False)
        valid_set = GraphDataset(train_dir, valid_df, graph_type=graph_type, create=False)
        test_set = GraphDataset(train_dir, test_df, graph_type=graph_type, create=False)

        train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4)

        logger = TrainLogger(args, cfg, create=True)
        logger.info(__file__)
        logger.info(f"train data: {len(train_set)}")
        logger.info(f"valid data: {len(valid_set)}")
        logger.info(f"test2013 data: {len(test_set)}")

        device = torch.device('cuda:0')
        model =  DTIPredictor(node_feat_size=35, edge_feat_size=17, hidden_feat_size=256).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        criterion = nn.MSELoss()

        running_loss = AverageMeter()
        running_acc = AverageMeter()
        running_best_mse = BestMeter("min")
        best_model_list = []

        # start training
        model.train()
        for epoch in range(epochs):
            for data in train_loader:
                bg, label = data
                bg, label = bg.to(device), label.to(device)

                pred_lp, pred_pl = model(bg)
                loss = (criterion(pred_lp, label) + criterion(pred_pl, label) + criterion(pred_lp, pred_pl)) / 3
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.update(loss.item(), label.size(0)) 

            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            running_loss.reset()

            # start validating
            valid_rmse, valid_pr = val(model, valid_loader, device)
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
            logger.info(msg)

            if valid_rmse < running_best_mse.get_best():
                running_best_mse.update(valid_rmse)
                if save_model:
                    msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
                    model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
                    best_model_list.append(model_path)
                    save_model_dict(model, logger.get_model_dir(), msg)
            else:
                count = running_best_mse.counter()
                if count > early_stop_epoch:
                    best_mse = running_best_mse.get_best()
                    msg = "best_rmse: %.4f" % best_mse
                    logger.info(f"early stop in epoch {epoch}")
                    logger.info(msg)
                    break_flag = True
                    break
            
            time.sleep(1)

        # final testing
        load_model_dict(model, best_model_list[-1])
        valid_rmse, valid_pr = val(model, valid_loader, device)
        test_rmse, test_pr = val(model, test_loader, device)

        msg = "valid_rmse-%.4f, valid_pr-%.4f, test_rmse-%.4f, test_pr-%.4f" \
                    % (valid_rmse, valid_pr, test_rmse, test_pr)

        logger.info(msg)
# %%