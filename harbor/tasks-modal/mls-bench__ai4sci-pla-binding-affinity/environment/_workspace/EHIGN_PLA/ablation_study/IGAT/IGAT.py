# %%
import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import global_add_pool, GATConv

class IGAT(nn.Module):
    def __init__(self, node_dim, hidden_dim, d_count=9):
        super().__init__()
        self.d_count = d_count

        self.lin_node = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
        self.lin_edge = nn.Sequential(Linear(d_count, hidden_dim), nn.SiLU())
    
        self.gconv1 = GATConv(hidden_dim, hidden_dim, heads=3, concat=False, edge_dim=hidden_dim)
        self.gconv2 = GATConv(hidden_dim, hidden_dim, heads=3, concat=False, edge_dim=hidden_dim)
        self.gconv3 = GATConv(hidden_dim, hidden_dim, heads=3, concat=False, edge_dim=hidden_dim)

        self.fc = FC(hidden_dim, hidden_dim, 3, 0.1, 1)

    def forward(self, data):
        x, edge_index_intra, edge_index_inter, pos = \
        data.x, data.edge_index_intra, data.edge_index_inter, data.pos
        # concat two types of edge_index to obtain homogeneous graph
        edge_index = torch.cat([edge_index_intra, edge_index_inter], dim=-1)

        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        edge_attr = self.lin_edge(_rbf(torch.norm(coord_diff, dim=-1), D_min=0., D_max=6., D_count=self.d_count, device=x.device))

        x = self.lin_node(x)
        x = self.gconv1(x, edge_index, edge_attr=edge_attr)
        x = self.gconv2(x, edge_index, edge_attr=edge_attr)
        x = self.gconv3(x, edge_index, edge_attr=edge_attr)
        x = global_add_pool(x, data.batch)
        x = self.fc(x)

        return x.squeeze(-1)

class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)

        return h

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

# %%