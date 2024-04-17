import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_scatter import scatter_add

import pandas as pd
from copy import copy
import logging
import os
from torch_geometric.nn import global_add_pool, SGConv
import math
log = logging.getLogger('torcheeg')


class CreatePyTorchDataset(Dataset):
    """
    This class aims to create a PyTorch dataset object. The data argument
    should be normalised in previous steps.
    """

    def __init__(self, data, labels, logic_gt):
        self.data = data
        self.labels = labels
        self.logic_gt = logic_gt

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, indx):
        if torch.is_tensor(indx):
            indx = indx.tolist()
        sample = torch.from_numpy(self.data[indx]).float()
        label = torch.as_tensor(self.labels[indx])
        logic_gt = torch.as_tensor(self.logic_gt[indx])
        return sample, label, logic_gt



def set_random_seed(seed=69):
    """Set seed for reproducibility in PyTorch and NumPy."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def score(y, y_pred,
          metrics=["chebyshev", "clark", "canberra", "kl_divergence", "cosine", "intersection"]):
    return tuple((eval(i)(y, y_pred) for i in metrics))


def normalize_A(A,lmax=2):
    A=F.relu(A)
    N=A.shape[0]
    A=A*(torch.ones(N,N).cuda()-torch.eye(N,N).cuda())
    A=A+A.T
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    L = torch.eye(N,N).cuda()-torch.matmul(torch.matmul(D, A), D)
    Lnorm=(2*L/lmax)-torch.eye(N, N).cuda()
    return Lnorm


def generate_cheby_adj(L, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(L.shape[-1]).cuda())
        elif i == 1:
            support.append(L)
        else:
            temp = torch.matmul(2*L,support[-1],)-support[-2]
            support.append(temp)
    return support


class GraphConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, bias=False):

        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels).cuda())
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).cuda())
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class Chebynet(nn.Module):
    def __init__(self, in_channels, K, out_channels):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()
        for i in range(K):
            self.gc1.append(GraphConvolution( in_channels,  out_channels))

    def forward(self, x,L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result

#######################################################

def get_edge_weight_from_electrode(edge_pos_value):
    num_nodes=len(edge_pos_value)
    edge_weight = np.zeros([num_nodes, num_nodes])
    # edge_pos_value = [edge_pos[key] for key in total_part]
    delta = 2  # Choosing delta=2 makes the proportion of non_negligible connections exactly 20%
    edge_index = [[], []]

    for i in range(num_nodes):
        for j in range(num_nodes):
            edge_index[0].append(i)
            edge_index[1].append(j)
            if i == j:
                edge_weight[i][j] = 1
            else:
                edge_weight[i][j] = np.sum(
                    [(edge_pos_value[i][k] - edge_pos_value[j][k])**2 for k in range(2)])
                if delta/edge_weight[i][j] > 1:
                    edge_weight[i][j] = math.exp(-edge_weight[i][j]/2)
                else:
                    edge_weight[i][j] = 0
    return edge_index, edge_weight


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes

def add_remaining_self_loops(edge_index,
                             edge_weight=None,
                             fill_value=1,
                             num_nodes=None):
    # reposition the diagonal values to the end
    '''
    edge_weight.shape : (num_nodes*num_nodes*batch_size,)
    ()
    '''
    # actually return num_nodes
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    mask = row != col

    inv_mask = ~mask  # diagonal positions
    # print("inv_mask", inv_mask)

    loop_weight = torch.full(
        (num_nodes,),
        fill_value,
        dtype=None if edge_weight is None else edge_weight.dtype,
        device=edge_index.device)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        remaining_edge_weight = edge_weight[inv_mask]

        if remaining_edge_weight.numel() > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight

        edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)
    loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    return edge_index, edge_weight


class NewSGConv(SGConv):
    def __init__(self, num_features, num_classes, K=1, cached=False,
                 bias=True):
        super(NewSGConv, self).__init__(num_features,
                                        num_classes, K=K, cached=cached, bias=bias)
        torch.nn.init.xavier_normal_(self.lin.weight)
    # allow negative edge weights
    @staticmethod
    # Note: here,num_nodes=self.num_nodes*batch_size
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index

        deg = scatter_add(torch.abs(edge_weight), row,
                          dim=0, dim_size=num_nodes)  # calculate degreematrix, i.e, D(stretched) in the paper.
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # calculate normalized adjacency matrix, i.e, S(stretched) in the paper.
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if not self.cached or self.cached_result is None:
            edge_index, norm = NewSGConv.norm(
                edge_index, x.size(0), edge_weight, dtype=x.dtype,)

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, norm=norm)
            self.cached_result = x

        return self.lin(self.cached_result)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

