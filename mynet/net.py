import torch
import numpy as np
import torch.nn.functional as F
import torch.nn
from torch.autograd import Variable
import scipy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import pickle
import os.path
from scipy import io
import sys
import torch.utils.data.dataset
use_cuda = torch.cuda.is_available()

from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (add_self_loops, sort_edge_index, remove_self_loops)
from mynet.braingraphconv import MyNNConv

import networkx as nx
from torch_sparse import coalesce

def createEdgeIdxAttr(x):
    
    num_nodes = x.shape[0]
    G = nx.from_numpy_array(x) 
    A = nx.to_scipy_sparse_array(G)
    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = x[adj.row[i], adj.col[i]]

    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes, num_nodes)
    
    return edge_index, edge_att.to(torch.float32)


class E2EBlock(torch.nn.Module):

    def __init__(self, in_planes, planes, bias=True):
        super(E2EBlock, self).__init__()
        self.d = 246
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias) 
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias) 

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)

        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2) 
    
class Network(torch.nn.Module):
    
    def __init__(self, k=8, dim0=8, dim1=4, dim2=2, dim3=32):
        super(Network, self).__init__()
        
        self.in_planes = 1 # num of channels
        self.d = 246  # number of nodes
        self.k = k # number of node clusters
        
        self.dim0 = dim0 # number of features per edge after E2EBlock
        self.dim1 = dim1 # number of features per node after E2N
        self.dim2 = dim2 # number of features per node after brainGNNconv.
        self.dim3 = dim3
        
        self.E2E = E2EBlock(1, self.dim0)
        self.E2N = torch.nn.Conv2d(self.dim0, self.dim1, (1, self.d))
        
        self.n1 = torch.nn.Sequential(torch.nn.Linear(self.d, self.k, bias=False), torch.nn.ReLU(), torch.nn.Linear(self.k, self.dim2 * self.dim1))
        self.brainGNNconv1 = MyNNConv(self.dim1, self.dim2, self.n1, normalize=False) # output shape: [num_nodes, dim2]
        #self.pool1 = TopKPooling(self.dim2, ratio=0.5, multiplier=1, nonlinearity=torch.sigmoid)

        self.fc1 = torch.nn.Linear(self.dim2, self.dim3)
        self.fc2 = torch.nn.Linear(self.dim3, 1)
        

    def forward(self, x, edge_index, batch, edge_attr,pos):
        
        
        out = self.E2E(x)
        out = F.leaky_relu(out, negative_slope=0.33)
        out = self.E2N(out)
        out = F.leaky_relu(out, negative_slope=0.33)
        
        out = out.squeeze()
        out = out.transpose(0, 1)

        out = self.brainGNNconv1(out, edge_index, edge_attr, pos)
        #out, edge_index, edge_attr, batch, perm, score1 = self.pool1(out, edge_index, edge_attr, batch)
        
        out = torch.mean(out, dim=0)

        out = F.relu(self.fc1(out)) 
        out = self.fc2(out)

        return out
    
    

    
