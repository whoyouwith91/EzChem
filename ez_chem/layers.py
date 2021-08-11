from typing import Union, Tuple, Callable
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
from torch.nn import Parameter

import numpy as np
import torch
import torch.nn as nn
from torch.nn import GRU, LSTM
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, ReLU, LeakyReLU, ELU, Tanh
from torch_scatter import scatter_mean
from torch_scatter import scatter
from typing import Optional, List, Dict
from torch_geometric.typing import Adj, OptTensor

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import EdgePooling, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set, TopKPooling, SAGPooling
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from k_gnn import avg_pool, add_pool, max_pool
from helper import *

class NNDropout(nn.Module):
    def __init__(self, weight_regularizer, dropout_regularizer, init_min=0.1, init_max=0.1):
        super(NNDropout, self).__init__()
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max)) # 
    
    def _concrete_dropout(self, x, p):
        # This is like reparameterization tricks. 
        eps = 1e-7
        temp = 0.1 
        # Concrete distribution relaxation of the Bernoulli random variable
        unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
        
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        
        x  = torch.mul(x, random_tensor)
        x /= retain_prob
        
        return x

    def forward(self, data, layer):
        #p = torch.sigmoid(self.p_logit) # this is the drop out probablity, trainable. 
        p = torch.scalar_tensor(0.1)
        out = layer(self._concrete_dropout(data, p))
        x = out
        
        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        
        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)
        
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)
        
        input_dimensionality = x[0].numel() # Number of elements of first item in batch
        #print(input_dimensionality)
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality
        
        regularization = weights_regularizer + dropout_regularizer
        return out, regularization          

def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.
    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Also: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb

class LayerNorm(nn.Module):
    """
        Layer Normalization class
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def PoolingFN(config):
    #Different kind of graph pooling
    if config['pooling'] == "sum":
        pool = global_add_pool
    elif config['pooling'] == "mean":
        pool = global_mean_pool
    elif config['pooling'] == "max":
        pool = global_max_pool
    elif config['pooling'] == "attention":
        if config['JK'] == "concat":
            pool = GlobalAttention(gate_nn = torch.nn.Linear((config['num_layer'] + 1) * config['emb_dim'], 1))
        else:
            pool = GlobalAttention(gate_nn = torch.nn.Linear(config['emb_dim'], 1))
    elif config['pooling'] == "set2set":
        set2set_iter = 2 # 
        if config['JK'] == "concat":
            pool = Set2Set((config['num_layer'] + 1) * config['emb_dim'], set2set_iter)
        else:
            pool = Set2Set(config['emb_dim'], set2set_iter)
    elif config['pooling'] == 'conv':
        poolList = []
        poolList.append(global_add_pool)
        poolList.append(global_mean_pool)
        poolList.append(global_max_pool)
        poolList.append(GlobalAttention(gate_nn = torch.nn.Linear(self.emb_dim, 1)))
        poolList.append(Set2Set(config['emb_dim'], 2))
        pool = nn.Conv1d(len(poolList), 1, 2, stride=2)
    elif config['pooling'] == 'edge':
        pool = []
        pool.extend([EdgePooling(config['emb_dim']).cuda() for _ in range(config['num_layer'])])
    elif config['pooling'] == 'topk':
        pool = []
        pool.extend([TopKPooling(config['dimension']).cuda() for _ in range(config['num_layer'])])
    elif config['pooling'] == 'sag':
        pool = []
        pool.extend([SAGPooling(config['dimension']).cuda() for _ in range(config['num_layer'])])
    elif config['pooling'] == 'atomic':
        pool = global_add_pool
    else:
        raise ValueError("Invalid graph pooling type.")

    return pool

class ResidualLayer(nn.Module):
    """
    The residual layer defined in PhysNet
    """
    def __init__(self, module, dim, activation, drop_ratio=0., batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.drop_ratio = drop_ratio
        self.activation = activation
        self.module = module

        self.lin1 = nn.Linear(dim, dim)
        self.lin1.weight.data = semi_orthogonal_glorot_weights(F, F)
        self.lin1.bias.data.zero_()
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(dim, momentum=1.)

        self.lin2 = nn.Linear(dim, dim)
        self.lin2.weight.data = semi_orthogonal_glorot_weights(dim, dim)
        self.lin2.bias.data.zero_()
        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(dim, momentum=1.)

    def forward(self, module_type, x, edge_index=None, edge_attr=None):
        ### in order of Skip >> BN >> ReLU
        if module_type == 'linear': 
            x_res = self.module(x)
        elif module_type in ['gineconv', 'pnaconv', 'nnconv']:
            gnn_x = self.module(x, edge_index, edge_attr)
            x_res = gnn_x
        else:  # conv without using edge attributes
            gnn_x = self.module(x, edge_index)
            x_res = gnn_x
        
        if self.batch_norm:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.lin1(x)
        x = F.dropout(x, self.drop_ratio, training = self.training)  
        if self.batch_norm:
            x = self.bn2(x)
        x = self.activation(x)
        
        x = self.lin2(x)
        x = F.dropout(x, self.drop_ratio, training = self.training)
        return x + x_res