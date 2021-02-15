import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, ELU, Tanh, SELU
from torch_scatter import scatter_mean, scatter_add, scatter_max
from k_gnn import GraphConv, avg_pool, add_pool, max_pool
import torch_geometric.transforms as T
from torch_geometric.nn import NNConv, GraphConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from helper import *

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 
num_i_2 = 200

class gnnBase(torch.nn.Module):
    def __init__(self, config):
        super(gnnBase, self).__init__()
        self.fn = ReLU()
        M_in, B_in, M_out = config['num_features'], config['num_bond_features'], int(config['dimension']/2)    
        
        self.conv = nn.ModuleList()
        for _ in range(config['num_layer']):
            ll = Sequential(Linear(B_in, 128), self.fn, Linear(128, M_in * M_out))
            _conv = NNConv(M_in, M_out, ll)
            self.conv.append(_conv)
            M_in, M_out = M_out, config['dimension']

    def forward(self, x, edge_index, edge_attr):
        x = x.float() 
        edge_attr = edge_attr.float()
        for layer in self.conv:
            x = self.fn(layer(x, edge_index, edge_attr)) # update atom embeddings
        out = x
        return out

class GNN_1(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1, self).__init__()
        self.fn = ReLU()
        self.pooling = pooling(config)
        self.config = config
        if config['baseModel'] == 'GCN':
           self.gnn_base = gnnBase(config)
        if config['baseModel'] == 'GIN':
           self.gnn_base = GNNPretrain(config)

        self.out1 = nn.ModuleList()
        L_in, L_out = config['dimension'], int(config['dimension'])
        fc = nn.Sequential(Linear(L_in, L_out), self.fn)
        self.out1.append(fc)
        for _ in range(config['NumOutLayers']-2):
            L_in, L_out = self.out1[-1][0].out_features, int(self.out1[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), self.fn)
            self.out1.append(fc)
        if config['taskType'] == 'multi':
            last_fc = nn.Sequential(nn.Linear(L_out, config['numTask']))
            self.out1.append(last_fc)
        if config['taskType'] == 'single':
            last_fc = nn.Sequential(nn.Linear(L_out, 1))
            self.out1.append(last_fc)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn_base.load_state_dict(torch.load(model_file))

    def forward(self, data):
        node_representation = self.gnn_base(data.x, data.edge_index, data.edge_attr)
        x_1 = self.pooling(node_representation, data.batch, dim=0)

        MolEmbed = x_1
        out1 = MolEmbed
        for layer in self.out1:
            out1 = layer(out1)

        if self.config['taskType'] == 'multi':
           return out1, None
        else:
           return out1.view(-1), None

class GNN_2(torch.nn.Module):
    def __init__(self, config):
        super(GNN_2, self).__init__()
        self.fn = ReLU()
        self.pooling = pooling(config)
        if config['baseModel'] == 'GCN': 
           self.gnn_base = gnnBase(config)
        if config['baseModel'] == 'GIN':
           self.gnn_base = GNNPretrain(config)

        self.conv4 = GraphConv(config['dimension'] + config['num_i_2'], config['dimension'])
        self.conv5 = GraphConv(config['dimension'], config['dimension'])
        #self.ll = nn.Linear(emb_dim*2, emb_dim)
        self.out1 = nn.ModuleList()
        L_in, L_out = config['dimension']*2, int(config['dimension'])
        fc = nn.Sequential(Linear(L_in, L_out), self.fn)
        self.out1.append(fc)
        for _ in range(config['NumOutLayers']-2):
            L_in, L_out = self.out1[-1][0].out_features, int(self.out1[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), self.fn)
            self.out1.append(fc)
        last_fc = nn.Sequential(nn.Linear(L_out, 1))
        self.out1.append(last_fc)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn_base.load_state_dict(torch.load(model_file))

    def forward(self, data):
        node_representation = self.gnn_base(data.x, data.edge_index, data.edge_attr)
        x_1 = self.pooling(node_representation, data.batch, dim=0)

        data.x = avg_pool(node_representation, data.assignment_index_2)
        #data.x = torch.cat([data.x, data_iso], dim=1)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)
        data.x = self.fn(self.conv4(data.x, data.edge_index_2))
        data.x = self.fn(self.conv5(data.x, data.edge_index_2))
        x_2 = self.pooling(data.x, data.batch_2, dim=0)

        MolEmbed = torch.cat([x_1, x_2], dim=1)  # add x_0
        out1 = MolEmbed
        for layer in self.out1:
            out1 = layer(out1)
            
        return out1.view(-1), None

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        
    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GNNPretrain(torch.nn.Module):
    """
    
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, config):
        super(GNNPretrain, self).__init__()
        self.config = config
        #self.num_layer = num_layer
        #self.drop_ratio = drop_ratio
        #self.JK = JK
        self.fn = activation_func(config)
        ###List of MLPs
        self.linear = nn.Linear(config['num_features'], config['dimension'])
        self.gnns = torch.nn.ModuleList()
        for _ in range(config['num_layer']):
            self.gnns.append(GINConv(config['dimension'], aggr = "add"))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(config['num_layer']):
            self.batch_norms.append(torch.nn.BatchNorm1d(config['dimension']))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0].float(), argv[1], argv[2].long()
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")
        
        x = self.fn(self.linear(x))
        h_list = [x]
        for layer in range(self.config['num_layer']):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            #if layer == self.config['num_layer'] - 1:
            #    #remove relu for the last layer
            #    h = F.dropout(h, self.drop_ratio, training = self.training)
            #else:
            #    h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.config['JK'] == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.config['JK'] == "last":
            node_representation = h_list[-1]
        elif self.config['JK'] == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.config['JK'] == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation
