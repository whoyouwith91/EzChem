import torch
from torch import nn
from torch_scatter import scatter_mean
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import NNConv, GATConv, GraphConv
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from k_gnn import avg_pool, add_pool, max_pool
from helper import *

num_atom_features = 40 #including the extra mask tokens

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

def get_model(config):
    name = config['model']
    if name == None:
        raise ValueError('Please specify one model you want to work on!')
    if name == '1-GNN':
        return GNN_1
    if name == '1-2-GNN':
        return GNN_1_2
    if name == '1-efgs-GNN':
        return GNN_1_EFGS
    if name == '1-2-efgs-GNN':
        return GNN_1_2_EFGS
    if name == '1-interaction-GNN':
        if config['dataset'] in ['sol_exp', 'deepchem/freesol', 'sol_calc/ALL', 'solOct_calc/ALL']:
            return GNN_1_interaction
        if config['dataset'] in ['ws', 'deepchem/delaney']:
            return GNN_1_interaction_solubility
        if config['dataset'] in ['logp', 'deepchem/logp']:
            return GNN_1_interaction_logp
    if name == '1-2-GNN_dropout':
        return knn_dropout
    if name == '1-2-GNN_swag':
        return knn_swag
    
class GNN(torch.nn.Module):
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
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Sequential(torch.nn.Linear(num_atom_features, emb_dim), torch.nn.ReLU())

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for _ in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == 'nnconv':
                self.gnns.append(NNCon(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATCon(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))
            elif gnn_type == 'nnconvInter':
                self.gnns.append(NNConvInteraction(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")
        x = self.x_embedding1(x)
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)
            
        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation


class GNN_1(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_readout_layers, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_1, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)
        self.outLayers = nn.ModuleList()
        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1][0] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1][0] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            L_in, L_out = self.mult * (self.num_layer + 1) * self.emb_dim, self.emb_dim
        else:
            L_in, L_out = self.mult * self.emb_dim, self.emb_dim

        fc = nn.Sequential(Linear(L_in, L_out), nn.ReLU())
        self.outLayers.append(fc)
        for _ in range(num_readout_layers):
            L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), torch.nn.ReLU())
            self.outLayers.append(fc)
        last_fc = nn.Linear(L_out, self.num_tasks)
        self.outLayers.append(last_fc)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr.long(), data.batch
        node_representation = self.gnn(x, edge_index, edge_attr)
        MolEmbed = self.pool(node_representation, batch)

        for layer in self.outLayers:
             MolEmbed = layer(MolEmbed)
        if self.num_tasks > 1:
            return MolEmbed, None
        else:
            return MolEmbed.view(-1), None

class GNN_1_2(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_readout_layers, num_tasks, num_i_2, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_1_2, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)
        self.convISO1 = GraphConv(emb_dim + num_i_2, emb_dim)
        self.convISO2 = GraphConv(emb_dim, emb_dim)
        #self.gnnAll = nn.ModuleList()
        #self.gnnAll.append(self.gnn)
        #self.gnnAll.append(self.convISO1)
        #self.gnnAll.append(self.convISO2)

        self.outLayers = nn.ModuleList()

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1][0] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1][0] == "set2set":
            self.mult = 3
        else:
            self.mult = 2
        if self.JK == "concat":
            L_in, L_out = self.mult * (self.num_layer + 1) * self.emb_dim, self.emb_dim
        else:
            L_in, L_out = self.mult * self.emb_dim, self.emb_dim
        
        #self.batch_norm = torch.nn.BatchNorm1d(L_in)
        fc = nn.Sequential(Linear(L_in, L_out), nn.ReLU())
        self.outLayers.append(fc)
        for _ in range(num_readout_layers):
            L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), torch.nn.ReLU())
            self.outLayers.append(fc)
        last_fc = nn.Linear(L_out, self.num_tasks)
        self.outLayers.append(last_fc)

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))
        #self.convISO1.load_state_dict(torch.load(conv1File))
        #self.convISO2.load_state_dict(torch.load(conv2File))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, iso_type_2, edge_index, edge_index_2, assignment_index_2, edge_attr, batch, batch_2 = \
                data.x, data.iso_type_2, data.edge_index, data.edge_index_2, data.assignment_index_2, \
                    data.edge_attr.long(), data.batch, data.batch_2
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)
        x_1 = self.pool(node_representation, batch)

        x = avg_pool(node_representation, data.assignment_index_2)
        #data.x = torch.cat([data.x, data_iso], dim=1)
        x = torch.cat([x, iso_type_2], dim=1)
        x = F.relu(self.convISO1(x, edge_index_2))
        x = F.relu(self.convISO2(x, edge_index_2))
        x_2 = scatter_mean(x, batch_2, dim=0)
        
        MolEmbed = torch.cat([x_1, x_2], dim=1)
        #MolEmbed = self.batch_norm(MolEmbed)
        for layer in self.outLayers:
             MolEmbed = layer(MolEmbed)
        
        if self.num_tasks > 1:
            return MolEmbed, None
        else:
            return MolEmbed.view(-1), None

class GNN_1_EFGS(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_readout_layers, num_tasks, num_i_2, efgs_vocab, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_1_EFGS, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)
        # For EFGS
        self.convISO3 = GraphConv(emb_dim + efgs_vocab, emb_dim)
        self.convISO4 = GraphConv(emb_dim, emb_dim)


        self.outLayers = nn.ModuleList()

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1][0] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1][0] == "set2set":
            self.mult = 3
        else:
            self.mult = 2
        if self.JK == "concat":
            L_in, L_out = self.mult * (self.num_layer + 1) * self.emb_dim, self.emb_dim
        else:
            L_in, L_out = self.mult * self.emb_dim, self.emb_dim

        fc = nn.Sequential(Linear(L_in, L_out), nn.ReLU())
        self.outLayers.append(fc)
        for _ in range(num_readout_layers):
            L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), torch.nn.ReLU())
            self.outLayers.append(fc)
        last_fc = nn.Linear(L_out, self.num_tasks)
        self.outLayers.append(last_fc)

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, iso_type_2, iso_type_3, edge_index, edge_index_2, edge_index_3, assignment_index_2, \
                assignment_index_3, edge_attr, batch, batch_2, batch_3 = data.x, data.iso_type_2, data.iso_type_3, \
                    data.edge_index, data.edge_index_2, data.edge_index_3, data.assignment_index_2, data.assignment_index_3, \
                        data.edge_attr.long(), data.batch, data.batch_2, data.batch_3
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)
        x_1 = self.pool(node_representation, batch)

        x = avg_pool(node_representation, assignment_index_3)
        x = torch.cat([x, iso_type_3], dim=1)
        x = F.relu(self.convISO3(x, edge_index_3))
        x = F.relu(self.convISO4(x, edge_index_3))
        x_3 = scatter_mean(x, batch_3, dim=0) # 

        MolEmbed = torch.cat([x_1, x_3], dim=1)
        for layer in self.outLayers:
             MolEmbed = layer(MolEmbed)

        if self.num_tasks > 1:
            return MolEmbed, None
        else:
            return MolEmbed.view(-1), None

class GNN_1_2_EFGS(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_readout_layers, num_tasks, num_i_2, efgs_vocab, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_1_2_EFGS, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)
        self.convISO1 = GraphConv(emb_dim + num_i_2, emb_dim)
        self.convISO2 = GraphConv(emb_dim, emb_dim)

        # For EFGS
        self.convISO3 = GraphConv(emb_dim + efgs_vocab, emb_dim)
        self.convISO4 = GraphConv(emb_dim, emb_dim)

        self.outLayers = nn.ModuleList()

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1][0] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1][0] == "set2set":
            self.mult = 6
        else:
            self.mult = 3
        if self.JK == "concat":
            L_in, L_out = self.mult * (self.num_layer + 1) * self.emb_dim, self.emb_dim
        else:
            L_in, L_out = self.mult * self.emb_dim, self.emb_dim

        fc = nn.Sequential(Linear(L_in, L_out), nn.ReLU())
        self.outLayers.append(fc)
        for _ in range(num_readout_layers):
            L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), torch.nn.ReLU())
            self.outLayers.append(fc)
        last_fc = nn.Linear(L_out, self.num_tasks)
        self.outLayers.append(last_fc)

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, iso_type_2, iso_type_3, edge_index, edge_index_2, edge_index_3, assignment_index_2, \
                assignment_index_3, edge_attr, batch, batch_2, batch_3 = data.x, data.iso_type_2, data.iso_type_3, \
                    data.edge_index, data.edge_index_2, data.edge_index_3, data.assignment_index_2, data.assignment_index_3, \
                        data.edge_attr.long(), data.batch, data.batch_2, data.batch_3
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)
        x_1 = self.pool(node_representation, batch)

        x = avg_pool(node_representation, assignment_index_2)
        x = torch.cat([x, iso_type_2], dim=1)
        x = F.relu(self.convISO1(x, edge_index_2))
        x = F.relu(self.convISO2(x, edge_index_2))
        x_2 = scatter_mean(x, batch_2, dim=0)

        x = avg_pool(node_representation, assignment_index_3)
        x = torch.cat([x, iso_type_3], dim=1)
        x = F.relu(self.convISO3(x, edge_index_3))
        x = F.relu(self.convISO4(x, edge_index_3))
        x_3 = scatter_mean(x, batch_3, dim=0)

        MolEmbed = torch.cat([x_1, x_2, x_3], dim=1)
        for layer in self.outLayers:
             MolEmbed = layer(MolEmbed)

        if self.num_tasks > 1:
            return MolEmbed, None
        else:
            return MolEmbed.view(-1), None

class GNN_1_interaction(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_readout_layers, num_tasks, solvent, interaction, soluteSelf=False, JK = "last", drop_ratio = 0, \
                 graph_pooling = "sum", gnn_type = "gin"):
        super(GNN_1_interaction, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.solvent = solvent
        self.interaction = interaction

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn_solute = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)
        if solvent == 'water':
            self.gnn_solvent = nn.Sequential(nn.Linear(num_atom_features, emb_dim),torch.nn.ReLU(), \
                                             nn.Linear(emb_dim, emb_dim))
        else:
            self.gnn_solvent = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        self.imap = nn.Linear(2*emb_dim, 1)
        self.outLayers = nn.ModuleList()
        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1][0] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(2*emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1][0] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            L_in, L_out = self.mult * (self.num_layer + 1) * self.emb_dim, self.emb_dim
        else:
            L_in, L_out = 4 * self.mult * self.emb_dim, self.emb_dim

        fc = nn.Sequential(Linear(L_in, L_out), nn.ReLU())
        self.outLayers.append(fc)
        for _ in range(num_readout_layers):
            L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), torch.nn.ReLU())
            self.outLayers.append(fc)
        last_fc = nn.Linear(L_out, self.num_tasks)
        self.outLayers.append(last_fc)

    def from_pretrained(self, model_file_solute, model_file_solvent):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn_solute.load_state_dict(torch.load(model_file_solute))
        self.gnn_solvent.load_state_dict(torch.load(model_file_solvent))


    def forward(self, data):
        solute_batch, solute_x, solute_edge_index, solute_edge_attr, solute_length_matrix, solvent_batch, solvent_x, solvent_length_matrix = \
                data.batch, data.x, data.edge_index, data.edge_attr.long(), data.solute_length_matrix, \
                data.solvent_batch, data.solvent_x, data.solvent_length_matrix
        if self.solvent == 'octanol':
            solvent_edge_index, solvent_edge_attr = data.solvent_edge_index, data.solvent_edge_attr.long()

        solute_representation = self.gnn_solute(solute_x, solute_edge_index, solute_edge_attr) # 22 * 64
        if self.solvent == 'water':
            solvent_representation = self.gnn_solvent(solvent_x)
        if self.solvent == 'octanol':
            solvent_representation = self.gnn_solvent(solvent_x, solvent_edge_index, solvent_edge_attr) # 27 * 64
        #MolEmbed = self.pool(node_representation, batch)

        # Interaction
        len_map = torch.mm(solute_length_matrix.t(), solvent_length_matrix)  # interaction map to control which solvent mols  22*27
        #corresponds to which solute mol
        if 'dot' not in self.interaction:
            X1 = solute_representation.unsqueeze(0) # 1*22*64
            Y1 = solvent_representation.unsqueeze(1) # 27*1*64
            X2 = X1.repeat(solvent_representation.shape[0], 1, 1) # 27*22*64
            Y2 = Y1.repeat(1, solute_representation.shape[0], 1) # 27*22*64
            Z = torch.cat([X2, Y2], -1) # 27*22*128

            if self.interaction == 'general':
                interaction_map = self.imap(Z).squeeze(2) # 27*22
            if self.interaction == 'tanh-general':
                interaction_map = torch.tanh(self.imap(Z)).squeeze(2)

            interaction_map = torch.mul(len_map.float(), interaction_map.t()) # 22*27
            ret_interaction_map = torch.clone(interaction_map)

        elif 'dot' in self.interaction:
            interaction_map = torch.mm(solute_representation, solvent_representation.t()) # interaction coefficient 22 * 27
            if 'scaled' in self.interaction:
                interaction_map = interaction_map / (np.sqrt(self.emb_dim))

            ret_interaction_map = torch.clone(interaction_map)
            ret_interaction_map = torch.mul(len_map.float(), ret_interaction_map) # 22 * 27
            interaction_map = torch.tanh(interaction_map) # 22*27
            interaction_map = torch.mul(len_map.float(), interaction_map) # 22 * 27

        solvent_prime = torch.mm(interaction_map.t(), solute_representation) # 27 * 64
        solute_prime = torch.mm(interaction_map, solvent_representation) # 22 * 64

        # Prediction
        solute_representation = torch.cat((solute_representation, solute_prime), dim=1) # 22 * 128
        solvent_representation = torch.cat((solvent_representation, solvent_prime), dim=1) # 27 * 128
        #print(solute_representation.shape)
        solute_representation = self.pool(solute_representation, solute_batch) # bs * 128
        solvent_representation = self.pool(solvent_representation, solvent_batch) # bs * 128
        #print(solute_representation.shape)
        final_representation = torch.cat((solute_representation, solvent_representation), 1) # bs * 256

        for layer in self.outLayers:
             final_representation = layer(final_representation)
        if self.num_tasks > 1:
            return final_representation, ret_interaction_map
        else:
            return final_representation.view(-1), ret_interaction_map

class GNN_1_interaction(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_readout_layers, num_tasks, solvent, interaction, soluteSelf=False, JK = "last", drop_ratio = 0, \
                 graph_pooling = "sum", gnn_type = "gin"):
        super(GNN_1_interaction, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.solvent = solvent
        self.interaction = interaction

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn_solute = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)
        if solvent == 'water':
            self.gnn_solvent = nn.Sequential(nn.Linear(num_atom_features, emb_dim),torch.nn.ReLU(), \
                                             nn.Linear(emb_dim, emb_dim))
        if solvent == 'octanol':
            self.gnn_solvent = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)
        if solvent == 'watOct':
            self.gnn_wat = nn.Sequential(nn.Linear(num_atom_features, emb_dim),torch.nn.ReLU(), \
                                             nn.Linear(emb_dim, emb_dim))
            self.gnn_oct = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        self.imap = nn.Linear(2*emb_dim, 1)
        self.outLayers = nn.ModuleList()
        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1][0] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(2*emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1][0] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            L_in, L_out = self.mult * (self.num_layer + 1) * self.emb_dim, self.emb_dim
        else:
            if self.solvent in ['water', 'octanol']:
                L_in, L_out = 2 * self.mult * self.emb_dim, self.emb_dim
            if self.solvent == 'watOct':
                L_in, L_out = 3 * self.mult * self.emb_dim, self.emb_dim

        fc = nn.Sequential(Linear(L_in, L_out), nn.ReLU())
        self.outLayers.append(fc)
        for _ in range(num_readout_layers):
            L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), torch.nn.ReLU())
            self.outLayers.append(fc)
        last_fc = nn.Linear(L_out, self.num_tasks)
        self.outLayers.append(last_fc)

    def from_pretrained(self, model_file_solute, model_file_solvent):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn_solute.load_state_dict(torch.load(model_file_solute))
        self.gnn_solvent.load_state_dict(torch.load(model_file_solvent))


    def forward(self, data):
        if self.config['dataset'] in ['ws', 'sol_exp', 'deepchem/freesol', 'deepchem/delaney']:
            solute_batch, solute_x, solute_edge_index, solute_edge_attr, solute_length_matrix, solvent_batch, solvent_x, solvent_length_matrix = \
                data.batch, data.x, data.edge_index, data.edge_attr.long(), data.solute_length_matrix, \
                data.solvent_batch, data.solvent_x, data.solvent_length_matrix
        
        if self.config['dataset'] in ['logp', 'deepchem/logp']:
            solute_batch, solute_x, solute_edge_index, solute_edge_attr, solute_length_matrix, wat_batch, wat_x, wat_length_matrix, \
            oct_batch, oct_x, oct_edge_index, oct_edge_attr, oct_length_matrix = \
                data.batch, data.x, data.edge_index, data.edge_attr.long(), data.solute_length_matrix, \
                data.wat_batch, data.wat_x, data.wat_length_matrix, data.oct_batch, data.oct_x, data.oct_edge_index, data.oct_edge_attr.long(), \
                data.oct_length_matrix

        if self.solvent == 'octanol':
            solvent_edge_index, solvent_edge_attr = data.solvent_edge_index, data.solvent_edge_attr.long()

        solute_representation = self.gnn_solute(solute_x, solute_edge_index, solute_edge_attr) # 22 * 64
        if self.solvent == 'water':
            solvent_representation = self.gnn_solvent(solvent_x)
        if self.solvent == 'octanol':
            solvent_representation = self.gnn_solvent(solvent_x, solvent_edge_index, solvent_edge_attr) # 27 * 64
        if self.solvent == 'watOct':
            wat_representation = self.gnn_wat(wat_x)
            oct_representation = self.gnn_oct(oct_x, oct_edge_index, oct_edge_attr)

        #MolEmbed = self.pool(node_representation, batch)

        solute_representation = self.pool(solute_representation, solute_batch) # bs * 128
        if self.solvent == 'water':
            solvent_representation = self.gnn_solvent(solvent_x)
        if self.solvent == 'octanol':
            solvent_representation = self.gnn_solvent(solvent_x, solvent_edge_index, solvent_edge_attr)
        if self.solvent == 'watOct':
            wat_representation = self.pool(wat_representation, wat_batch) # bs * 128
            oct_representation = self.pool(oct_representation, oct_batch) # bs * 128
        
        if self.solvent in ['water', 'octanol']:
            final_representation = torch.cat((solute_representation, solvent_representation), 1) # bs * 256
        if self.solvent == 'watOct':
            final_representation = torch.cat((solute_representation, wat_representation, oct_representation), 1) # bs * 256

        for layer in self.outLayers:
             final_representation = layer(final_representation)
        if self.num_tasks > 1:
            return final_representation, None
        else:
            return final_representation.view(-1), None

class GNN_1_interaction_solubility(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_readout_layers, num_tasks, solvent, interaction, soluteSelf=False, JK = "last", drop_ratio = 0, \
                 graph_pooling = "sum", gnn_type = "gin"):
        super(GNN_1_interaction_solubility, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.solvent = solvent
        self.soluteSelf = soluteSelf
        self.interaction = interaction

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn_solute = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)
        if self.soluteSelf:
            self.gnn_solute_1 = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        if solvent == 'water':
            self.gnn_solvent = nn.Sequential(nn.Linear(num_atom_features, emb_dim),torch.nn.ReLU(), \
                                             nn.Linear(emb_dim, emb_dim))
        else:
            self.gnn_solvent = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        self.imap = nn.Linear(2*emb_dim, 1)
        self.outLayers = nn.ModuleList()
        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1][0] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(2*emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1][0] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            L_in, L_out = self.mult * (self.num_layer + 1) * self.emb_dim, self.emb_dim
        else:
            L_in, L_out = 6 * self.mult * self.emb_dim, self.emb_dim

        fc = nn.Sequential(Linear(L_in, L_out), nn.ReLU())
        self.outLayers.append(fc)
        for _ in range(num_readout_layers):
            L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), torch.nn.ReLU())
            self.outLayers.append(fc)
        last_fc = nn.Linear(L_out, self.num_tasks)
        self.outLayers.append(last_fc)

    def from_pretrained(self, model_file_solute, model_file_solvent):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn_solute.load_state_dict(torch.load(model_file_solute))
        self.gnn_solvent.load_state_dict(torch.load(model_file_solvent))


    def forward(self, data):
        solute_batch, solute_x, solute_edge_index, solute_edge_attr, solute_length_matrix, solvent_batch, solvent_x, solvent_length_matrix = \
                data.batch, data.x, data.edge_index, data.edge_attr.long(), data.solute_length_matrix, \
                data.solvent_batch, data.solvent_x, data.solvent_length_matrix
        if self.solvent == 'octanol':
            solvent_edge_index, solvent_edge_attr = data.solvent_edge_index, data.solvent_edge_attr.long()

        solute_representation = self.gnn_solute(solute_x, solute_edge_index, solute_edge_attr) # 22 * 64
        if self.soluteSelf:
            solute_representation_1 = self.gnn_solute_1(solute_x, solute_edge_index, solute_edge_attr)
        if self.solvent == 'water':
            solvent_representation = self.gnn_solvent(solvent_x)
        if self.solvent == 'octanol':
            solvent_representation = self.gnn_solvent(solvent_x, solvent_edge_index, solvent_edge_attr) # 27 * 64
        #MolEmbed = self.pool(node_representation, batch)

        # Interaction
        len_map = torch.mm(solute_length_matrix.t(), solvent_length_matrix)  # interaction map to control which solvent mols  22*27
        if self.soluteSelf:
            len_map_solute = torch.mm(solute_length_matrix.t(), solute_length_matrix)  # interaction map to control which solvent mols  22*27
        #corresponds to which solute mol
        if 'dot' not in self.interaction:
            X1 = solute_representation.unsqueeze(0) # 1*22*64
            Y1 = solvent_representation.unsqueeze(1) # 27*1*64
            X2 = X1.repeat(solvent_representation.shape[0], 1, 1) # 27*22*64
            Y2 = Y1.repeat(1, solute_representation.shape[0], 1) # 27*22*64
            Z = torch.cat([X2, Y2], -1) # 27*22*128

            if self.interaction == 'general':
                interaction_map = self.imap(Z).squeeze(2) # 27*22
            if self.interaction == 'tanh-general':
                interaction_map = torch.tanh(self.imap(Z)).squeeze(2)

            interaction_map = torch.mul(len_map.float(), interaction_map.t()) # 22*27
            ret_interaction_map = torch.clone(interaction_map)

        elif 'dot' in self.interaction:
            interaction_map = torch.mm(solute_representation, solvent_representation.t()) # interaction coefficient 22 * 27
            if 'scaled' in self.interaction:
                interaction_map = interaction_map / (np.sqrt(self.emb_dim))

            ret_interaction_map = torch.clone(interaction_map)
            ret_interaction_map = torch.mul(len_map.float(), ret_interaction_map) # 22 * 27
            interaction_map = torch.tanh(interaction_map) # 22*27
            interaction_map = torch.mul(len_map.float(), interaction_map) # 22 * 27

            if self.soluteSelf:
                interaction_map_solute = torch.mm(solute_representation, solute_representation_1.t()) # interaction coefficient 22 * 22
                if 'scaled' in self.interaction:
                    interaction_map_solute = interaction_map_solute / (np.sqrt(self.emb_dim))

                #ret_interaction_map = torch.clone(interaction_map)
                #ret_interaction_map = torch.mul(len_map_solute.float(), ret_interaction_map) # 22 * 27
                interaction_map_solute = torch.tanh(interaction_map_solute) # 22*22
                interaction_map_solute = torch.mul(len_map_solute.float(), interaction_map_solute) # 22 * 22


        solvent_prime = torch.mm(interaction_map.t(), solute_representation) # 27 * 64
        solute_prime = torch.mm(interaction_map, solvent_representation) # 22 * 64
        if self.soluteSelf:
            solute_solute_prime = torch.mm(interaction_map_solute, solute_representation_1) # 22 * 64

        # Prediction
        solute_representation_solvent = torch.cat((solute_representation, solute_prime), dim=1) # 22 * 128
        solvent_representation = torch.cat((solvent_representation, solvent_prime), dim=1) # 27 * 128
        if self.soluteSelf:
            solute_solute_representation = torch.cat((solute_representation, solute_solute_prime), dim=1) # 22*128
        #print(solute_representation.shape)
        solute_representation = self.pool(solute_representation_solvent, solute_batch) # bs * 128
        solvent_representation = self.pool(solvent_representation, solvent_batch) # bs * 128
        if self.soluteSelf:
            solute_solute_representation = scatter_mean(solute_solute_representation, solute_batch, dim=0) # bs * 128
        #print(solute_representation.shape)
        if not self.soluteSelf:
            final_representation = torch.cat((solute_representation, solvent_representation), 1) # bs * 256
        if self.soluteSelf:
            final_representation = torch.cat((solute_solute_representation, solute_representation, solvent_representation), 1) # bs * 256
            #print(final_representation.shape)

        for layer in self.outLayers:
             final_representation = layer(final_representation)
        if self.num_tasks > 1:
            return final_representation, ret_interaction_map
        else:
            return final_representation.view(-1), ret_interaction_map

class GNN_1_interaction_logp(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_readout_layers, num_tasks, solvent, interaction, soluteSelf=False, JK = "last", drop_ratio = 0, \
                 graph_pooling = "sum", gnn_type = "gin"):
        super(GNN_1_interaction_logp, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.solvent = solvent
        self.interaction = interaction

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn_solute = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)
        if solvent == 'watOct':
            self.gnn_wat = nn.Sequential(nn.Linear(num_atom_features, emb_dim),torch.nn.ReLU(), \
                                             nn.Linear(emb_dim, emb_dim))
            self.gnn_oct = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        self.imap = nn.Linear(2*emb_dim, 1)
        self.outLayers = nn.ModuleList()
        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1][0] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(2*emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1][0] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            L_in, L_out = self.mult * (self.num_layer + 1) * self.emb_dim, self.emb_dim
        else:
            L_in, L_out = 8 * self.mult * self.emb_dim, self.emb_dim

        fc = nn.Sequential(Linear(L_in, L_out), nn.ReLU())
        self.outLayers.append(fc)
        for _ in range(num_readout_layers):
            L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), torch.nn.ReLU())
            self.outLayers.append(fc)
        last_fc = nn.Linear(L_out, self.num_tasks)
        self.outLayers.append(last_fc)

    def from_pretrained(self, model_file_solute, model_file_solvent):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn_solute.load_state_dict(torch.load(model_file_solute))
        self.gnn_solvent.load_state_dict(torch.load(model_file_solvent))


    def forward(self, data):
        solute_batch, solute_x, solute_edge_index, solute_edge_attr, solute_length_matrix, wat_batch, wat_x, wat_length_matrix, \
            oct_batch, oct_x, oct_edge_index, oct_edge_attr, oct_length_matrix = \
                data.batch, data.x, data.edge_index, data.edge_attr.long(), data.solute_length_matrix, \
                data.wat_batch, data.wat_x, data.wat_length_matrix, data.oct_batch, data.oct_x, data.oct_edge_index, data.oct_edge_attr.long(), \
                data.oct_length_matrix

        solute_representation = self.gnn_solute(solute_x, solute_edge_index, solute_edge_attr) # 22 * 64
        if self.solvent == 'watOct':
            wat_representation = self.gnn_wat(wat_x)
            oct_representation = self.gnn_oct(oct_x, oct_edge_index, oct_edge_attr) # 27 * 64
        #MolEmbed = self.pool(node_representation, batch)

        # Interaction
        len_map_wat = torch.mm(solute_length_matrix.t(), wat_length_matrix)  # interaction map to control which solvent mols  22*27
        len_map_oct = torch.mm(solute_length_matrix.t(), oct_length_matrix)

        if 'dot' not in self.interaction:
            X1 = solute_representation.unsqueeze(0) # 1*22*64
            Y1 = solvent_representation.unsqueeze(1) # 27*1*64
            X2 = X1.repeat(solvent_representation.shape[0], 1, 1) # 27*22*64
            Y2 = Y1.repeat(1, solute_representation.shape[0], 1) # 27*22*64
            Z = torch.cat([X2, Y2], -1) # 27*22*128

            if self.interaction == 'general':
                interaction_map = self.imap(Z).squeeze(2) # 27*22
            if self.interaction == 'tanh-general':
                interaction_map = torch.tanh(self.imap(Z)).squeeze(2)

            interaction_map = torch.mul(len_map.float(), interaction_map.t()) # 22*27
            ret_interaction_map = torch.clone(interaction_map)

        elif 'dot' in self.interaction:
            interaction_map_wat = torch.mm(solute_representation, wat_representation.t()) # interaction coefficient 22 * 27
            interaction_map_oct = torch.mm(solute_representation, oct_representation.t()) # interaction coefficient 22 * 27
            if 'scaled' in self.interaction:
                interaction_map_wat = interaction_map_wat / (np.sqrt(self.emb_dim))
                interaction_map_oct = interaction_map_oct / (np.sqrt(self.emb_dim))

            ret_interaction_map_wat = torch.clone(interaction_map_wat)
            ret_interaction_map_oct = torch.clone(interaction_map_oct)
            ret_interaction_map_wat = torch.mul(len_map_wat.float(), ret_interaction_map_wat) # 22 * 27
            ret_interaction_map_oct = torch.mul(len_map_oct.float(), ret_interaction_map_oct) # 22 * 27
            interaction_map_wat = torch.tanh(interaction_map_wat) # 22*27
            interaction_map_oct = torch.tanh(interaction_map_oct) # 22*27
            interaction_map_wat = torch.mul(len_map_wat.float(), interaction_map_wat) # 22 * 27
            interaction_map_oct = torch.mul(len_map_oct.float(), interaction_map_oct) # 22 * 27

        wat_prime = torch.mm(interaction_map_wat.t(), solute_representation) # 27 * 64
        oct_prime = torch.mm(interaction_map_oct.t(), solute_representation) # 27 * 64
        solute_prime_wat = torch.mm(interaction_map_wat, wat_representation) # 22 * 64
        solute_prime_oct = torch.mm(interaction_map_oct, oct_representation) # 22 * 64

        # Prediction
        solute_representation_wat = torch.cat((solute_representation, solute_prime_wat), dim=1) # 22 * 128
        solute_representation_oct = torch.cat((solute_representation, solute_prime_oct), dim=1) # 22 * 128
        wat_representation = torch.cat((wat_representation, wat_prime), dim=1) # 27 * 128
        oct_representation = torch.cat((oct_representation, oct_prime), dim=1) # 27 * 128
        #print(solute_representation.shape)
        solute_representation_wat = self.pool(solute_representation_wat, solute_batch) # bs * 128
        solute_representation_oct = self.pool(solute_representation_oct, solute_batch) # bs * 128
        wat_representation = self.pool(wat_representation, wat_batch) # bs * 128
        oct_representation = self.pool(oct_representation, oct_batch) # bs * 128
        
        final_representation = torch.cat((solute_representation_wat, solute_representation_oct, wat_representation, oct_representation), 1) # bs * 512
        for layer in self.outLayers:
             final_representation = layer(final_representation)
        if self.num_tasks > 1:
            return final_representation, ret_interaction_map
        else:
            return final_representation.view(-1), (ret_interaction_map_wat, ret_interaction_map_oct) 


if __name__ == "__main__":
    pass
