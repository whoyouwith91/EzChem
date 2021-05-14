import torch
from torch import nn
from torch_scatter import scatter_mean
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from helper import *
from layers import *

def get_model(config):
    name = config['model']
    if name == None:
        raise ValueError('Please specify one model you want to work on!')
    if name == '1-GNN':
        return GNN_1(config)
    if name == '1-2-GNN':
        return GNN_1_2(config)
    if name == '1-efgs-GNN':
        return GNN_1_EFGS(config)
    if name == '1-interaction-GNN':
        if config['dataset'] in ['solWithWater', 'solWithWater_calc/ALL', 'logpWithWater', 'logpWithWater_calc/ALL']:
            if config['interaction_simpler']:
                return GNN_1_WithWater_simpler(config)
            else:
                return GNN_1_WithWater(config)
    if name == '1-2-GNN_dropout':
        return knn_dropout
    if name == '1-2-GNN_swag':
        return knn_swag
    
class GNN(torch.nn.Module):
    """
    Basic GNN unit modoule.
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
        super(GNN, self).__init__()
        self.config = config
        self.num_layer = config['num_layer']
        self.emb_dim = config['emb_dim']
        self.drop_ratio = config['drop_ratio']
        self.JK = config['JK']
        self.gnn_type = config['gnn_type']
        self.aggregate = config['aggregate']
        self.num_atom_features = config['num_atom_features']
        self.gradCam = config['gradCam']
        self.act_fn = activation_func(config)

        if self.num_layer < 1:
            raise ValueError("Number of GNN layers must be no less than 1.")
        
        # embed nodes
        self.x_embedding1 = torch.nn.Sequential(torch.nn.Linear(self.num_atom_features, self.emb_dim), torch.nn.ReLU())

        # define graph conv layers
        self.gnns = torch.nn.ModuleList()
        for _ in range(self.num_layer):
            if self.gnn_type == "gin":
                self.gnns.append(GINConv(self.emb_dim, aggr=self.aggregate))
            elif self.gnn_type == "gcn":
                self.gnns.append(GCNConv(self.emb_dim, aggr=self.aggregate))
            elif self.gnn_type == 'nnconv':
                self.gnns.append(NNCon(self.emb_dim, aggregate=self.aggregate))
            elif self.gnn_type == "gat":
                self.gnns.append(GATCon(self.emb_dim, aggr=self.aggregate))
            elif self.gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(self.emb_dim, aggr=self.aggregate))
            elif self.gnn_type == 'pnaconv':
                self.gnns.append(PNAConv_rev(self.emb_dim, aggr=self.aggregate, deg=config['degree']))
               
        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(self.num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(self.emb_dim))
    
        self.gradients = None # for GradCAM     

    ## hook for the gradients of the activations GradCAM
    def activations_hook(self, grad):
        self.gradients = grad
    
    # method for the gradient extraction GradCAM
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return x

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
            if self.config['residual_connect']: # adding residual connection
                if layer > self.config['resLayer']: # need to change. currently default to 7 for 12 layers in total
                    residual = h_list[layer]
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            if self.config['residual_connect']:
                if layer > self.config['resLayer']:
                    h += residual
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
                self.last_conv = self.get_activations(h)
            else:
                h = self.act_fn(h)
                h = F.dropout(h, self.drop_ratio, training = self.training)
            h_list.append(h)
            if self.gradCam and layer == self.num_layer - 1:
                h.register_hook(self.activations_hook)
            
        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
            #print(node_representation.shape)
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)
            #print(node_representation.shape)

        return node_representation


class GNN_1(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1, self).__init__()
        self.dataset = config['dataset']
        self.num_layer = config['num_layer']
        self.NumOutLayers = config['NumOutLayers']
        self.JK = config['JK']
        self.emb_dim = config['emb_dim']
        self.num_tasks = config['num_tasks']
        self.graph_pooling = config['pooling']
        self.propertyLevel = config['propertyLevel']
        self.gnn_type = config['gnn_type']
        self.gradCam = config['gradCam']
        self.uncertainty = config['uncertainty']
        self.uncertaintyMode = config['uncertaintyMode']
        self.weight_regularizer = config['weight_regularizer']
        self.dropout_regularizer = config['dropout_regularizer']
        self.features = config['mol_features']

        self.gnn = GNN(config)
        self.outLayers = nn.ModuleList()
        if self.uncertainty:
            self.uncertaintyLayers = nn.ModuleList()
        #Different kind of graph pooling
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * self.emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(self.emb_dim, 1))
        elif self.graph_pooling == "set2set":
            set2set_iter = 2
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * self.emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(self.emb_dim, set2set_iter)
        elif self.graph_pooling == 'conv':
            self.pool = []
            self.pool.append(global_add_pool)
            self.pool.append(global_mean_pool)
            self.pool.append(global_max_pool)
            self.pool.append(GlobalAttention(gate_nn = torch.nn.Linear(self.emb_dim, 1)))
            self.pool.append(Set2Set(self.emb_dim, 2))
            self.convPool = nn.Conv1d(len(self.pool), 1, 2, stride=2)

        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level property predictions
        if self.graph_pooling[:-1][0] == "set2set": # set2set will double dimension
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat": # change readout layers input and output dimension
            L_in, L_out = self.mult * (self.num_layer + 1) * self.emb_dim, self.emb_dim
        elif self.graph_pooling == 'conv': # change readout layers input and output dimension
            L_in = self.mult * self.emb_dim / 2, self.mult * self.emb_dim / 2
        elif self.features:
            if self.dataset == 'sol_calc/ALL': # 19 selected top mol descriptors 
                L_in, L_out = self.mult * self.emb_dim + 19, self.emb_dim # 
        else: # change readout layers input and output dimension
            L_in, L_out = self.mult * self.emb_dim, self.emb_dim

        fc = nn.Sequential(Linear(L_in, L_out), nn.ReLU())
        self.outLayers.append(fc)
        for _ in range(self.NumOutLayers):
            L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), activation_func(config))
            self.outLayers.append(fc)
            if self.uncertainty: # for uncertainty 
                self.uncertaintyLayers.append(NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer)) 
        last_fc = nn.Linear(L_out, self.num_tasks)
        self.outLayers.append(last_fc)

        if self.uncertaintyMode == 'epistemic': 
            self.drop_mu = NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer) # working on last layer
        if self.uncertaintyMode == 'aleatoric': # 
            self.outLayers.append(last_fc)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr.long(), data.batch
        node_representation = self.gnn(x, edge_index, edge_attr) # node updating 

        # graph pooling 
        if self.propertyLevel == 'molecule': 
            if self.graph_pooling == 'conv':
                MolEmbed_list = []
                for p in self.pool:
                    MolEmbed_list.append(p(node_representation, batch))
                MolEmbed_stack = torch.stack(MolEmbed_list, 1).squeeze()
                MolEmbed = self.convPool(MolEmbed_stack).squeeze()
            else:
                MolEmbed = self.pool(node_representation, batch)
        if self.propertyLevel == 'atom':
            MolEmbed = node_representation 
        if self.features: # concatenating molecular features 
            MolEmbed = torch.cat((MolEmbed, data.features), -1)
        if not self.training and not self.gradCam: # for TSNE analysis
            return node_representation, MolEmbed

        # read-out layers
        if not self.uncertainty:
            for layer in self.outLayers:
                MolEmbed = layer(MolEmbed)
            if self.num_tasks > 1:
                return MolEmbed, None
            if self.propertyLevel == 'atom':
                return MolEmbed.view(-1,1), None
            else:
                return MolEmbed.view(-1), None
        # for uncertainty analysis
        else:
            for layer, drop in zip(self.outLayers[1:-1], self.uncertaintyLayers):
                x, _ = drop(MolEmbed, layer)
            if self.config['uncertainty'] == 'epistemic':
                mean, regularization[-1] = self.drop_mu(x, self.outLayers[-1])
                return mean.squeeze()
            if self.config['uncertainty'] == 'aleatoric':
                mean = self.outLayers[-2](x)
                log_var = self.outLayers[-1](x)
                return mean, log_var
        

class GNN_1_2(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1_2, self).__init__()
        self.num_layer = config['num_layer']
        self.NumOutLayers = config['NumOutLayers']
        self.JK = config['JK']
        self.emb_dim = config['emb_dim']
        self.num_i_2 = config['num_i_2']
        self.num_tasks = config['num_tasks']
        self.graph_pooling = config['pooling']

        self.gnn = GNN(config)
        self.convISO1 = GraphConv(self.emb_dim + self.num_i_2, self.emb_dim)
        self.convISO2 = GraphConv(self.emb_dim, self.emb_dim)
        self.outLayers = nn.ModuleList()
        #self.out_batch_norms = torch.nn.ModuleList()

        #Different kind of graph pooling
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * self.emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(self.emb_dim, 1))
        elif self.graph_pooling == "set2set":
            set2set_iter = 2
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * self.emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(self.emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if self.graph_pooling[:-1][0] == "set2set":
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
        #self.out_batch_norms.append(torch.nn.BatchNorm1d(L_out))
        for _ in range(self.NumOutLayers):
            L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), activation_func(config))
            self.outLayers.append(fc)
            #self.out_batch_norms.append(torch.nn.BatchNorm1d(L_out))
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
        x_2 = scatter_mean(x, batch_2, dim=0)   # to add stability to models
        
        MolEmbed = torch.cat([x_1, x_2], dim=1)
        if not self.training and not self.gradCam: # for TSNE analysis
            return node_representation, MolEmbed

        #MolEmbed = self.batch_norm(MolEmbed)
        for layer, out_norm in zip(self.outLayers[:-1], self.out_batch_norms):
             MolEmbed = layer(MolEmbed)
             MolEmbed = out_norm(MolEmbed)
        MolEmbed = self.outLayers[-1](MolEmbed)
        
        if self.num_tasks > 1:
            return MolEmbed, None
        else:
            return MolEmbed.view(-1), None

class GNN_1_EFGS(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1_EFGS, self).__init__()
        self.num_layer = config['num_layer']
        self.NumOutLayers = config['NumOutLayers']
        self.JK = config['JK']
        self.emb_dim = config['emb_dim']
        self.num_tasks = config['num_tasks']
        self.graph_pooling = config['pooling']
        self.efgs_vocab = config['efgs_lenth']

        self.gnn = GNN(config)
        # For EFGS
        self.convISO3 = GraphConv(self.emb_dim + self.efgs_vocab, self.emb_dim)
        self.convISO4 = GraphConv(self.emb_dim, self.emb_dim)


        self.outLayers = nn.ModuleList()

        #Different kind of graph pooling
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * self.emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(self.emb_dim, 1))
        elif self.graph_pooling == "set2set":
            set2set_iter = 2
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * self.emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(self.emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if self.graph_pooling[:-1][0] == "set2set":
            self.mult = 3
        else:
            self.mult = 2
        if self.JK == "concat":
            L_in, L_out = self.mult * (self.num_layer + 1) * self.emb_dim, self.emb_dim
        else:
            L_in, L_out = self.mult * self.emb_dim, self.emb_dim

        fc = nn.Sequential(Linear(L_in, L_out), nn.ReLU())
        self.outLayers.append(fc)
        for _ in range(self.NumOutLayers):
            L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), activation_func(config))
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


class GNN_1_WithWater(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1_WithWater, self).__init__()
        self.num_layer = config['num_layer']
        self.NumOutLayers = config['NumOutLayers']
        self.JK = config['JK']
        self.emb_dim = config['emb_dim']
        self.num_tasks = config['num_tasks']
        self.graph_pooling = config['pooling']
        self.dataset = config['dataset']
        self.gnn_solute = GNN(config)
        self.gnn_hydrated_solute = GNN(config)

        self.outLayers = nn.ModuleList()
        #Different kind of graph pooling
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * self.emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(self.emb_dim, 1))
        elif self.graph_pooling == "set2set":
            set2set_iter = 2
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * self.emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(2*self.emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if self.graph_pooling[:-1][0] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            L_in, L_out = self.mult * (self.num_layer + 1) * self.emb_dim, self.emb_dim
        else:
            L_in, L_out = 2 * self.mult * self.emb_dim, self.emb_dim

        fc = nn.Sequential(Linear(L_in, L_out), nn.ReLU())
        self.outLayers.append(fc)
        for _ in range(self.NumOutLayers):
            L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), activation_func(config))
            self.outLayers.append(fc)
        last_fc = nn.Linear(L_out, self.num_tasks)
        self.outLayers.append(last_fc)

    def from_pretrained(self, model_file_solute, model_file_solvent):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn_solute.load_state_dict(torch.load(model_file_solute))
        self.gnn_hydrated_solute.load_state_dict(torch.load(model_file_solvent))


    def forward(self, data):
        solute_x, solute_edge_index, solute_edge_attr, batch = data.x, data.edge_index, data.edge_attr.long(), data.batch
        hyd_solute_x, hyd_solute_edge_index, hyd_solute_edge_attr, hyd_solute_batch, mask = data.hyd_solute_x, data.hyd_solute_edge_index, data.hyd_solute_edge_attr.long(), \
            data.hyd_solute_batch, data.hyd_solute_mask

        solute_node_representation = self.gnn_solute(solute_x, solute_edge_index, solute_edge_attr) # 22 * 64
        hydrated_solute_node_representation = self.gnn_hydrated_solute(hyd_solute_x, hyd_solute_edge_index, hyd_solute_edge_attr) # 22 * 64

        solute_representation = self.pool(solute_node_representation, batch)
        if self.dataset in ['logpWithWater']:
            solute_representation = self.pool(solute_node_representation[mask>0], batch[mask>0])
        hydrated_solute_representation = self.pool(hydrated_solute_node_representation[mask>0], hyd_solute_batch[mask>0])

        final_representation = torch.cat([solute_representation, hydrated_solute_representation], dim=1)

        for layer in self.outLayers:
             final_representation = layer(final_representation)
        if self.num_tasks > 1:
            return final_representation, None
        else:
            return final_representation.view(-1), None

class GNN_1_WithWater_simpler(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1_WithWater_simpler, self).__init__()
        self.num_layer = config['num_layer']
        self.NumOutLayers = config['NumOutLayers']
        self.JK = config['JK']
        self.emb_dim = config['emb_dim']
        self.num_tasks = config['num_tasks']
        self.graph_pooling = config['pooling']
        self.dataset = config['dataset']
        self.gnn_solute = GNN(config)
        self.gnn_hydrated_solute = GNN(config)

        self.outLayers = nn.ModuleList()
        #Different kind of graph pooling
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * self.emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(self.emb_dim, 1))
        elif self.graph_pooling[:-1][0] == "set2set":
            set2set_iter = 2
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * self.emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(2*self.emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if self.graph_pooling[:-1][0] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.dataset in ['logpWithWater']:
            self.mult = self.mult * 2

        if self.JK == "concat":
            L_in, L_out = self.mult * (self.num_layer + 1) * self.emb_dim, self.emb_dim
        else:
            L_in, L_out = 2 * self.mult * self.emb_dim, self.emb_dim

        fc = nn.Sequential(Linear(L_in, L_out), nn.ReLU())
        self.outLayers.append(fc)
        for _ in range(self.NumOutLayers):
            L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), activation_func(config))
            self.outLayers.append(fc)
        last_fc = nn.Linear(L_out, self.num_tasks)
        self.outLayers.append(last_fc)

    def from_pretrained(self, model_file_solute, model_file_solvent):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn_solute.load_state_dict(torch.load(model_file_solute))
        self.gnn_hydrated_solute.load_state_dict(torch.load(model_file_solvent))


    def forward(self, data):
        solute_x, solute_edge_index, solute_edge_attr, batch = data.x, data.edge_index, data.edge_attr.long(), data.batch
        hyd_solute_x, hyd_solute_edge_index, hyd_solute_edge_attr, hyd_solute_batch, mask = data.hyd_solute_x, data.hyd_solute_edge_index, data.hyd_solute_edge_attr.long(), \
            data.hyd_solute_batch, data.hyd_solute_mask

        solute_node_representation = self.gnn_solute(solute_x, solute_edge_index, solute_edge_attr) # 22 * 64
        hydrated_solute_node_representation = self.gnn_hydrated_solute(hyd_solute_x, hyd_solute_edge_index, hyd_solute_edge_attr) # 22 * 64

        solute_representation = self.pool(solute_node_representation, batch)
        if self.dataset in ['logpWithWater']:
            solute_representation = self.pool(solute_node_representation[mask<1], batch[mask<1])
        hydrated_solute_representation = self.pool(hydrated_solute_node_representation[mask<1], hyd_solute_batch[mask<1])
        
        if self.dataset in ['logpWithWater']:
            final_representation = torch.cat([solute_representation, hydrated_solute_representation], dim=1)
        else:
            final_representation = hydrated_solute_representation

        for layer in self.outLayers:
             final_representation = layer(final_representation)
        if self.num_tasks > 1:
            return final_representation, None
        else:
            return final_representation.view(-1), None

class DNN(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_hidden, activation, bn):
        super().__init__()
        self.bn = bn
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        self.hiddens = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(n_hidden)])
        self.bn_hiddens = nn.ModuleList([nn.BatchNorm1d(hidden_size) for i in range(n_hidden)])
        self.bn = nn.BatchNorm1d(hidden_size)
        # output layer
        self.linear2 = nn.Linear(hidden_size, out_size)
        self.activation = activation
        
    def forward(self, X):
        out = self.linear1(X)
        for linear, bn in zip(self.hiddens, self.bn_hiddens):
            if self.bn:
                out = bn(out)
            out = self.activation(out)
            out = linear(out)
        if self.bn:
            out = self.bn(out)
        out = self.activation(out)
        out = self.linear2(out)
        return out

if __name__ == "__main__":
    pass
