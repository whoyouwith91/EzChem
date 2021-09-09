import torch
from torch import nn
from torch_geometric.nn.glob.glob import global_add_pool
from torch_scatter import scatter_mean
from torch_geometric.utils import add_self_loops, degree, softmax
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from helper import *
from layers import *
from gnns import *
from PhysDimeNet import PhysDimeNet
from torch_geometric.nn.norm import PairNorm
import time, sys

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
    if name == 'physnet':
        return PhysDimeNet(**config)
    
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
        self.gradCam = config['gradCam']
        self.bn = config['bn']
        self.act_fn = activation_func(config)

        if self.num_layer < 1:
            raise ValueError("Number of GNN layers must be no less than 1.")
        if self.config['gnn_type'] in ['gineconv', 'pnaconv', 'nnconv']:
            self.linear_b = Linear(self.config['num_bond_features'], self.emb_dim)
        
        # define graph conv layers
        if self.gnn_type in ['dmpnn']:
            self.gnns = get_gnn(self.config) # already contains multiple layers
        else:
            self.linear_x = Linear(self.config['num_atom_features'], self.emb_dim)
            self.gnns = nn.ModuleList()
            for _ in range(self.num_layer):
                self.gnns.append(get_gnn(self.config).model())
        
        if config['pooling'] == 'edge': # for edge pooling only
            self.pool = PoolingFN(self.config)
            assert len(self.pool) == self.num_layer
        
        ###List of batchnorms
        if self.bn and self.gnn_type not in ['dmpnn']:
            self.batch_norms = nn.ModuleList()
            for _ in range(self.num_layer):
                self.batch_norms.append(nn.BatchNorm1d(self.emb_dim))

        #self.pair_norm = PairNorm()
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
        elif len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        elif len(argv) == 5:
            f_atoms, f_bonds, a2b, b2a, b2revb = argv[0], argv[1], argv[2], argv[3], argv[4]
        else:
            raise ValueError("unmatched number of arguments.")

        if self.gnn_type == 'dmpnn':
            node_representation = self.gnns(f_atoms, f_bonds, a2b, b2a, b2revb)
        else:
            x = self.linear_x(x) # first linear on atoms 
            h_list = [x]
            #x = F.relu(self.linear_x(x))
            if self.config['gnn_type'] in ['gineconv', 'pnaconv', 'nnconv']:
                edge_attr = self.linear_b(edge_attr.float()) # first linear on bonds 
                #edge_attr = F.relu(self.linear_b(edge_attr.float())) # first linear on bonds 

            for layer in range(self.num_layer):
                if self.config['residual_connect']: # adding residual connection
                    residual = h_list[layer] 
                if self.config['gnn_type'] in ['gineconv', 'pnaconv', 'nnconv']:
                    h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
                else:
                    h = self.gnns[layer](h_list[layer], edge_index)
                
                ### in order of Skip >> BN >> ReLU
                if self.config['residual_connect']:
                    h += residual
                if self.bn:
                    h = self.batch_norms[layer](h)
                
                #h = self.pair_norm(h, data.batch)
                if layer == self.num_layer - 1:
                    #remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training = self.training)
                    self.last_conv = self.get_activations(h)
                else:
                    h = self.act_fn(h)
                    h = F.dropout(h, self.drop_ratio, training = self.training)
                if self.config['pooling'] == 'edge':
                    h, edge_index, batch, _ = self.pool[layer](h, edge_index, batch=batch)
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
        
        if self.config['pooling'] == 'edge':
            return node_representation, batch
        else:
            return node_representation

class GNN_1(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1, self).__init__()
        self.config = config
        self.dataset = config['dataset']
        self.num_layer = config['num_layer']
        self.fully_connected_layer_sizes = config['fully_connected_layer_sizes']
        self.JK = config['JK']
        self.emb_dim = config['emb_dim']
        self.num_tasks = config['num_tasks']
        self.graph_pooling = config['pooling']
        self.propertyLevel = config['propertyLevel']
        self.gnn_type = config['gnn_type']
        self.gradCam = config['gradCam']
        self.tsne = config['tsne']
        self.uncertainty = config['uncertainty']
        self.uncertaintyMode = config['uncertaintyMode']
        self.weight_regularizer = config['weight_regularizer']
        self.dropout_regularizer = config['dropout_regularizer']
        self.features = config['mol_features']
        self.twoHop = config['twoHop']

        if self.twoHop:
            self.gnn = GNN_twoHop(config)
        else:
            self.gnn = GNN(config)
        self.outLayers = nn.ModuleList()
        if self.uncertainty:
            self.uncertaintyLayers = nn.ModuleList()
        if self.graph_pooling not in ['edge', 'topk', 'sag']: # except for edge pooling, coded here
            self.pool = PoolingFN(config) # after node embedding updating and pooling 

        #For graph-level property predictions
        if self.graph_pooling[:-1][0] == "set2set": # set2set will double dimension
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat": # change readout layers input and output dimension
            embed_size = self.mult * (self.num_layer + 1) * self.emb_dim
        elif self.graph_pooling == 'conv': # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim / 2  
        elif self.features:
            if self.dataset == 'sol_calc/ALL': # 208 total mol descriptors # total is 200
               embed_size = self.mult * self.emb_dim + 208
        else: # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim 

        for idx, (L_in, L_out) in enumerate(zip([embed_size] + self.fully_connected_layer_sizes, self.fully_connected_layer_sizes + [self.num_tasks])):
            if idx != len(self.fully_connected_layer_sizes):
                fc = nn.Sequential(Linear(L_in, L_out), activation_func(config), nn.Dropout(config['drop_ratio']))
                #L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)    
                if self.uncertainty: # for uncertainty 
                    self.uncertaintyLayers.append(NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer)) 
            else:
                fc = nn.Sequential(Linear(L_in, L_out), nn.Dropout(config['drop_ratio']))
                last_fc = fc
            self.outLayers.append(fc)
        #if self.propertyLevel == 'atomMol':
        #    self.outLayers1 = copy.deepcopy(self.outLayers) # another trainable linear layers

        if self.uncertaintyMode == 'epistemic': 
            self.drop_mu = NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer) # working on last layer
        if self.uncertaintyMode == 'aleatoric': # 
            self.outLayers.append(last_fc) 
        if self.config['normalize']:
            shift_matrix = torch.zeros(self.emb_dim, 1)
            scale_matrix = torch.zeros(self.emb_dim, 1).fill_(1.0)
            shift_matrix[:, :] = self.config['energy_shift'].view(1, -1)
            scale_matrix[:, :] = self.config['energy_scale'].view(1, -1)
            self.register_parameter('scale', torch.nn.Parameter(scale_matrix, requires_grad=True))
            self.register_parameter('shift', torch.nn.Parameter(shift_matrix, requires_grad=True))

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file)) 

    def forward(self, data):
        if self.gnn_type == 'dmpnn':
             f_atoms, f_bonds, a2b, b2a, b2revb, a_scope = data.x, data.edge_attr, data.a2b, data.b2a, data.b2revb, data.a_scope
        else:
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr.long(), data.batch
        
        if self.config['normalize']:
            Z = data.Z
            if self.gnn_type == 'dmpnn':
               Z = torch.cat((torch.zeros(1).int(), data.Z))
        if self.graph_pooling == 'edge':
            node_representation, final_batch = self.gnn(x, edge_index, edge_attr, batch) # node updating
        elif self.twoHop:
            node_representation = self.gnn(data)
        elif self.gnn_type == 'dmpnn':
            node_representation = self.gnn(f_atoms, f_bonds, a2b, b2a, b2revb) # node updating 
        else:
            node_representation = self.gnn(x, edge_index, edge_attr) # node updating 

        # graph pooling 
        if self.propertyLevel in ['molecule', 'atomMol', 'multiMol']: 
            if self.graph_pooling == 'conv':
                MolEmbed_list = []
                for p in self.pool:
                    MolEmbed_list.append(p(node_representation, batch))
                MolEmbed_stack = torch.stack(MolEmbed_list, 1).squeeze()
                MolEmbed = self.pool(MolEmbed_stack).squeeze()
            elif self.graph_pooling == 'edge':
                MolEmbed = global_mean_pool(node_representation, final_batch).squeeze()
            elif self.graph_pooling == 'atomic': # 
                MolEmbed = node_representation #(-1, emb_dim)
            else: # normal pooling functions besides conv and edge
                MolEmbed = self.pool(node_representation, batch)  # atomic read-out (-1, 1)
        if self.propertyLevel == 'atom':
            MolEmbed = node_representation #(-1, emb_dim)
        #elif self.propertyLevel in ['atom', 'atomMol']:
        #    MolEmbed = node_representation  # atomic read-out
        if self.features: # concatenating molecular features
            #print(data.features.shape, MolEmbed.shape)
            MolEmbed = torch.cat((MolEmbed, data.features.view(MolEmbed.shape[0], -1)), -1)
        if not self.training and not self.gradCam and self.tsne: # for TSNE analysis
            return node_representation, MolEmbed

        # read-out layers
        if not self.uncertainty:
            for layer in self.outLayers: # 
                MolEmbed = layer(MolEmbed)
            if self.config['normalize']:
                MolEmbed = self.scale[Z, :] * MolEmbed + self.shift[Z, :]
                #print(MolEmbed.shape)
                #print(self.scale[Z, :].shape)
            #if self.dataset == 'solNMR' and self.propertyLevel == 'atomMol':
            #    for layer in self.outLayers1:
            #        node_representation = layer(node_representation)
            if self.num_tasks > 1:
                if self.dataset in ['solNMR', 'solALogP', 'qm9/nmr/allAtoms']:
                    assert MolEmbed.size(-1) == self.num_tasks
                    if self.propertyLevel == 'atomMol':
                        return MolEmbed[:,0].view(-1), self.pool(MolEmbed[:,1], batch).view(-1)
                    if self.propertyLevel == 'multiMol':
                        return self.pool(MolEmbed[:,0], batch).view(-1), \
                            self.pool(MolEmbed[:,1], batch).view(-1), \
                            self.pool(MolEmbed[:,2], batch).view(-1), \
                            self.pool(MolEmbed[:,1]-MolEmbed[:,0], batch).view(-1), \
                            self.pool(MolEmbed[:,2]-MolEmbed[:,0], batch).view(-1), \
                            self.pool(MolEmbed[:,2]-MolEmbed[:,1], batch).view(-1)
                    if self.propertyLevel == 'atomMultiMol':
                        return MolEmbed[:,0].view(-1), \
                            self.pool(MolEmbed[:,1], batch).view(-1), \
                            self.pool(MolEmbed[:,2], batch).view(-1), \
                            self.pool(MolEmbed[:,3], batch).view(-1), \
                            self.pool(MolEmbed[:,2]-MolEmbed[:,1], batch).view(-1), \
                            self.pool(MolEmbed[:,3]-MolEmbed[:,1], batch).view(-1), \
                            self.pool(MolEmbed[:,3]-MolEmbed[:,2], batch).view(-1)

                elif self.dataset == 'solEFGs':
                    assert MolEmbed.size(-1) == self.num_tasks
                    return MolEmbed[:,:-1], self.pool(MolEmbed[:,-1], batch).view(-1)
                else:
                    return MolEmbed, None
            elif self.propertyLevel == 'atom' and self.dataset not in ['solNMR', 'solALogP', 'qm9/nmr/allAtoms']:
                return MolEmbed.view(-1,1), None
            elif self.propertyLevel == 'molecule' and self.graph_pooling == 'atomic':
                if self.gnn_type == 'dmpnn':
                    return MolEmbed.view(-1), self.pool(MolEmbed, a_scope).view(-1)
                else:
                    return MolEmbed.view(-1), global_add_pool(MolEmbed, batch).view(-1)
            else:
                return MolEmbed.view(-1), MolEmbed.view(-1)
            
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
        
class GNN_twoHop(torch.nn.Module):
    def __init__(self, config):
        super(GNN_twoHop, self).__init__()
        self.config = config
        self.config_twoHop = self.config.copy()
        self.num_layer = config['num_layer']
        self.emb_dim = config['emb_dim']
        self.drop_ratio = config['drop_ratio']
        self.JK = config['JK']
        self.gnn_type = config['gnn_type']
        self.gradCam = config['gradCam']
        self.bn = config['bn']
        self.config_twoHop['gnn_type'] = 'graphconv'
        self.act_fn = activation_func(config)

        if self.num_layer < 1:
            raise ValueError("Number of GNN layers must be no less than 1.")
        self.linear_x = Linear(self.config['num_atom_features'], self.emb_dim)
        if self.config['gnn_type'] in ['gineconv', 'pnaconv', 'nnconv']:
            self.linear_b = Linear(self.config['num_bond_features'], self.emb_dim)
        # define graph conv layers
        self.gnns = torch.nn.ModuleList()
        self.gnns_twoHop = torch.nn.ModuleList()
        for _ in range(self.num_layer):
            self.gnns.append(get_gnn(self.config).model())
            self.gnns_twoHop.append(get_gnn(self.config_twoHop).model())
        if config['pooling'] == 'edge': # for edge pooling only
            self.pool = PoolingFN(self.config)
            assert len(self.pool) == self.num_layer
        
        ###List of batchnorms
        if self.bn:
            self.batch_norms = torch.nn.ModuleList()
            for _ in range(self.num_layer):
                self.batch_norms.append(torch.nn.BatchNorm1d(self.emb_dim))

        #self.pair_norm = PairNorm()
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
        elif len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            #x, edge_index, edge_attr, edge_index_2, edge_attr_2 = data.x, data.edge_index, data.edge_attr, data.edge_index_twoHop, data.edge_attr_twoHop
            x, edge_index, edge_attr, edge_index_full = data.x, data.edge_index, data.edge_attr, data.edge_index_full
        else:
            raise ValueError("unmatched number of arguments.")
        
        x = self.linear_x(x) # first linear on atoms 
        if self.config['gnn_type'] in ['gineconv', 'pnaconv', 'nnconv']:
            edge_attr = self.linear_b(edge_attr.float()) # first linear on bonds 
        h_list = [x]
        for layer in range(self.num_layer):
            if self.config['residual_connect']: # adding residual connection
                residual = h_list[layer] 
            if self.config['gnn_type'] in ['gineconv', 'pnaconv', 'nnconv']:
                h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            else:
                h = self.gnns[layer](h_list[layer], edge_index)
            #print(edge_index_2.dtype, edge_attr_2.dtype)
            #h_twoHop = self.gnns_twoHop[layer](h_list[layer], edge_index_2, edge_attr_2.float())
            if layer == self.num_layer-1:
                h = self.gnns_twoHop[layer](h_list[layer], edge_index_full)
            #h = h + h_twoHop
            ### in order of Skip >> BN >> ReLU
            if self.config['residual_connect']:
                h += residual
            if self.bn:
                h = self.batch_norms[layer](h)
            
            #h = self.pair_norm(h, data.batch)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
                self.last_conv = self.get_activations(h)
            else:
                h = self.act_fn(h)
                h = F.dropout(h, self.drop_ratio, training = self.training)
            if self.config['pooling'] == 'edge':
                h, edge_index, batch, _ = self.pool[layer](h, edge_index, batch=batch)
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
        if self.config['pooling'] == 'edge':
            return node_representation, batch
        else:
            return node_representation


class GNN_1_2(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1_2, self).__init__()
        self.num_i_2 = config['num_i_2']
        self.config = config
        self.dataset = config['dataset']
        self.num_layer = config['num_layer']
        self.fully_connected_layer_sizes = config['fully_connected_layer_sizes']
        self.JK = config['JK']
        self.emb_dim = config['emb_dim']
        self.num_tasks = config['num_tasks']
        self.graph_pooling = config['pooling']
        self.propertyLevel = config['propertyLevel']
        self.gnn_type = config['gnn_type']
        self.gradCam = config['gradCam']
        self.tsne = config['tsne']
        self.uncertainty = config['uncertainty']
        self.uncertaintyMode = config['uncertaintyMode']
        self.weight_regularizer = config['weight_regularizer']
        self.dropout_regularizer = config['dropout_regularizer']
        self.features = config['mol_features']

        self.gnn = GNN(config)
        self.convISO1 = GraphConv(self.emb_dim + self.num_i_2, self.emb_dim)
        self.convISO2 = GraphConv(self.emb_dim, self.emb_dim)
        self.outLayers = nn.ModuleList()
        #self.out_batch_norms = torch.nn.ModuleList()

        if self.uncertainty:
            self.uncertaintyLayers = nn.ModuleList()
        if self.graph_pooling != 'edge': # except for edge pooling, coded here
            self.pool = PoolingFN(config)

        #For graph-level property predictions
        if self.graph_pooling[:-1][0] == "set2set": # set2set will double dimension
            self.mult = 3
        else:
            self.mult = 2
        
        if self.JK == "concat": # change readout layers input and output dimension
            embed_size = self.mult * (self.num_layer + 1) * self.emb_dim
        elif self.graph_pooling == 'conv': # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim / 2  
        elif self.features:
            if self.dataset == 'sol_calc/ALL': # 208 total mol descriptors # total is 200
               embed_size = self.mult * self.emb_dim + 208
        else: # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim 
        
        for idx, (L_in, L_out) in enumerate(zip([embed_size] + self.fully_connected_layer_sizes, self.fully_connected_layer_sizes + [self.num_tasks])):
            if idx != len(self.fully_connected_layer_sizes):
                fc = nn.Sequential(Linear(L_in, L_out), activation_func(config), nn.Dropout(config['drop_ratio']))
                #L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)    
                if self.uncertainty: # for uncertainty 
                    self.uncertaintyLayers.append(NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer)) 
            else:
                fc = nn.Sequential(Linear(L_in, L_out), nn.Dropout(config['drop_ratio']))
                last_fc = fc
            self.outLayers.append(fc)

        if self.uncertaintyMode == 'epistemic': 
            self.drop_mu = NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer) # working on last layer
        if self.uncertaintyMode == 'aleatoric': # 
            self.outLayers.append(last_fc) 

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))
        #self.convISO1.load_state_dict(torch.load(conv1File))
        #self.convISO2.load_state_dict(torch.load(conv2File))

    def forward(self, data):
        x, iso_type_2, edge_index, edge_index_2, assignment_index_2, edge_attr, batch, batch_2 = \
                data.x, data.iso_type_2, data.edge_index, data.edge_index_2, data.assignment_index_2, \
                    data.edge_attr.long(), data.batch, data.batch_2

        node_representation = self.gnn(x, edge_index, edge_attr)
        x_1 = self.pool(node_representation, batch)

        x = avg_pool(node_representation, data.assignment_index_2)
        #data.x = torch.cat([data.x, data_iso], dim=1)
        x = torch.cat([x, iso_type_2], dim=1)
        x = F.relu(self.convISO1(x, edge_index_2))
        x = F.relu(self.convISO2(x, edge_index_2))
        x_2 = scatter_mean(x, batch_2, dim=0)   # to add stability to models
        
        MolEmbed = torch.cat([x_1, x_2], dim=1)
        if not self.training and not self.gradCam and self.tsne: # for TSNE analysis
            return node_representation, MolEmbed

        # read-out layers
        if not self.uncertainty:
            for layer in self.outLayers:
                MolEmbed = layer(MolEmbed)
            if self.num_tasks > 1:
                return MolEmbed, None
            if self.propertyLevel == 'atom':
                return MolEmbed.view(-1,1), None
            if self.graph_pooling == 'edge':
                return MolEmbed.view(-1), None
            else:
                #return self.pool(MolEmbed, batch).view(-1), None
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
        self.dataset = config['dataset']
        self.num_layer = config['num_layer']
        self.fully_connected_layer_sizes = config['fully_connected_layer_sizes']
        self.JK = config['JK']
        self.emb_dim = config['emb_dim']
        self.num_tasks = config['num_tasks']
        self.graph_pooling = config['pooling']
        self.propertyLevel = config['propertyLevel']
        self.gnn_type = config['gnn_type']
        self.gradCam = config['gradCam']
        self.tsne = config['tsne']
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
            embed_size = self.mult * (self.num_layer + 1) * self.emb_dim
        elif self.graph_pooling == 'conv': # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim / 2  
        elif self.features:
            if self.dataset == 'sol_calc/ALL': # 208 total mol descriptors # total is 200
               embed_size = self.mult * self.emb_dim + 208
        else: # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim 

        for idx, (L_in, L_out) in enumerate(zip([embed_size] + self.fully_connected_layer_sizes, self.fully_connected_layer_sizes + [self.num_tasks])):
            if idx != len(self.fully_connected_layer_sizes):
                fc = nn.Sequential(Linear(L_in, L_out), activation_func(config), nn.Dropout(config['drop_ratio']))
                #L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)    
                if self.uncertainty: # for uncertainty 
                    self.uncertaintyLayers.append(NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer)) 
            else:
                fc = nn.Sequential(Linear(L_in, L_out), nn.Dropout(config['drop_ratio']))
            self.outLayers.append(fc)

        if self.uncertaintyMode == 'epistemic': 
            self.drop_mu = NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer) # working on last layer
        if self.uncertaintyMode == 'aleatoric': # 
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
        self.config = config
        self.dataset = config['dataset']
        self.num_layer = config['num_layer']
        self.fully_connected_layer_sizes = config['fully_connected_layer_sizes']
        self.JK = config['JK']
        self.emb_dim = config['emb_dim']
        self.num_tasks = config['num_tasks']
        self.graph_pooling = config['pooling']
        self.propertyLevel = config['propertyLevel']
        self.gnn_type = config['gnn_type']
        self.gradCam = config['gradCam']
        self.tsne = config['tsne']
        self.uncertainty = config['uncertainty']
        self.uncertaintyMode = config['uncertaintyMode']
        self.weight_regularizer = config['weight_regularizer']
        self.dropout_regularizer = config['dropout_regularizer']
        self.features = config['mol_features']
        self.twoHop = config['twoHop']

        self.gnn_solute = GNN(config)
        self.gnn_hydrated_solute = GNN(config)
        self.outLayers = nn.ModuleList()
        if self.uncertainty:
            self.uncertaintyLayers = nn.ModuleList()
        if self.graph_pooling not in ['edge', 'topk', 'sag']: # except for edge pooling, coded here
            self.pool = PoolingFN(config) # after node embedding updating and pooling 
        
        #For graph-level property predictions
        if self.graph_pooling[:-1][0] == "set2set": # set2set will double dimension
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat": # change readout layers input and output dimension
            embed_size = self.mult * (self.num_layer + 1) * self.emb_dim
        elif self.graph_pooling == 'conv': # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim / 2  
        elif self.features:
            if self.dataset == 'sol_calc/ALL': # 208 total mol descriptors # total is 200
               embed_size = self.mult * self.emb_dim + 208
        else: # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim 

        for idx, (L_in, L_out) in enumerate(zip([embed_size] + self.fully_connected_layer_sizes, self.fully_connected_layer_sizes + [self.num_tasks])):
            if idx != len(self.fully_connected_layer_sizes):
                fc = nn.Sequential(Linear(L_in, L_out), activation_func(config), nn.Dropout(config['drop_ratio']))
                #L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)    
                if self.uncertainty: # for uncertainty 
                    self.uncertaintyLayers.append(NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer)) 
            else:
                fc = nn.Sequential(Linear(L_in, L_out), nn.Dropout(config['drop_ratio']))
                last_fc = fc
            self.outLayers.append(fc)
        #if self.propertyLevel == 'atomMol':
        #    self.outLayers1 = copy.deepcopy(self.outLayers) # another trainable linear layers

        if self.uncertaintyMode == 'epistemic': 
            self.drop_mu = NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer) # working on last layer
        if self.uncertaintyMode == 'aleatoric': # 
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

        #solute_representation = self.pool(solute_node_representation, batch)
        if self.dataset in ['logpWithWater']:
            solute_representation = self.pool(solute_node_representation[mask<1], batch[mask<1])
        #hydrated_solute_representation = self.pool(hydrated_solute_node_representation[mask<1], hyd_solute_batch[mask<1])
        
        if self.dataset in ['logpWithWater']:
            final_representation = torch.cat([solute_node_representation, hydrated_solute__node_representation], dim=1)
        else:
            batch = hyd_solute_batch
            final_representation = hydrated_solute_node_representation
        
        node_representation = final_representation
        # graph pooling 
        if self.propertyLevel in ['molecule', 'atomMol']: 
            if self.graph_pooling == 'conv':
                MolEmbed_list = []
                for p in self.pool:
                    MolEmbed_list.append(p(node_representation, batch))
                MolEmbed_stack = torch.stack(MolEmbed_list, 1).squeeze()
                MolEmbed = self.pool(MolEmbed_stack).squeeze()
            elif self.graph_pooling == 'edge':
                MolEmbed = global_mean_pool(node_representation, final_batch).squeeze()
            elif self.graph_pooling == 'atomic': # 
                MolEmbed = node_representation #(-1, emb_dim)
            else: # normal pooling functions besides conv and edge
                MolEmbed = self.pool(node_representation, batch)  # atomic read-out (-1, 1)
        if self.propertyLevel == 'atom':
            MolEmbed = node_representation #(-1, emb_dim)
        #elif self.propertyLevel in ['atom', 'atomMol']:
        #    MolEmbed = node_representation  # atomic read-out
        if self.features: # concatenating molecular features
            #print(data.features.shape, MolEmbed.shape)
            MolEmbed = torch.cat((MolEmbed, data.features.view(MolEmbed.shape[0], -1)), -1)
        if not self.training and not self.gradCam and self.tsne: # for TSNE analysis
            return node_representation, MolEmbed

        # read-out layers
        if not self.uncertainty:
            for layer in self.outLayers: # 
                MolEmbed = layer(MolEmbed) 
            #if self.dataset == 'solNMR' and self.propertyLevel == 'atomMol':
            #    for layer in self.outLayers1:
            #        node_representation = layer(node_representation)
            if self.num_tasks > 1:
                if self.dataset == 'solNMR':
                    assert MolEmbed.size(-1) == self.num_tasks
                    return MolEmbed[:,0].view(-1), self.pool(MolEmbed[:,1], batch).view(-1)
                if self.dataset == 'solEFGs':
                    assert MolEmbed.size(-1) == self.num_tasks
                    return MolEmbed[:,:-1], self.pool(MolEmbed[:,-1], batch).view(-1)
                else:
                    return MolEmbed, None
            elif self.propertyLevel == 'atom' and self.dataset != 'solNMR':
                return MolEmbed.view(-1,1), None
            elif self.propertyLevel == 'molecule' and self.graph_pooling == 'atomic':
                return MolEmbed.view(-1), self.pool(MolEmbed, batch).view(-1)
            else:
                return MolEmbed.view(-1), MolEmbed.view(-1)
            
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

class GNN_1_interaction(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1_interaction, self).__init__()
        self.dataset = config['dataset']
        self.num_layer = config['num_layer']
        self.fully_connected_layer_sizes = config['fully_connected_layer_sizes']
        self.JK = config['JK']
        self.emb_dim = config['emb_dim']
        self.num_tasks = config['num_tasks']
        self.graph_pooling = config['pooling']
        self.propertyLevel = config['propertyLevel']
        self.gnn_type = config['gnn_type']
        self.gradCam = config['gradCam']
        self.tsne = config['tsne']
        self.uncertainty = config['uncertainty']
        self.uncertaintyMode = config['uncertaintyMode']
        self.weight_regularizer = config['weight_regularizer']
        self.dropout_regularizer = config['dropout_regularizer']
        self.features = config['mol_features']

        self.gnn_solute = GNN(config)
        if self.solvent == 'water': # to do adding to args
            self.gnn_solvent = nn.Sequential(nn.Linear(self.config['num_atom_features'], self.emb_dim),
                                            torch.nn.ReLU(), \
                                            nn.Linear(self.emb_dim, self.emb_dim))
        elif self.solvent == 'octanol':
            self.gnn_solvent = GNN(config)
        else:
            raise ValueError('Solvent need to be specified.')

        self.imap = nn.Linear(2*self.emb_dim, 1) # create a placeholder for interaction map
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

        #For graph-level binary classification
        if self.graph_pooling[:-1][0] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat": # change readout layers input and output dimension
            embed_size = self.mult * (self.num_layer + 1) * self.emb_dim
        elif self.graph_pooling == 'conv': # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim / 2  
        elif self.features:
            if self.dataset == 'sol_calc/ALL': # 208 total mol descriptors # total is 200
               embed_size = self.mult * self.emb_dim + 208
        else: # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim 

        for idx, (L_in, L_out) in enumerate(zip([embed_size] + self.fully_connected_layer_sizes, self.fully_connected_layer_sizes + [self.num_tasks])):
            if idx != len(self.fully_connected_layer_sizes):
                fc = nn.Sequential(Linear(L_in, L_out), activation_func(config), nn.Dropout(config['drop_ratio']))
                #L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)    
                if self.uncertainty: # for uncertainty 
                    self.uncertaintyLayers.append(NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer)) 
            else:
                fc = nn.Sequential(Linear(L_in, L_out), nn.Dropout(config['drop_ratio']))
                last_fc = fc
            self.outLayers.append(fc)

        if self.uncertaintyMode == 'epistemic': 
            self.drop_mu = NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer) # working on last layer
        if self.uncertaintyMode == 'aleatoric': # 
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

        # Interaction part 
        len_map = torch.mm(solute_length_matrix.t(), solvent_length_matrix)  # interaction map to control which solvent mols  22*27
        #corresponds to which solute mol
        if 'dot' not in self.interaction: # to be adding to args
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
        
    def forward(self, X, atom_mol_batch):
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
        out = scatter(reduce='add', src=out, index=atom_mol_batch, dim=0)
        return out

if __name__ == "__main__":
    pass
