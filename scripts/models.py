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
                elif self.config['gnn_type'] in ['dnn']:
                    h = self.gnns[layer](h_list[layer])
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
        self.features = config['mol_features']

        self.gnn = GNN(config)
        if 'Residue' in self.config['style']:
            config_res = config.copy()
            config_res['num_atom_features'] = 20
            config_res['num_bond_features'] = 64
            self.gnn_residue = GNN(config_res)
        self.outLayers = nn.ModuleList()

        if self.graph_pooling not in ['edge', 'topk', 'sag']: # except for edge pooling, coded here
            self.pool = PoolingFN(config) # after node embedding updating and pooling 

        #For graph-level property predictions
        if self.graph_pooling == "set2set": # set2set will double dimension
            self.mult = 2
        else:
            self.mult = 1

        if 'Residue' in self.config['style']:
            self.mult = 2
        
        if self.JK == "concat": # change readout layers input and output dimension
            embed_size = self.mult * (self.num_layer + 1) * self.emb_dim
        elif self.graph_pooling == 'conv': # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim / 2  
        elif self.features:
            if self.graph_pooling not in ['atomic']:
                embed_size = self.mult * self.emb_dim + 208
        else: # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim 

        for idx, (L_in, L_out) in enumerate(zip([embed_size] + self.fully_connected_layer_sizes, self.fully_connected_layer_sizes + [self.num_tasks])):
            if idx != len(self.fully_connected_layer_sizes):
                fc = nn.Sequential(Linear(L_in, L_out), activation_func(config), nn.Dropout(config['drop_ratio']))
                self.outLayers.append(fc)
            else:
                if self.uncertainty:
                    if self.uncertaintyMode == 'aleatoric':
                        fc_mean = nn.Sequential(Linear(L_in, L_out), nn.Dropout(config['drop_ratio']))
                        fc_var = nn.Sequential(Linear(L_in, L_out), nn.Dropout(config['drop_ratio']))
                        self.outLayers.append(fc_mean)
                        self.outLayers.append(fc_var)
                    elif self.uncertaintyMode == 'evidence':
                        L_out = 4 * self.num_tasks
                        fc = nn.Sequential(Linear(L_in, L_out), nn.Dropout(config['drop_ratio']))
                        self.outLayers.append(fc)
                    else:
                        pass
                else:
                    fc = nn.Sequential(Linear(L_in, L_out), nn.Dropout(config['drop_ratio']))
                    self.outLayers.append(fc)
                    
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
        
        if 'Residue' in self.config['style']:
            x_res, edge_index_res, edge_attr_res, atom_res_map = data.x_res, data.edge_index_res, data.edge_attr_res.type(torch.float), data.atom_res_map
        
        if self.config['normalize']:
            Z = data.Z
            if self.gnn_type == 'dmpnn':
               Z = torch.cat((torch.zeros(1).int(), data.Z))
        
        if self.graph_pooling == 'edge':
            node_representation, final_batch = self.gnn(x, edge_index, edge_attr, batch) # node updating
        if self.gnn_type == 'dmpnn':
            node_representation = self.gnn(f_atoms, f_bonds, a2b, b2a, b2revb) # node updating 
        elif self.gnn_type == 'dnn':
            node_representation = self.gnn(data) # node updating 
        else:
            node_representation = self.gnn(x, edge_index, edge_attr) # node updating
        
        if 'Residue' in self.config['style']: # amino acid embedding
            res_representation = self.gnn_residue(x_res, edge_index_res, edge_attr_res) # node updating
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
       
        if self.features: # concatenating molecular features
            if self.graph_pooling not in ['atomic']:
                MolEmbed = torch.cat((MolEmbed, data.features.view(MolEmbed.shape[0], -1)), -1)
        if not self.training and not self.gradCam and self.tsne: # for TSNE analysis
            return node_representation, MolEmbed
        
        if 'Residue' in self.config['style']:
            #residue_representation = self.fc_res_layer0(data.x_res)
            #residue_representation = self.fc_res_layer1(residue_representation)
            #residue_representation = self.fc_res_layer2(residue_representation)
            #reside_pos_embedding = self.pos_encoder(residue_representation.view(-1, 1, self.emb_dim)).squeeze()
            #reside_pos_embedding = reside_pos_embedding.repeat_interleave(data.atom_res_map, dim = 0)
            residue_to_atom_embedding = res_representation.repeat_interleave(atom_res_map, dim = 0) # assign each residue embedding to its corresponding atom embeddings
            #interaction = torch.mm(node_representation, reside_pos_embedding.transpose(1,0)) / np.sqrt(self.emb_dim)
            #node_representation_interaction = torch.mm(interaction, node_representation)
            MolEmbed = torch.cat((node_representation, residue_to_atom_embedding), 1)
        # read-out layers
        if not self.uncertainty:
            for layer in self.outLayers: # 
                MolEmbed = layer(MolEmbed)
            if self.config['normalize']:
                MolEmbed = self.scale[Z, :] * MolEmbed + self.shift[Z, :]
                
            if self.num_tasks > 1:
                if self.dataset in ['qm9/nmr/allAtoms', 'sol_calc/smaller', 'sol_calc/all']:
                    assert MolEmbed.size(-1) == self.num_tasks
                    if self.propertyLevel == 'atomMol':
                        return MolEmbed[:,0].view(-1), self.pool(MolEmbed[:,1], batch).view(-1)
                    if self.propertyLevel == 'multiMol': # multi-task mol-level properties
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
                else:
                    return MolEmbed, None
            
            elif self.propertyLevel == 'atom': 
                if self.gnn_type == 'dmpnn':
                    a_start, a_size = a_scope[0]
                    MolEmbed = MolEmbed.narrow(0, a_start, a_size)
                    return MolEmbed.view(-1), None 
                else:
                    return MolEmbed.view(-1,1), None
            
            elif self.propertyLevel == 'molecule' and self.graph_pooling == 'atomic':
                if self.gnn_type == 'dmpnn':
                    return MolEmbed.view(-1), self.pool(MolEmbed, a_scope).view(-1)
                else: # pnaconv, ginconv
                    return MolEmbed.view(-1), global_add_pool(MolEmbed, batch).view(-1)
            
            else: # 
                return MolEmbed.view(-1), MolEmbed.view(-1)
            
        # for uncertainty analysis
        else:
            #assert len(self.outLayers) == 4
            #assert self.pool not in ['conv', 'edge', 'atomic']
            if self.uncertaintyMode == 'aleatoric':
                for layer in self.outLayers[:-2]: # 
                    MolEmbed = layer(MolEmbed)
                if self.config['normalize']:
                    MolEmbed = self.scale[Z, :] * MolEmbed + self.shift[Z, :]
                MolEmbed_mean = self.outLayers[-2](MolEmbed)
                MolEmbed_log_var = self.outLayers[-1](MolEmbed)

                if self.propertyLevel == 'atom':
                    return (MolEmbed_mean.view(-1,1), MolEmbed_log_var.view(-1,1), None)
                
                elif self.propertyLevel == 'molecule' and self.graph_pooling == 'atomic':
                    if self.gnn_type == 'dmpnn':
                        return (None, self.pool(MolEmbed_mean, a_scope).view(-1), self.pool(MolEmbed_log_var, a_scope).view(-1))
                    else:
                        return (None, global_add_pool(MolEmbed_mean, batch).view(-1), global_add_pool(MolEmbed_log_var, batch).view(-1))
                
                else: # molecule level but not atomic pooling 
                    return (None, MolEmbed_mean.view(-1), MolEmbed_log_var.view(-1))
            
            if self.uncertaintyMode == 'evidence':
                for layer in self.outLayers: # 
                    MolEmbed = layer(MolEmbed)
                min_val = 1e-6
                # Split the outputs into the four distribution parameters
                means, loglambdas, logalphas, logbetas = torch.split(MolEmbed, MolEmbed.shape[1]//4, dim=1)
                if self.config['normalize']:
                    means = self.scale[Z, :] * means + self.shift[Z, :]
                    #means = global_add_pool(means, batch)
                lambdas = torch.nn.Softplus()(loglambdas) + min_val
                alphas = torch.nn.Softplus()(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
                betas = torch.nn.Softplus()(logbetas) + min_val
                
                if self.graph_pooling == 'atomic':
                    if self.gnn_type == 'dmpnn':
                        means = self.pool(means, a_scope).view(-1)
                        lambdas = self.pool(lambdas, a_scope).view(-1)
                        alphas = self.pool(alphas, a_scope).view(-1)
                        betas = self.pool(betas, a_scope).view(-1)
                    else:
                        means = global_add_pool(means, batch)
                        lambdas = global_add_pool(lambdas, batch)
                        alphas = global_add_pool(alphas, batch)
                        betas = global_add_pool(betas, batch)

                return torch.stack((means, lambdas, alphas, betas),
                                         dim = -1).view(-1,4)

