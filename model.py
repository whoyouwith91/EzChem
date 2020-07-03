import torch
import torch.nn as nn
from torch.nn import GRU, LSTM
import torch.nn.functional as F
from torch_geometric.nn import NNConv, EdgeConv, GATConv, GraphConv, SAGPooling
from torch.autograd import Variable
from k_gnn import GraphConv, avg_pool, add_pool, max_pool
from layers import *
from helper import *

#__all__ = [knn, loopybp]
def get_model(config):

    name = config['model']
    if name == None:
       raise ValueError('Please specify one model you want to work on!')
    if name == '1-2-GNN':
       return knn
    if name == '1-2-GNN_dropout':
        return knn_dropout
    if name == '1-2-GNN_swag':
        return knn_swag
    if name == 'ConvGRU':
       return ConvGRU
    if name == '1-GNN':
       return GNN_1
    if name == 'CONVATT':
       return ConvATT
    if name == 'adaConv':
       return AdaConv
    if name == 'NNConvGAT':
       return NNConvGAT
    if name == 'gradcam': # To determine which atoms play most important
       return knn_GradCAM_rev
    if name == 'loopybp':
       return LoopyBP
    if name == 'loopybp_dropout':
        return LoopyBP_dropout
    if name == 'loopybp_swag':
        return LoopyBP_swag
    if name == 'wlkernel':
       return wlkernel
    if name == 'wlkernel_dropout':
        return wlkernel_dropout
    if name == 'wlkernel_swag':
        return wlkernel_swag
    if name == 'VAE':
        return VAE
    

class GNN_1(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1, self).__init__()
        self.config = config.copy()
        self.fn = activation_func(config)
        self.pooling = pooling(config)
        M_in, B_in, M_out = config['num_features'], config['num_bond_features'], int(config['dimension']/2)

        self.dropout = nn.Dropout(p=config['dropout'])
        self.conv = nn.ModuleList()
        self.out = nn.ModuleList()
        
        if config['water_interaction']: # water molecules embedding
            self.embWater = torch.load('/beegfs/dz1061/datasets/ws/raw/water.pt')
            self.f_water = Linear(config['num_features'], config['dimension'])
 
        #M_in, M_out = config['dimension']/2, config['dimension']
        for i in range(config['depths']):
            ll = Sequential(Linear(B_in, 128), self.fn, self.dropout, Linear(128, M_in * M_out)) 
            _conv = NNConv(M_in, M_out, ll)
            self.conv.append(_conv)
            M_in, M_out = M_out, config['dimension']
        
        if config['mol']: # molecular features
            self.mol = Linear(196, config['dimension'])

        L_in, L_out = config['dimension'], int(config['dimension']/2)
        if self.config['water_interaction']:
            L_in, L_out = config['dimension']*2, int(config['dimension'])
        if config['mol']:
            L_in, L_out = config['dimension']*2, int(config['dimension'])
        fc = nn.Sequential(Linear(L_in, L_out), self.fn, self.dropout)
        self.out.append(fc)
        for i in range(config['NumOutLayers']-2):
            L_in, L_out = self.out[-1][0].out_features, int(self.out[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), self.fn, self.dropout)
            self.out.append(fc)
        last_fc = nn.Sequential(nn.Linear(L_out, 1), self.dropout)
        self.out.append(last_fc)
 
    def forward(self, data):
        #hidden_water = self.fn(self.f_water(self.embWater.cuda())) # (1, dimension)
        for layer in self.conv:
            data.x = self.fn(layer(data.x, data.edge_index, data.edge_attr)) # update atom embeddings
        if self.config['mol']:
            mol_feature_embed = self.fn(self.mol(data.mol.view(-1, 196)))
            x = torch.cat([x, mol_feature_embed], dim=1)
        
        if self.config['water_att']:
            hidden_water = self.fn(self.f_water(self.embWater.cuda()))
            data.x = self.gat(data.x, adj.cuda(), hidden_water)
        if self.config['water_interaction']:
            hidden_water = self.fn(self.f_water(self.embWater.cuda())) # Embed water molecules.SMILES: O; (1, dimension)
            interaction = torch.tanh(torch.mm(data.x, hidden_water.transpose(0, 1))) # (?, dimension) * (dimension, 1) = (?, 1)
            #interaction = torch.nn.functional.softmax(torch.mm(data.x, hidden_water.transpose(0, 1)), dim=0)
            #interaction = (interaction > 0.8).float()
            solute_after_interaction = torch.mm(interaction, hidden_water) # (?, 1) * (1, dimension) = (?, dimension)
            #water_after_interaction = torch.mm(interaction.t(), data.x) # (1, ?) * (?, dimension) = (1, dimension)
            data.x = torch.cat([data.x, solute_after_interaction], 1) # dimension increase to 3 folds
            #data.x = data.x * solute_after_interaction
        #else:
        #    data.x = self.gat(data.x, adj.cuda())

        x = self.pooling(data.x, data.batch, dim=0) # generate initial mol embedding using this pooling method
        for layer in self.out:
            x = layer(x) # update mol embeddings

        return x.view(-1), None

class knn(torch.nn.Module):
    def __init__(self, config):
        super(knn, self).__init__()
        self.fn = activation_func(config)
        self.pooling = pooling(config)
        self.config = config.copy()
        M_in, B_in, M_out = config['num_features'], config['num_bond_features'], int(config['dimension']/2)
        
        self.dropout = nn.Dropout(p=config['dropout'])
        self.conv = nn.ModuleList()
        #self.water = nn.ModuleList()
        self.out1 = nn.ModuleList()
        if self.config['taskType'] == 'multi':
            self.out2 = nn.ModuleList()
            
        if config['water_interaction']:
            self.embWater = torch.load('/beegfs/dz1061/gcn/chemGraph/data/WATER/graphs/water.pt')['x']
            self.f_water = Linear(config['num_features'], config['dimension'])
            #self.water.append(self.f_water_0)
            #self.f_water_1 = Linear(config['num_features'], config['dimension'])
            #self.water.append(self.f_water_1)
            #self.f_water_2 = Linear(config['num_features'], config['dimension'])
            #self.water.append(self.f_water_2)
            if config['dataset'] in ['ccdc_logp', 'logp', 'ccdc_sollogp']:
                self.embOct = torch.load('/beegfs/dz1061/datasets/logp/raw/octanol.pt')
                self.f_Oct = Linear(config['num_features'], config['dimension'])
        if config['mol']: # molecular features
            self.mol = Linear(196, config['dimension'])
        for _ in range(config['depths']):
            ll = Sequential(Linear(B_in, 128), self.fn, self.dropout, Linear(128, M_in * M_out))
            _conv = NNConv(M_in, M_out, ll)
            self.conv.append(_conv)
            M_in, M_out = M_out, config['dimension']
        
        self.conv4 = GraphConv(config['dimension'] + config['num_i_2'], config['dimension'])
        if config['water_interaction']:
            if config['InterByConcat']:
               self.conv4 = GraphConv(config['dimension']*2 + config['num_i_2'], config['dimension'])
               if config['dataset'] == 'ccdc_sollogp':
                   self.conv4_1 = GraphConv(config['dimension']*2 + config['num_i_2'], config['dimension'])
            if config['InterBySub']:
               self.conv4 = GraphConv(config['dimension'] + config['num_i_2'], config['dimension'])
               if config['dataset'] == 'ccdc_sollogp':
                   self.conv4_1 = GraphConv(config['dimension']*2 + config['num_i_2'], config['dimension'])
        self.conv5 = GraphConv(config['dimension'], config['dimension'])
        if config['dataset'] == 'ccdc_sollogp':
            self.conv5_1 = GraphConv(config['dimension'], config['dimension'])

        L_in, L_out = config['dimension']*2, int(config['dimension'])
        if config['mol']:
            L_in, L_out = config['dimension']*3, int(config['dimension'])
        if config['pooling'] == 'set2set':
            L_in, L_out = config['dimension']*4, int(config['dimension'])
        if config['water_interaction']:
            if config['InterByConcat']:
                L_in, L_out = config['dimension']*3, int(config['dimension'])
            if config['InterBySub']:
                L_in, L_out = config['dimension']*2, int(config['dimension'])
        fc = nn.Sequential(Linear(L_in, L_out), self.fn, self.dropout)
        self.out1.append(fc)
        if self.config['taskType'] == 'multi':
            fc = nn.Sequential(Linear(L_in, L_out), self.fn, self.dropout)
            self.out2.append(fc)
        for i in range(config['NumOutLayers']-2):
            L_in, L_out = self.out1[-1][0].out_features, int(self.out1[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), self.fn, self.dropout)
            self.out1.append(fc)
            if self.config['taskType'] == 'multi':
                fc = nn.Sequential(Linear(L_in, L_out), self.fn, self.dropout)
                self.out2.append(fc)
        last_fc = nn.Sequential(nn.Linear(L_out, 1))
        self.out1.append(last_fc)
        if self.config['taskType'] == 'multi':
            last_fc = nn.Sequential(nn.Linear(L_out, 1))
            self.out2.append(last_fc)
 
    def forward(self, data):
        for layer in self.conv:
            data.x = self.fn(layer(data.x, data.edge_index, data.edge_attr)) # update atom embeddings

        if self.config['water_interaction']:
            hidden_water = self.fn(self.f_water(self.embWater.cuda()))
            interaction = torch.tanh(torch.mm(data.x, hidden_water.transpose(0, 1)))
            solute_after_interaction = torch.mm(interaction, hidden_water)
            if self.config['dataset'] in ['ccdc_sollogp', 'ccdc_logp', 'logp']:
               hidden_oct = self.fn(self.f_Oct(self.embOct.cuda()))
               interaction = torch.tanh(torch.mm(data.x, hidden_oct.transpose(0, 1)))
               solute_after_oct = torch.mm(interaction, hidden_oct)
            if self.config['InterByConcat']:
               data.x = torch.cat([data.x, solute_after_interaction], 1) # dimension increase to 2 fold
               if self.config['dataset'] in ['ccdc_logp', 'logp']:
                   data.x = torch.cat([solute_after_oct, solute_after_interaction], 1) # dimension increase to 2 fold
               if self.config['dataset'] == 'ccdc_sollogp':
                   data.x1 = torch.cat([solute_after_oct, solute_after_interaction], 1)
            if self.config['InterBySub']:
               data.x = data.x - solute_after_interaction
               if self.config['dataset'] == ['ccdc_logp', 'logp']:
                  data.x = solute_after_oct - solute_after_interaction
               if self.config['dataset'] == 'ccdc_sollogp':
                  data.x1 = solute_after_oct - solute_after_interaction
        
        x_1 = self.pooling(data.x, data.batch, dim=0) # generate initial mol embedding using this pooling method
        if self.config['dataset'] == 'ccdc_sollogp':
            x_1_1 = self.pooling(data.x1, data.batch, dim=0)
         
        data.x = avg_pool(data.x, data.assignment_index_2)
        #data.x = torch.cat([data.x, data_iso], dim=1)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)
        data.x = self.fn(self.conv4(data.x, data.edge_index_2))
        data.x = self.fn(self.conv5(data.x, data.edge_index_2))
        if self.config['dataset'] == 'ccdc_sollogp':
            data.x1 = avg_pool(data.x1, data.assignment_index_2)
            #data.x = torch.cat([data.x, data_iso], dim=1)
            data.x1 = torch.cat([data.x1, data.iso_type_2], dim=1)
            data.x1 = self.fn(self.conv4_1(data.x1, data.edge_index_2))
            data.x1 = self.fn(self.conv5_1(data.x1, data.edge_index_2))
        
        x_2 = self.pooling(data.x, data.batch_2, dim=0)
        if self.config['dataset'] == 'ccdc_sollogp':
            x_2_1 = self.pooling(data.x1, data.batch_2, dim=0)
        
        #MolEmbed = x_2
        MolEmbed = torch.cat([x_1, x_2], dim=1)  # add x_0
        if self.config['dataset'] == 'ccdc_sollogp':
            MolEmbed_1 = torch.cat([x_1_1, x_2_1], dim=1)
        
        if self.config['mol']:
            mol_feature_embed = self.fn(self.mol(data.mol))
            MolEmbed = torch.cat([x_1, x_2, mol_feature_embed], dim=1)

        out1, out2 = MolEmbed, MolEmbed
        if self.config['dataset'] == 'ccdc_sollogp':
            out1, out2 = MolEmbed, MolEmbed_1
        for layer in self.out1:
            out1 = layer(out1)
        if self.config['taskType'] == 'multi':
            for layer in self.out2:
                 out2 = layer(out2)
            return out1.view(-1), out2.view(-1)
        return out1.view(-1), None

class knn_(torch.nn.Module):
    def __init__(self, config):
        super(knn_, self).__init__()
        self.fn = activation_func(config)
        self.pooling = pooling(config)
        self.config = config.copy()
        M_in, B_in, M_out = config['num_features'], config['num_bond_features'], int(config['dimension'])  # M_out=64 in order for layerwise water embeddings consistent 
        
        self.dropout = nn.Dropout(p=config['dropout'])
        self.conv = nn.ModuleList()
        self.water = nn.ModuleList()
        self.out1 = nn.ModuleList()
        if self.config['taskType'] == 'multi':
            self.out2 = nn.ModuleList()
        
        if config['water_interaction']:
            self.embWater = torch.load('/beegfs/dz1061/datasets/ws/raw/water.pt')
            self.f_water_0 = Linear(config['num_features'], int(config['dimension']))
            self.water.append(self.f_water_0)
            self.f_water_1 = Linear(config['num_features'], config['dimension'])
            self.water.append(self.f_water_1)
            self.f_water_2 = Linear(config['num_features'], config['dimension'])
            self.water.append(self.f_water_2)
            if config['dataset'] in ['ccdc_logp', 'logp']:
                self.embOct = torch.load('/beegfs/dz1061/datasets/logp/raw/octanol.pt')
                self.f_Oct = Linear(config['num_features'], config['dimension'])
        
        if config['mol']: # molecular features
            self.mol = Linear(196, config['dimension'])
        
        for _ in range(config['depths']):
            ll = Sequential(Linear(B_in, 128), self.fn, Linear(128, M_in * M_out))
            _conv = NNConv(M_in, M_out, ll)
            self.conv.append(_conv)
            M_in, M_out = M_out, config['dimension']
        
        self.fn_iso_type = Linear(config['num_i_2'], config['dimension'])
        #self.sig = torch.nn.Sigmoid()
        #self.conv4 = GraphConv(config['dimension'],  config['dimension'])
        self.conv4 = GraphConv(config['dimension'] + config['num_i_2'], config['dimension'])
        if config['water_interaction']:
            if config['InterByConcat']:
               self.conv4 = GraphConv(config['dimension']*2 + config['num_i_2'], config['dimension'])
            if config['InterBySub']:
               self.conv4 = GraphConv(config['dimension'] + config['num_i_2'], config['dimension'])
        self.conv5 = GraphConv(config['dimension'], config['dimension'])

        L_in, L_out = config['dimension']*2, int(config['dimension'])
        if config['mol']:
            L_in, L_out = config['dimension']*3, int(config['dimension'])
        if config['pooling'] == 'set2set':
            L_in, L_out = config['dimension']*4, int(config['dimension'])
        if config['water_interaction']:
            if config['InterByConcat']:
                L_in, L_out = config['dimension']*3, int(config['dimension'])
            if config['InterBySub']:
                L_in, L_out = config['dimension']*2, int(config['dimension'])
        fc = nn.Sequential(Linear(L_in, L_out), self.fn)
        self.out1.append(fc)
        if self.config['taskType'] == 'multi':
            fc = nn.Sequential(Linear(L_in, L_out), self.fn)
            self.out2.append(fc)
        for i in range(config['NumOutLayers']-2):
            L_in, L_out = self.out1[-1][0].out_features, int(self.out1[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), self.fn)
            self.out1.append(fc)
            if self.config['taskType'] == 'multi':
                fc = nn.Sequential(Linear(L_in, L_out), self.fn)
                self.out2.append(fc)
        last_fc = nn.Sequential(nn.Linear(L_out, 1))
        self.out1.append(last_fc)
        if self.config['taskType'] == 'multi':
            last_fc = nn.Sequential(nn.Linear(L_out, 1))
            self.out2.append(last_fc)
 
    def forward(self, data):
        interEmd = []
        for layer, f_water in zip(self.conv, self.water):
            data.x = self.fn(layer(data.x, data.edge_index, data.edge_attr)) # update atom embeddings
            hidden_water = self.fn(f_water(self.embWater.cuda()))
            interaction = torch.tanh(torch.mm(data.x, hidden_water.transpose(0, 1)))
            solute_after_interaction = torch.mm(interaction, hidden_water)
            data.x = solute_after_interaction
            interEmd.append(data.x)
        data.x = torch.mean(torch.stack(interEmd), axis=0) # mean of three interactions
        #data.x = torch.stack(interEmd).sum(dim=0)

        if not self.config['water_interaction']:
            hidden_water = self.fn(self.f_water(self.embWater.cuda()))
            interaction = torch.tanh(torch.mm(data.x, hidden_water.transpose(0, 1)))
            solute_after_interaction = torch.mm(interaction, hidden_water)
            if self.config['dataset'] in ['ccdc_logp', 'logp']:
               hidden_oct = self.fn(self.f_Oct(self.embOct.cuda()))
               interaction = torch.tanh(torch.mm(data.x, hidden_oct.transpose(0, 1)))
               solute_after_oct = torch.mm(interaction, hidden_oct)
            if self.config['InterByConcat']:
               data.x = torch.cat([data.x, solute_after_interaction], 1) # dimension increase to 2 fold
               if self.config['dataset'] in ['ccdc_logp', 'logp']:
                   data.x = torch.cat([solute_after_oct, solute_after_interaction], 1) # dimension increase to 2 fold
            if self.config['InterBySub']:
               data.x = data.x - solute_after_interaction
               if self.config['dataset'] == ['ccdc_logp', 'logp']:
                  data.x = solute_after_oct - solute_after_interaction
        x_1 = self.pooling(data.x, data.batch, dim=0) # generate initial mol embedding using this pooling method

        data.x = avg_pool(data.x, data.assignment_index_2)
        #data.x = torch.cat([data.x, data_iso], dim=1)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)
        data.x = self.fn(self.conv4(data.x, data.edge_index_2))
        data.x = self.fn(self.conv5(data.x, data.edge_index_2))

        
        x_2 = self.pooling(data.x, data.batch_2, dim=0)
        
        #MolEmbed = x_2
        MolEmbed = torch.cat([x_1, x_2], dim=1)  # add x_0
        
        if self.config['mol']:
            mol_feature_embed = self.fn(self.mol(data.mol))
            MolEmbed = torch.cat([x_1, x_2, mol_feature_embed], dim=1)

        out1, out2 = MolEmbed, MolEmbed
        for layer in self.out1:
            out1 = layer(out1)
        if self.config['taskType'] == 'multi':
            for layer in self.out2:
                 out2 = layer(out2)
            return out1.view(-1), out2.view(-1)
        return out1.view(-1), None

class LoopyBP(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, config):
        """Initializes the MPNEncoder.
        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param atom_messages: Whether to use atoms to pass messages instead of bonds.
        """
        super(LoopyBP, self).__init__()
        self.atom_fdim = config['atom_fdim']
        self.bond_fdim = config['bond_fdim']
        self.atom_messages = config['atom_messages']
        self.hidden_size = config['dimension']
        self.readout_size = config['outDim']
        self.depth = config['depths']
        self.device = config['device']
        self.config = config
        self.out = nn.ModuleList()
        self.dropout = nn.Dropout(p=config['dropout'])  

        # Activation
        self.fn = activation_func(config)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)
        
        fc0 = nn.Sequential(Linear(self.hidden_size, self.readout_size), self.fn, self.dropout)
        self.out.append(fc0)
        
        for i in range(config['NumOutLayers']-2):
            fc = nn.Sequential(Linear(self.readout_size, self.readout_size), self.fn, self.dropout)
            self.out.append(fc)
        last_fc = Linear(self.readout_size, 1)
        self.out.append(last_fc)

    def forward(self,
                mol_graph):
        """
        Encodes a batch of molecular graphs.
        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(atom_messages=self.atom_messages)
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)

        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.fn(input)  # num_bonds x hidden_size
        message = self.dropout(message)

        # Message passing
        for depth in range(self.depth - 1):

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.fn(input + message)  # num_bonds x hidden_size
            message = self.dropout(message)
            #message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.fn(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout(atom_hiddens)
        #atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        
        for layer in self.out:
            out = layer(mol_vecs)
            mol_vecs = out
        
        return out.view(-1), None  # num_molecules x 1

class LoopyBP_dropout(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, config):
        """Initializes the MPNEncoder.
        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param atom_messages: Whether to use atoms to pass messages instead of bonds.
        """
        super(LoopyBP_dropout, self).__init__()
        self.atom_fdim = config['atom_fdim']
        self.bond_fdim = config['bond_fdim']
        self.atom_messages = config['atom_messages']
        self.hidden_size = config['dimension']
        self.readout_size = config['outDim']
        self.depth = config['depths']
        self.device = config['device']
        self.config = config
        self.out = nn.ModuleList()
        self.dropout = nn.Dropout(p=config['dropout'])  
        self.fn = activation_func(config)
        self.weight_regularizer = config['weight_regularizer']
        self.dropout_regularizer = config['dropout_regularizer']
        
        ######### dropout layers. 
        self.ll_drop1 = NNDropout(level='graph', weight_regularizer=self.weight_regularizer,
                                          dropout_regularizer=self.dropout_regularizer)
        self.ll_drop2 = NNDropout(level='graph', weight_regularizer=self.weight_regularizer,
                                          dropout_regularizer=self.dropout_regularizer)
        if self.config['uncertainty'] == 'epistemic': 
            self.drop_mu = NNDropout(level='graph', weight_regularizer=self.weight_regularizer,
                                             dropout_regularizer=self.dropout_regularizer)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)
        
        fc0 = nn.Sequential(Linear(self.hidden_size, self.readout_size), self.fn, self.dropout)
        self.out.append(fc0)
        for i in range(config['NumOutLayers']-2):
            fc = nn.Sequential(Linear(self.readout_size, self.readout_size), self.fn, self.dropout)
            self.out.append(fc)

        self.last_fc_mu = nn.Linear(self.readout_size, 1)
        #self.out.append(last_fc_mu)
        self.last_fc_logvar = nn.Linear(self.readout_size, 1)
        #self.out.append(last_fc_logvar)

    def forward(self,
                mol_graph):
        """
        Encodes a batch of molecular graphs.
        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.config['uncertainty'] == 'epistemic':
            regularization = torch.empty(3, device='cuda')
        if self.config['uncertainty'] == 'aleatoric':
           regularization = torch.empty(2, device='cuda')

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(atom_messages=self.atom_messages)
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)

        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.fn(input)  # num_bonds x hidden_size
        message = self.dropout(message)

        # Message passing
        for depth in range(self.depth - 1):

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.fn(input + message)  # num_bonds x hidden_size
            message = self.dropout(message)
            #message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.fn(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout(atom_hiddens)
        #atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        #x = mol_vecs 
        for idx, layer, drop in zip(range(2), self.out, [self.ll_drop1, self.ll_drop2]):
            mol_vecs, regularization[idx] = drop(mol_vecs, layer)
        if self.config['uncertainty'] == 'epistemic':
            mean, regularization[2] = self.drop_mu(mol_vecs, self.last_fc_mu)
            return mean.view(-1)
        if self.config['uncertainty'] == 'aleatoric':
            mean = self.last_fc_mu(mol_vecs)
            log_var = self.last_fc_logvar(mol_vecs)
            return mean, log_var, regularization.sum()

class wlkernel(torch.nn.Module):
    def __init__(self, config):
        """Initializes the MPNEncoder.
        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param atom_messages: Whether to use atoms to pass messages instead of bonds.
        """
        super(wlkernel, self).__init__()
        self.atom_fdim = config['atom_fdim']
        self.bond_fdim = config['bond_fdim']
        self.atom_messages = config['atom_messages']
        self.hidden_size = config['dimension']
        self.readout_size = config['outDim']
        self.depth = config['depths']
        self.device = config['device']
        self.config = config
        self.out = nn.ModuleList()
        self.dropout = nn.Dropout(p=config['dropout'])
        #self.layers = []
        

        # Activation
        self.fn = activation_func(config)
        
        self.fc_atom00 = nn.Linear(self.atom_fdim, self.hidden_size)
        self.fc_atom01 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_atom02 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_atom03 = nn.Linear(self.hidden_size, self.hidden_size)

        self.fc_atom0 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_bond0 = nn.Linear(self.bond_fdim, self.hidden_size)
        self.fc_atom1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_bond1 = nn.Linear(self.bond_fdim, self.hidden_size)
        
        self.fc_lei = nn.Linear(self.hidden_size+self.bond_fdim, self.hidden_size)
        self.fc_new_lei = nn.Linear(self.hidden_size*2, self.hidden_size)
        
        fc0 = nn.Sequential(Linear(self.hidden_size, self.readout_size), self.fn, self.dropout)
        self.out.append(fc0) 
        for i in range(config['NumOutLayers']-2):
            fc = nn.Sequential(Linear(self.readout_size, self.readout_size), self.fn, self.dropout)
            self.out.append(fc)
        last_fc = Linear(self.readout_size, 1)
        self.out.append(last_fc)
        
    def forward(self, mol_graph):
        atom_features, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(atom_messages=self.atom_messages)
        atom_features, f_bonds, a2b, b2a, b2revb = atom_features.to(self.device), f_bonds.to(self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)
        
        a2a = mol_graph.get_a2a().to(self.device)
        #atom_features, f_bonds = atom_features.type(torch.float16), f_bonds.type(torch.float16)
        
        f_atoms = self.fc_atom00(atom_features)
        del atom_features
        del b2a
        del b2revb

        for _ in range(self.depth):
    
            nei_f_atom = index_select_ND(f_atoms, a2a)
            nei_f_bond = index_select_ND(f_bonds, a2b)
            
            h_nei_atom = self.fc_atom0(nei_f_atom)
            h_nei_bond = self.fc_bond0(nei_f_bond)
            g_nei_atom = self.fc_atom1(nei_f_atom)
            g_nei_bond = self.fc_bond1(nei_f_bond)
            
            g_self = self.fc_atom01(f_atoms)
            g_self = g_self.unsqueeze(1)
            g_nei = torch.sigmoid(g_nei_atom + g_nei_bond + g_self) * 10
            del g_nei_atom
            del g_nei_bond
            del g_self
            h_nei = g_nei * h_nei_atom * h_nei_bond
            del g_nei
            del h_nei_atom
            del h_nei_bond
            f_nei = torch.sum(h_nei, -2)
            del h_nei
            f_self = self.fc_atom02(f_atoms)
            atom_hiddens = f_nei * f_self
            del f_self
            del f_nei

            l_nei = torch.cat([nei_f_atom, nei_f_bond], -1)
            del nei_f_atom
            del nei_f_bond
            nei_label = F.relu(self.fc_lei(l_nei))
            del l_nei
            nei_label = torch.sum(nei_label, -2)
            new_label = torch.cat([f_atoms, nei_label], -1)
            del nei_label
            f_atoms = F.relu(self.fc_new_lei(new_label))
            del new_label 
        
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        
        for layer in self.out:
            out = layer(mol_vecs)
            mol_vecs = out
        
        return out.view(-1), None  # num_molecules x 1

class wlkernel_dropout(torch.nn.Module):
    def __init__(self, config):
        """Initializes the MPNEncoder.
        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param atom_messages: Whether to use atoms to pass messages instead of bonds.
        """
        super(wlkernel_dropout, self).__init__()
        self.atom_fdim = config['atom_fdim']
        self.bond_fdim = config['bond_fdim']
        self.atom_messages = config['atom_messages']
        self.hidden_size = config['dimension']
        self.readout_size = config['outDim']
        self.depth = config['depths']
        self.device = config['device']
        self.config = config
        self.out = nn.ModuleList()
        self.dropout = nn.Dropout(p=config['dropout'])
        self.weight_regularizer = config['weight_regularizer']
        self.dropout_regularizer = config['dropout_regularizer']
        
        ######### dropout layers. 
        self.ll_drop1 = NNDropout(level='graph', weight_regularizer=self.weight_regularizer,
                                          dropout_regularizer=self.dropout_regularizer)
        self.ll_drop2 = NNDropout(level='graph', weight_regularizer=self.weight_regularizer,
                                          dropout_regularizer=self.dropout_regularizer)
        if self.config['uncertainty'] == 'epistemic': 
            self.drop_mu = NNDropout(level='graph', weight_regularizer=self.weight_regularizer,
                                             dropout_regularizer=self.dropout_regularizer)
        #self.layers = []
        

        # Activation
        self.fn = activation_func(config)
        
        self.fc_atom00 = nn.Linear(self.atom_fdim, self.hidden_size)
        self.fc_atom01 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_atom02 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_atom03 = nn.Linear(self.hidden_size, self.hidden_size)

        self.fc_atom0 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_bond0 = nn.Linear(self.bond_fdim, self.hidden_size)
        self.fc_atom1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_bond1 = nn.Linear(self.bond_fdim, self.hidden_size)
        
        self.fc_lei = nn.Linear(self.hidden_size+self.bond_fdim, self.hidden_size)
        self.fc_new_lei = nn.Linear(self.hidden_size*2, self.hidden_size)
        
        fc0 = nn.Sequential(Linear(self.hidden_size, self.readout_size), self.fn, self.dropout)
        self.out.append(fc0) 
        for _ in range(config['NumOutLayers']-2):
            fc = nn.Sequential(Linear(self.readout_size, self.readout_size), self.fn, self.dropout)
            self.out.append(fc)
        self.last_fc_mu = nn.Linear(self.readout_size, 1)
        #self.out.append(last_fc_mu)
        self.last_fc_logvar = nn.Linear(self.readout_size, 1)
        
    def forward(self, mol_graph):
        if self.config['uncertainty'] == 'epistemic':
            regularization = torch.empty(3, device='cuda')
        if self.config['uncertainty'] == 'aleatoric':
           regularization = torch.empty(2, device='cuda')

        atom_features, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(atom_messages=self.atom_messages)
        atom_features, f_bonds, a2b, b2a, b2revb = atom_features.to(self.device), f_bonds.to(self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)
        
        a2a = mol_graph.get_a2a().to(self.device)
        #atom_features, f_bonds = atom_features.type(torch.float16), f_bonds.type(torch.float16)
        
        f_atoms = self.fc_atom00(atom_features)
        del atom_features
        del b2a
        del b2revb

        for _ in range(self.depth):
    
            nei_f_atom = index_select_ND(f_atoms, a2a)
            nei_f_bond = index_select_ND(f_bonds, a2b)
            
            h_nei_atom = self.fc_atom0(nei_f_atom)
            h_nei_bond = self.fc_bond0(nei_f_bond)
            g_nei_atom = self.fc_atom1(nei_f_atom)
            g_nei_bond = self.fc_bond1(nei_f_bond)
            
            g_self = self.fc_atom01(f_atoms)
            g_self = g_self.unsqueeze(1)
            g_nei = torch.sigmoid(g_nei_atom + g_nei_bond + g_self) * 10
            del g_nei_atom
            del g_nei_bond
            del g_self
            h_nei = g_nei * h_nei_atom * h_nei_bond
            del g_nei
            del h_nei_atom
            del h_nei_bond
            f_nei = torch.sum(h_nei, -2)
            del h_nei
            f_self = self.fc_atom02(f_atoms)
            atom_hiddens = f_nei * f_self
            del f_self
            del f_nei

            l_nei = torch.cat([nei_f_atom, nei_f_bond], -1)
            del nei_f_atom
            del nei_f_bond
            nei_label = F.relu(self.fc_lei(l_nei))
            del l_nei
            nei_label = torch.sum(nei_label, -2)
            new_label = torch.cat([f_atoms, nei_label], -1)
            del nei_label
            f_atoms = F.relu(self.fc_new_lei(new_label))
            del new_label 
        
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        x = mol_vecs 
        for idx, layer, drop in zip(range(2), self.out, [self.ll_drop1, self.ll_drop2]):
            x, regularization[idx] = drop(x, layer)
        if self.config['uncertainty'] == 'epistemic':
            mean, regularization[2] = self.drop_mu(x, self.last_fc_mu)
            return mean.view(-1)
        if self.config['uncertainty'] == 'aleatoric':
            mean = self.last_fc_mu(x)
            log_var = self.last_fc_logvar(x)
            return mean, log_var, regularization.sum()

class AdaConv(torch.nn.Module):
    def __init__(self, config):
        super(AdaConv, self).__init__()
        self.fn = activation_func(config)
        self.pooling = pooling(config)
        M_in, B_in, M_out = config['num_features'], config['num_bond_features'], int(config['dimension'])

        #self.dropout = nn.Dropout(p=config['dropout'])
        #self.conv = nn.ModuleList()
        #self.sagpool = nn.ModuleList()
        l1 = Sequential(Linear(B_in, 128), self.fn, Linear(128, M_in * M_out))
        self.conv1 = NNConv(M_in, M_out, l1)
        self.pool1 = SAGPooling(M_out, ratio=0.5)

        M_in, M_out = M_out, config['dimension']
        l2 = Sequential(Linear(B_in, 128), self.fn, Linear(128, M_in * M_out))
        self.conv2 = NNConv(M_in, M_out, l2)
        self.pool2 = SAGPooling(M_out, ratio=0.5)

        M_in, M_out = M_out, config['dimension']
        l3 = Sequential(Linear(B_in, 128), self.fn, Linear(128, M_in * M_out))
        self.conv3 = NNConv(M_in, M_out, l3)
        self.pool3 = SAGPooling(M_out, ratio=0.5)

        self.lin1 = torch.nn.Linear(config['dimension']*3, config['dimension'])
        self.lin2 = torch.nn.Linear(config['dimension'], config['dimension']//2)
        self.lin3 = torch.nn.Linear(config['dimension']//2, 1)
    
    def forward(self, data):
        data.x = self.fn(self.conv1(data.x, data.edge_index, data.edge_attr)) # update atom embeddings
        data.x, data.edge_index, data.edge_attr, data.batch, _ = self.pool1(data.x, data.edge_index, data.edge_attr, data.batch)
        x_1 = self.pooling(data.x, data.batch, dim=0)

        data.x = self.fn(self.conv2(data.x, data.edge_index, data.edge_attr)) # update atom embeddings
        data.x, data.edge_index, data.edge_attr, data.batch, _ = self.pool2(data.x, data.edge_index, data.edge_attr, data.batch)
        x_2 = self.pooling(data.x, data.batch, dim=0)

        data.x = self.fn(self.conv3(data.x, data.edge_index, data.edge_attr)) # update atom embeddings
        data.x, data.edge_index, data.edge_attr, data.batch, _ = self.pool3(data.x, data.edge_index, data.edge_attr, data.batch)
        x_3 = self.pooling(data.x, data.batch, dim=0)

        x = torch.cat([x_1, x_2, x_3], dim=1)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        
        return x.view(-1), None

class ConvATT(torch.nn.Module):
    def __init__(self, config):
        super(ConvATT, self).__init__()
        self.fn = activation_func(config)
        self.pooling = pooling(config)
        M_in, B_in, M_out = config['num_features'], config['num_bond_features'], int(config['dimension']/2)

        self.dropout = nn.Dropout(p=config['dropout'])
        self.conv = nn.ModuleList()
        #self.att = nn.ModuleList()
        
        for _ in range(config['depths']):
            ll = Sequential(Linear(B_in, 128), self.fn, self.dropout, Linear(128, M_in * M_out))
            _conv = NNConv(M_in, M_out, ll)
            self.conv.append(_conv)
            M_in, M_out = M_out, config['dimension']
        
        A_in, A_out = config['dimension'], config['dimension']
        #self.gat = GATConv(A_in, A_out, heads=3) fom pytorch geometric
        self.gat = GAT(nfeat=A_in, nhid=A_out, dropout=config['dropout'], \
                alpha=0.2, nheads=config['nheads'])
        self.fc = nn.Sequential(nn.Linear(config['dimension'], 1), self.dropout)
    
    def forward(self, data):
        adj = torch.zeros([data.num_nodes, data.num_nodes], dtype=torch.int32)
        i, j = data.edge_index
        adj[i,j] = 1

        for layer in self.conv:
            data.x = self.fn(layer(data.x, data.edge_index, data.edge_attr)) # update atom embeddings
        
        data.x = self.gat(data.x, adj.cuda())
        #data.x = self.gat(data.x, data.edge_index)
        x = scatter_add(data.x, data.batch, dim=0)
        x = self.fc(x)
        return x.view(-1)
 
class Net_1_2_3(torch.nn.Module):
    def __init__(self, data_set, num_i_2, num_i_3):
        super(Net_1_2_3, self).__init__()
        M_in, M_out = data_set.num_features, 32
        nn1 = Sequential(Linear(7, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv1 = NNConv(M_in, M_out, nn1)

        M_in, M_out = M_out, 64
        nn2 = Sequential(Linear(7, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv2 = NNConv(M_in, M_out, nn2)

        M_in, M_out = M_out, 64
        nn3 = Sequential(Linear(7, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv3 = NNConv(M_in, M_out, nn3)

        self.conv4 = GraphConv(64 + num_i_2, 64)
        self.conv5 = GraphConv(64, 64)

        self.conv6 = GraphConv(64 + num_i_3, 64)
        self.conv7 = GraphConv(64, 64)

        self.fc1 = torch.nn.Linear(6 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        
    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        x = data.x
        x_1 = scatter_add(data.x, data.batch, dim=0)

        data.x = avg_pool(x, data.assignment_index_2)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)

        data.x = F.elu(self.conv4(data.x, data.edge_index_2))
        data.x = F.elu(self.conv5(data.x, data.edge_index_2))
        x_2 = scatter_add(data.x, data.batch_2, dim=0)

        data.x = avg_pool(x, data.assignment_index_3)
        data.x = torch.cat([data.x, data.iso_type_3], dim=1)

        data.x = F.elu(self.conv6(data.x, data.edge_index_3))
        data.x = F.elu(self.conv7(data.x, data.edge_index_3))
        x_3 = scatter_add(data.x, data.batch_3, dim=0)

        x = torch.cat([x_1, x_2, x_3], dim=1)
        x = torch.cat([x, x], dim=1) # for melting point
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)

class Net_1_3(torch.nn.Module):
    def __init__(self, data_set, num_i_3):
        super(Net_1_3, self).__init__()
        M_in, M_out = data_set.num_features, 32
        nn1 = Sequential(Linear(7, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv1 = NNConv(M_in, M_out, nn1)

        M_in, M_out = M_out, 64
        nn2 = Sequential(Linear(7, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv2 = NNConv(M_in, M_out, nn2)

        M_in, M_out = M_out, 64
        nn3 = Sequential(Linear(7, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv3 = NNConv(M_in, M_out, nn3)

        self.conv6 = GraphConv(64 + num_i_3, 64)
        self.conv7 = GraphConv(64, 64)

        self.fc1 = torch.nn.Linear(2 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
    
    def forward(self, data):
        data.x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = F.relu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = F.relu(self.conv3(data.x, data.edge_index, data.edge_attr))
        x_1 = scatter_add(data.x, data.batch, dim=0)

        data.x = avg_pool(data.x, data.assignment_index_3)
        data.x = torch.cat([data.x, data.iso_type_3], dim=1)

        data.x = F.relu(self.conv6(data.x, data.edge_index_3))
        data.x = F.relu(self.conv7(data.x, data.edge_index_3))
        x_3 = scatter_add(data.x, data.batch_3, dim=0)

        x = torch.cat([x_1, x_3], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)

class ConvGRU(torch.nn.Module):
    def __init__(self, config):
        super(ConvGRU, self).__init__()
        self.fn = activation_func(config)
        self.pooling = pooling(config)
        M_in, B_in, M_out = config['num_features'], config['num_bond_features'], int(config['dimension']/2)
 
        self.dropout = nn.Dropout(p=config['dropout'])
        self.conv = nn.ModuleList()
        self.gru = nn.ModuleList()
        self.out = nn.ModuleList()

        self.fc0 = torch.nn.Linear(M_in, M_out)
        self.gru0 = GRU(M_out, config['dimension'], 1)
        M_in, M_out = config['dimension'], config['dimension']
        for _ in range(config['depths']): 
            ll = Sequential(Linear(B_in, 128), self.fn, self.dropout, Linear(128, M_in * M_out))
            _conv = NNConv(M_in, M_out, ll)
            self.conv.append(_conv)
            M_in, M_out = config['dimension'], config['dimension']
        
        G_in, G_out = config['dimension'], config['dimension']
        for _ in range(config['depths']):
            _gru = GRU(G_in, G_out, 1)
            self.gru.append(_gru)
        
        L_in, L_out = config['dimension'], config['dimension']
        fc = nn.Sequential(Linear(L_in, L_out), self.fn, self.dropout)
        self.out.append(fc)
        if config['NumOutLayers'] == 1:
            last_fc = nn.Sequential(nn.Linear(config['dimension'], 1), self.dropout)
            self.out.append(last_fc)
        elif config['NumOutLayers'] == 2:
            last_fc1 = nn.Sequential(nn.Linear(config['dimension'], 32), self.dropout)
            self.out.append(last_fc1)
            last_fc2 = nn.Sequential(nn.Linear(32, 1), self.dropout)
            self.out.append(last_fc2)
        else:
            for i in range(config['NumOutLayers']-2, 0, -1):
                L_in, L_out = L_out, 32*i
                fc = nn.Sequential(Linear(L_in, L_out), self.fn, self.dropout)
                self.out.append(fc)
            last_fc = nn.Sequential(nn.Linear(32, 1), self.dropout)
            self.out.append(last_fc)
 
    def forward(self, data):
        x = self.fn(self.fc0(data.x)) # Initialize atom embedding
        _, h_ = self.gru0(torch.unsqueeze(x, 0), None) # Initialize GRU with hidden size equal to zero

        for conv_layer, gru_layer in zip(self.conv, self.gru):
            #print(conv_layer, gru_layer)
            x = self.fn(conv_layer(torch.squeeze(h_), data.edge_index, data.edge_attr))# update atom embeddings
            out_, h_ = gru_layer(torch.unsqueeze(x, 0), h_)

        MolEmbedding = self.pooling(torch.squeeze(out_), data.batch, dim=0) # generate initial mol embedding using this pooling method
        for layer in self.out:
            MolEmbedding = layer(MolEmbedding) # update mol embeddings

        return MolEmbedding.view(-1)



class NNConvGAT(torch.nn.Module):
    def __init__(self, config):
        super(NNConvGAT, self).__init__()
        self.fn = activation_func(config)
        self.pooling = pooling(config)
        self.config = config.copy()
        M_in, B_in, M_out = config['num_features'], config['num_bond_features'], int(config['dimension']/2)
        
        self.dropout = nn.Dropout(p=config['dropout'])
        self.conv = nn.ModuleList()
        self.out = nn.ModuleList()
        
        if config['water_att']:
            self.embWater = torch.load('/beegfs/dz1061/datasets/ws/raw/water.pt')
            self.f_water = Linear(config['num_features'], config['dimension']) 
        if config['water_interaction']:
            self.embWater = torch.load('/beegfs/dz1061/datasets/ws/raw/water.pt')
            self.f_water = Linear(config['num_features'], config['dimension'])

        for _ in range(config['depths']):
            ll = Sequential(Linear(B_in, 128), self.fn, self.dropout, Linear(128, M_in * M_out))
            _conv = NNConv(M_in, M_out, ll)
            self.conv.append(_conv)
            M_in, M_out = M_out, config['dimension']
        
        A_in, A_out = config['dimension'] + config['num_i_2'], config['dimension']
        self.gat = GAT(nfeat=A_in, nhid=A_out, dropout=config['dropout'], \
                alpha=0.2, nheads=config['nheads'])
        
        L_in, L_out = config['dimension']*2, int(config['dimension'])
        fc = nn.Sequential(Linear(L_in, L_out), self.fn, self.dropout)
        self.out.append(fc)
        for i in range(config['NumOutLayers']-2):
            L_in, L_out = self.out[-1][0].out_features, int(self.out[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), self.fn, self.dropout)
            self.out.append(fc)
        last_fc = nn.Sequential(nn.Linear(L_out, 1), self.dropout)
        self.out.append(last_fc)
 
    def forward(self, data):
        for layer in self.conv:
            data.x = self.fn(layer(data.x, data.edge_index, data.edge_attr)) # update atom embeddings
        x_1 = self.pooling(data.x, data.batch, dim=0) # generate initial mol embedding using this pooling method

        data.x = avg_pool(data.x, data.assignment_index_2)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)
        #print(data.x.shape, data.batch.shape)
        iso_num_nodes = data.x.shape[0]
        adj = torch.zeros([iso_num_nodes, iso_num_nodes], dtype=torch.int32)
        i, j = data.edge_index_2
        adj[i,j] = 1
        
        if self.config['water_att']:
            hidden_water = self.fn(self.f_water(self.embWater.cuda()))
            data.x = self.gat(data.x, adj.cuda(), hidden_water) 
        if self.config['water_interaction']:
            hidden_water = self.fn(self.f_water(self.embWater.cuda()))
            interaction = torch.tanh(torch.mm(data.x, hidden_water.transpose(0, 1)))
            solute_after_interaction = torch.mm(interaction, hidden_water)
            data.x = torch.cat([data.x, solute_after_interaction], 1) # dimension increase to 2 folds  
        else:
            data.x = self.gat(data.x, adj.cuda())
        #data.x = self.fn(self.conv4(data.x, data.edge_index_2))
        #data.x = self.fn(self.conv5(data.x, data.edge_index_2))
        x_2 = self.pooling(data.x, data.batch_2, dim=0)
        
        #x = torch.add(x_1, x_2)
        x = torch.cat([x_1, x_2], dim=1)  # add x_0

        for layer in self.out:
            x = layer(x)
        return x.view(-1), None


class knn_GradCAM(torch.nn.Module):
    def __init__(self, config):
        super(knn_GradCAM, self).__init__()
        self.fn = activation_func(config)
        self.pooling = pooling(config)
        self.config = config.copy()
        M_in, B_in, M_out = config['num_features'], config['num_bond_features'], int(config['dimension']/2)

        self.dropout = nn.Dropout(p=config['dropout'])
        self.conv = nn.ModuleList()
        self.out1 = nn.ModuleList()
        if self.config['taskType'] == 'multi':
            self.out2 = nn.ModuleList()
        
        if config['water_interaction']:
            self.embWater = torch.load('/beegfs/dz1061/datasets/ws/raw/water.pt')
            self.f_water = Linear(config['num_features'], config['dimension'])
            self.f_water_2 = Linear(config['num_features'], config['dimension'])
        if config['mol']: # molecular features
            self.mol = Linear(196, config['dimension'])
        for _ in range(config['depths']):
            ll = Sequential(Linear(B_in, 128), self.fn, self.dropout, Linear(128, M_in * M_out))
            _conv = NNConv(M_in, M_out, ll)
            self.conv.append(_conv)
            M_in, M_out = M_out, config['dimension']

        #self.fn_iso_type = Linear(config['num_i_2'], 1)
        #self.sig = torch.nn.Sigmoid()
        #self.conv4 = GraphConv(config['dimension'],  config['dimension'])
        self.conv4 = GraphConv(config['dimension'] + config['num_i_2'], config['dimension'])
        if config['water_interaction']:
            self.conv4 = GraphConv(config['dimension']*2 + config['num_i_2'], config['dimension'])
        self.conv5 = GraphConv(config['dimension'], config['dimension'])

        L_in, L_out = config['dimension']*2, int(config['dimension'])
        if config['water_interaction']:
            L_in, L_out = config['dimension']*3, int(config['dimension'])
        if config['mol']:
            L_in, L_out = config['dimension']*3, int(config['dimension'])
        fc = nn.Sequential(Linear(L_in, L_out), self.fn, self.dropout)
        self.out1.append(fc)
        if self.config['taskType'] == 'multi':
            self.out2.append(fc)
        for i in range(config['NumOutLayers']-2):
            L_in, L_out = self.out1[-1][0].out_features, int(self.out1[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), self.fn, self.dropout)
            self.out1.append(fc)
            if self.config['taskType'] == 'multi':
                self.out2.append(fc)
        last_fc = nn.Sequential(nn.Linear(L_out, 1), self.dropout)
        self.out1.append(last_fc)
        if self.config['taskType'] == 'multi':
            self.out2.append(last_fc)
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, data):
        for layer in self.conv:
            data.x = self.fn(layer(data.x, data.edge_index, data.edge_attr)) # update atom embeddings
        return data.x

    def forward(self, data, gradcam=False):
        for index, layer in enumerate(self.conv):
            data.x = self.fn(layer(data.x, data.edge_index, data.edge_attr)) # update atom embeddings
            if gradcam and index == self.config['depths'] - 1:
                data.x.register_hook(self.activations_hook)
        
        if self.config['water_interaction']:
            hidden_water = self.fn(self.f_water(self.embWater.cuda()))
            interaction = torch.tanh(torch.mm(data.x, hidden_water.transpose(0, 1)))
            solute_after_interaction = torch.mm(interaction, hidden_water)
            data.x = torch.cat([data.x, solute_after_interaction], 1) # dimension increase to 2 fold
        x_1 = self.pooling(data.x, data.batch, dim=0) # generate initial mol embedding using this pooling method

        data.x = avg_pool(data.x, data.assignment_index_2)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)
        data.x = self.fn(self.conv4(data.x, data.edge_index_2))
        data.x = self.fn(self.conv5(data.x, data.edge_index_2))
        x_2 = self.pooling(data.x, data.batch_2, dim=0)

        
        MolEmbed = torch.cat([x_1, x_2], dim=1)

        if self.config['mol']:
            mol_feature_embed = self.fn(self.mol(data.mol))
            MolEmbed = torch.cat([x_1, x_2, mol_feature_embed], dim=1)

        out1, out2 = MolEmbed, MolEmbed
        for layer in self.out1:
            out1 = layer(out1)
        if self.config['taskType'] == 'multi':
            for layer in self.out2:
                 out2 = layer(out2)
            return out1.view(-1), out2.view(-1)
        return out1.view(-1), None

class knn_GradCAM_rev(torch.nn.Module):
    def __init__(self, config):
        super(knn_GradCAM_rev, self).__init__()
        self.fn = activation_func(config)
        self.pooling = pooling(config)
        self.config = config.copy()
        M_in, B_in, M_out = config['num_features'], config['num_bond_features'], int(config['dimension']/2)

        self.dropout = nn.Dropout(p=config['dropout'])
        self.conv = nn.ModuleList()
        self.out1 = nn.ModuleList()
        if self.config['taskType'] == 'multi':
            self.out2 = nn.ModuleList()
        
        if config['water_interaction']:
            self.embWater = torch.load('/beegfs/dz1061/datasets/ws/raw/water.pt')
            self.f_water = Linear(config['num_features'], config['dimension'])
            self.f_water_2 = Linear(config['num_features'], config['dimension'])
            if config['dataset'] in ['logp']:
                self.embOctanol = torch.load('/beegfs/dz1061/datasets/logp/raw/octanol.pt')
                self.f_octanol = Linear(config['num_features'], config['dimension'])
        if config['mol']: # molecular features
            self.mol = Linear(196, config['dimension'])
        for _ in range(config['depths']):
            ll = Sequential(Linear(B_in, 128), self.fn, self.dropout, Linear(128, M_in * M_out))
            _conv = NNConv(M_in, M_out, ll)
            self.conv.append(_conv)
            M_in, M_out = M_out, config['dimension']

        #self.fn_iso_type = Linear(config['num_i_2'], 1)
        #self.sig = torch.nn.Sigmoid()
        #self.conv4 = GraphConv(config['dimension'],  config['dimension'])
        self.conv4 = GraphConv(config['dimension'] + config['num_i_2'], config['dimension'])
        if config['water_interaction']:
            self.conv4 = GraphConv(config['dimension']*2 + config['num_i_2'], config['dimension'])
            if config['dataset'] in ['logp']:
                self.conv4 = GraphConv(config['dimension']*4 + config['num_i_2'], config['dimension'])
        self.conv5 = GraphConv(config['dimension'], config['dimension'])

        L_in, L_out = config['dimension']*2, int(config['dimension'])
        if config['water_interaction']:
            L_in, L_out = config['dimension']*3, int(config['dimension'])
            if config['dataset'] in ['logp']:
                L_in, L_out = config['dimension']*5, int(config['dimension'])
        if config['mol']:
            L_in, L_out = config['dimension']*3, int(config['dimension'])
        fc = nn.Sequential(Linear(L_in, L_out), self.fn, self.dropout)
        self.out1.append(fc)
        if self.config['taskType'] == 'multi':
            self.out2.append(fc)
        for i in range(config['NumOutLayers']-2):
            L_in, L_out = self.out1[-1][0].out_features, int(self.out1[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), self.fn, self.dropout)
            self.out1.append(fc)
            if self.config['taskType'] == 'multi':
                self.out2.append(fc)
        last_fc = nn.Sequential(nn.Linear(L_out, 1), self.dropout)
        self.out1.append(last_fc)
        if self.config['taskType'] == 'multi':
            self.out2.append(last_fc)
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, data):
        for layer in self.conv:
            data.x = self.fn(layer(data.x, data.edge_index, data.edge_attr)) # update atom embeddings
        return data.x

    def forward(self, data, gradcam=False):
        for index, layer in enumerate(self.conv):
            data.x = self.fn(layer(data.x, data.edge_index, data.edge_attr)) # update atom embeddings
            if gradcam and index == self.config['depths'] - 1:
                data.x.register_hook(self.activations_hook)
        
        if self.config['water_interaction']:
            hidden_water = self.fn(self.f_water(self.embWater.cuda()))
            interaction = torch.tanh(torch.mm(data.x, hidden_water.transpose(0, 1)))
            solute_after_interaction = torch.mm(interaction, hidden_water)
            data.x_water = torch.cat([data.x, solute_after_interaction], 1) # dimension increase to 2 fold
            #data.x = data.x_water
            if self.config['dataset'] in ['logp']:
                hidden_octanol = self.fn(self.f_octanol(torch.FloatTensor(self.embOctanol).cuda()))
                interaction = torch.zeros(data.x.shape[0], hidden_octanol.shape[0]).cuda()
                for i, solute_row in enumerate(data.x):
                     for j, solvent_row in enumerate(hidden_octanol):
                           interaction[i][j] = torch.sum(torch.mul(solute_row, solvent_row))
                interaction = torch.tanh(interaction)
                solute_after_interaction = torch.mm(interaction, hidden_octanol)
                data.x_octanol = torch.cat([data.x, solute_after_interaction], 1) # dimension increase to 2 fold
                data.x = torch.cat([data.x_water, data.x_octanol], 1) # dimension increase to 4 fold
            else:
                data.x = data.x_water
        x_1 = self.pooling(data.x, data.batch, dim=0) # generate initial mol embedding using this pooling method

        data.x = avg_pool(data.x, data.assignment_index_2)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)
        data.x = self.fn(self.conv4(data.x, data.edge_index_2))
        data.x = self.fn(self.conv5(data.x, data.edge_index_2))
        x_2 = self.pooling(data.x, data.batch_2, dim=0)

        
        MolEmbed = torch.cat([x_1, x_2], dim=1)

        if self.config['mol']:
            mol_feature_embed = self.fn(self.mol(data.mol))
            MolEmbed = torch.cat([x_1, x_2, mol_feature_embed], dim=1)

        out1, out2 = MolEmbed, MolEmbed
        for layer in self.out1:
            out1 = layer(out1)
        if self.config['taskType'] == 'multi':
            for layer in self.out2:
                 out2 = layer(out2)
            return out1.view(-1), out2.view(-1)
        if self.config['water_interaction']:
            return out1.view(-1), interaction
        return out1.view(-1), None

class knn_dropout(nn.Module):
    def __init__(self, config):
        super(knn_dropout, self).__init__()
        self.fn = activation_func(config)
        self.pooling = pooling(config)
        self.config = config.copy()
        M_in, B_in, M_out = config['num_features'], config['num_bond_features'], int(config['dimension']/2)
        
        #self.dropout = nn.Dropout(p=config['dropout'])
        self.conv = nn.ModuleList()
        self.out = nn.ModuleList()
        self.weight_regularizer = config['weight_regularizer']
        self.dropout_regularizer = config['dropout_regularizer']
        #nnconv_drop = nn.ModuleList()

        ######### dropout layers. 
        self.ll_drop1 = NNDropout(level='graph', weight_regularizer=self.weight_regularizer,
                                          dropout_regularizer=self.dropout_regularizer)
        self.ll_drop2 = NNDropout(level='graph', weight_regularizer=self.weight_regularizer,
                                          dropout_regularizer=self.dropout_regularizer)
        if self.config['uncertainty'] == 'epistemic': 
            self.drop_mu = NNDropout(level='graph', weight_regularizer=self.weight_regularizer,
                                             dropout_regularizer=self.dropout_regularizer)
    
        ######### atom embedding updating.
        for _ in range(config['depths']):
            ll = Sequential(Linear(B_in, 128), self.fn, Linear(128, M_in * M_out), self.fn)
            _conv = NNConv(M_in, M_out, ll)
            
            self.conv.append(_conv)
            M_in, M_out = M_out, config['dimension']

        ######### atom-pair sudo-node embedding.
        self.conv4 = GraphConv(config['dimension'] + config['num_i_2'], config['dimension'])
        self.conv5 = GraphConv(config['dimension'], config['dimension'])
        
        ######## readout layers.
        L_in, L_out = config['dimension']*2, int(config['dimension'])
        fc = nn.Sequential(Linear(L_in, L_out), self.fn)
        self.out.append(fc)
        for i in range(config['NumOutLayers']-2):
            L_in, L_out = self.out[-1][0].out_features, int(self.out[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), self.fn)
            self.out.append(fc)
        
        self.last_fc_mu = nn.Linear(L_out, 1)
        #self.out.append(last_fc_mu)
        self.last_fc_logvar = nn.Linear(L_out, 1)
        #self.out.append(last_fc_logvar)

    def forward(self, data):
        if self.config['uncertainty'] == 'epistemic':
            regularization = torch.empty(3, device='cuda')
        if self.config['uncertainty'] == 'aleatoric':
           regularization = torch.empty(2, device='cuda')
        
        for layer in self.conv:
            data.x = self.fn(layer(data.x, data.edge_index, data.edge_attr)) # update atom embeddings

        x_1 = self.pooling(data.x, data.batch, dim=0) # generate initial mol embedding using this pooling method

        data.x = avg_pool(data.x, data.assignment_index_2)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)
        data.x = self.fn(self.conv4(data.x, data.edge_index_2))
        data.x = self.fn(self.conv5(data.x, data.edge_index_2))

        x_2 = self.pooling(data.x, data.batch_2, dim=0)
        
        x = torch.cat([x_1, x_2], dim=1)  # add x_0

        for idx, layer, drop in zip(range(2), self.out, [self.ll_drop1, self.ll_drop2]):
            x, regularization[idx] = drop(x, layer)
        if self.config['uncertainty'] == 'epistemic':
            mean, regularization[2] = self.drop_mu(x, self.last_fc_mu)
            return mean.squeeze()
        if self.config['uncertainty'] == 'aleatoric':
            mean = self.last_fc_mu(x)
            log_var = self.last_fc_logvar(x)
            return mean, log_var, regularization.sum()


############# SWAG model ###############
class Base:
    base = knn
    args = list()
    kwargs = dict()
    
class Base1:
    base = LoopyBP
    args = list()
    kwargs = dict()

class Base2: 
    base = wlkernel
    args = list()
    kwargs = dict()

class knn_swag(Base):
    pass

class LoopyBP_swag(Base1):
    pass

class wlkernel_swag(Base2):
    pass


########### VAE #####################################################################################
class VAE(nn.Module):
    def __init__(self, config):
        '''
        https://pytorch.org/docs/master/_modules/torch/nn/modules/transformer.html#TransformerEncoder
        '''
        super(VAE, self).__init__()
        self.config = config
        self.src_embedding = nn.Embedding(self.config['vocab_size'], self.config['dimension'])
        self.trg_embedding = nn.Embedding(self.config['vocab_size'], self.config['dimension'])

        self.pe = PositionalEncoding(self.config['dropout'], self.config['dimension'])
        
        encoder_layer = nn.TransformerEncoderLayer(self.config['dimension'], self.config['numEncoLayers']) 
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config['numEncoders'])
        decoder_layer = nn.TransformerDecoderLayer(self.config['dimension'], self.config['numDecoLayers'])
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.config['numDecoders'])
        self.norm_layer = LayerNorm(self.config['varDimen'])
        
        self.fc00 = nn.Linear(self.config['dimension'], self.config['dimension'])
        self.fc01 =  nn.Linear(self.config['dimension'], self.config['dimension'])
        self.fc0 = nn.Linear(self.config['dimension'], self.config['varDimen'])
        self.fc1 = nn.Linear(self.config['dimension'], self.config['varDimen'])
        self.fc11 = nn.Linear(self.config['varDimen'], self.config['dimension'])
        self.fc2 = nn.Linear(self.config['dimension'], self.config['vocab_size'])
        self.fc3 = nn.Linear(self.config['dimension'], self.config['dimension'])
        

    def _generate_square_subsequent_mask(self, sz):
        # sz: length of seqeunce
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def reparameterize(self, mu, logvar):
        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation
            std = logvar.mul(0.5).exp_()  # type: Variable
            eps = Variable(std.data.new(std.size()).normal_())
            sample_z = eps.mul(std).add_(mu)

            return sample_z
        else:
            return mu
    
    def encode(self, x):
        srcEmbed = self.src_embedding(x.SRC)
        encEmbed = self.pe(srcEmbed)
        self.src_key_mask = (x.SRC == 1).to('cuda')
        #seq_size = x.SRC.size()[1]
        #self.src_atten_mask = self._generate_square_subsequent_mask(seq_size)
        output_encoder = self.encoder(src=encEmbed.transpose(0,1), \
                                      #mask=self.src_atten_mask, \
                              src_key_padding_mask=self.src_key_mask)
        mu, logvar = F.relu(self.fc00(output_encoder.transpose(0,1))), \
                     F.relu(self.fc01(output_encoder.transpose(0,1)))
        mu, logvar = self.fc0(mu), self.fc1(logvar)
        
        return mu, logvar
    
    def decode(self, x, z):
        seq_size = x.TRG.size()[1]
        normEmd = self.norm_layer(z)
        toDecoderEmbed = F.relu(self.fc11(normEmd))
        trgEmbed = self.trg_embedding(x.TRG)
        decEmbed = self.pe(trgEmbed)
        self.tgt_key_mask = (x.TRG == 1).to('cuda')
        self.tgt_atten_mask = self._generate_square_subsequent_mask(seq_size).to('cuda')
        output_decoder = self.decoder(tgt=decEmbed.transpose(0,1), \
                             tgt_mask=self.tgt_atten_mask, \
                             memory=toDecoderEmbed.transpose(0,1), \
                             #memory_mask=self.src_atten_mask, \
                             tgt_key_padding_mask = self.tgt_key_mask, \
                             memory_key_padding_mask=self.src_key_mask)
        
        return output_decoder
    
    def get_z(self, x):
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar)

    def forward(self, data):
        mu, logvar = self.encode(data)
        #print(mu.shape, logvar.shape)
        sample_z = self.reparameterize(mu, logvar).to('cuda')
        output_decoder = self.decode(data, sample_z)
        
        
        output = F.relu(self.fc3(output_decoder.transpose(0,1)))
        output = self.fc2(output)

        return output.view(-1, self.config['vocab_size']), mu, logvar
