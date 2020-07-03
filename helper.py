import os, random, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, ELU, Tanh, SELU
from torch_scatter import scatter_mean, scatter_add, scatter_max
import torch_geometric.transforms as T

import numpy as np

################### Configuration setting names ############################
BASIC = ['dataset', 'running_path', 'model', 'normalize', 'batch_size', 'dimension', \
         'dropout', 'act_fn' , 'weights', 'seed', 'optimizer', 'loss', 'metrics', 'lr', 'lr_style', \
         'epochs', 'early_stopping', 'train_type', 'taskType', 'train_size', 'val_size', 'num_features', \
          'num_bond_features', 'data_path', 'efgs', 'water_interaction', 'InterByConcat', 'InterBySub', 'mol']
GNN = ['depths', 'NumOutLayers', 'pooling', 'num_features', 'num_i_2']
GNNVariants = ['efgs', 'water_interaction', 'InterByConcat', 'InterBySub', 'mol', 'num_i_2']
VAE_opts = ['vocab', 'numEncoLayers', 'numDecoLayers', 'numEncoders', 'numDecoders', 'varDimen']
UQ = ['depths', 'NumOutLayers', 'pooling', 'num_features', 'num_i_2', 'uncertainty', 'uncertainty_method', 'swag_start', 'num_i_2', 'atom_fdim', 'bond_fdim', 'atom_messages', 'outDim', 'weight_regularizer', 'dropout_regularizer']
TRANSFER = ['depths', 'NumOutLayers', 'pooling', 'num_features', 'num_i_2', 'transfer_from', 'pre_trained_path', 'pre_trained_model']
###############################################################################

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

def heteroscedastic_loss(true, mean, log_var):
    '''
    dropout loss
    '''
    precision = torch.exp(-log_var)
    return torch.mean(torch.sum(0.5*precision * (true - mean.reshape(-1,))**2 + 0.5*log_var, 1), 0)

def vae_loss(recon_x, x, mu, logvar, config):
    seq_size = x.size()[0]/config['batch_size']
    CELoss = F.cross_entropy(recon_x, x, ignore_index=1)  
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= seq_size
    return CELoss + KLD

def get_loss_fn(name):
    if name == 'l1':
        return F.l1_loss
    if name == 'l2':
        return F.mse_loss
    if name == 'smooth_l1':
        return F.smooth_l1_loss
    if name == 'dropout':
        return heteroscedastic_loss
    if name == 'vae':
       return vae_loss

def get_metrics_fn(name):
    if name == 'l1':
        return F.l1_loss
    if name == 'l2':
        return F.mse_loss
    if name == 'smooth_l1':
        return F.smooth_l1_loss

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def activation_func(config):
    name = config['act_fn']
    if name == 'relu':
       return ReLU()
    if name == 'elu':
       return ELU()
    if name == 'leaky_relu':
       return LeakyReLU()
    if name == 'tanh':
       return Tanh()
    if name == 'selu':
       return SELU()

def he_norm(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
       torch.nn.init.kaiming_normal_(m.weight.data)

def xavier_norm(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
       torch.nn.init.xavier_normal_(m.weight.data)

def he_uniform(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
       torch.nn.init.kaiming_uniform_(m.weight.data)

def xavier_uniform(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
       torch.nn.init.xavier_uniform_(m.weight.data)

def init_weights(model, config):
    name = config['weights']
    if name == 'he_norm':
       model.apply(he_norm)
    if name == 'xavier_norm':
       model.apply(xavier_norm)
    if name == 'he_uni':
       model.apply(he_uniform)
    if name == 'xavier_uni':
       model.apply(xavier_uniform)
    return model

def pooling(config):
    name = config['pooling']
    if name == 'add':
       return scatter_add
    if name == 'mean':
       return scatter_mean
    if name == 'max':
       return scatter_max
    if name == 'set2set':
       config['set2setSteps'] = 3
       return set2set(config['dimension']*2, config['set2setSteps'])

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

def createResultsFile(this_dic, name='data.txt'):
    with open(os.path.join(this_dic['running_path'], name), 'w') as f:
            if this_dic['dataset'] == 'qm9':
                header = 'Epoch' + '\t' + 'Time' + '\t' + 'Train MAE' + '\t' + 'Valid MAE' + '\t' + 'Test MAE' + '\t' + 'PNorm'+ '\t' + 'GNorm' + '\n'
            elif this_dic['uncertainty'] and this_dic['uncertainty_method'] == 'swag':
                header = 'Epoch' + '\t' + 'Time' + '\t' + 'Train RMSE' + '\t' + 'Valid RMSE' + '\t' + 'Test RMSE' + '\t' + 'Train SWAG RMSE' + '\t' + 'Valid SWAG RMSE' + '\t' + 'Test SWAG RMSE' + '\t' + 'PNorm'+ '\t' + 'GNorm' + '\n' 
            elif this_dic['uncertainty'] and this_dic['uncertainty_method'] == 'dropout':
                header = 'Epoch' + '\t' + 'Time' + '\t' + 'Train Loss' + '\t' + 'Train RMSE' + '\t' + 'Valid Loss' + '\t' + 'Valid RMSE' + '\t' + 'Test Loss' + '\t' + 'Test RMSE' + '\t' + 'PNorm'+ '\t' + 'GNorm' + '\n'
            elif this_dic['model'] == 'VAE':
                header = 'Epoch' + '\t' + 'Time' + '\t' + 'Train Loss' + '\t' + 'Valid Loss' + '\t' + 'Test Loss' + '\t' + 'PNorm'+ '\t' + 'GNorm' + '\n'
            else:
                header = 'Epoch' + '\t' + 'Time' + '\t' + 'Train RMSE' + '\t' + 'Valid RMSE' + '\t' + 'Test RMSE' + '\t' + 'PNorm'+ '\t' + 'GNorm' + '\n'
            f.write(header)

def saveToResultsFile(this_dic, contents, name='data.txt'):

    assert os.path.exists(os.path.join(this_dic['running_path'], name))
    torch.save(contents[0].state_dict(), os.path.join(this_dic['running_path'], 'trained_model', 'model_'+str(contents[1])+'.pt'))

    if this_dic['model'].endswith('dropout'):
        assert len(contents) == 12
        with open(os.path.join(this_dic['running_path'], 'data.txt'), 'a') as f1:
            f1.write(str(contents[1]) + '\t' + str(round(contents[2]-contents[3], 2)) + '\t' + str(round(contents[4], 7)) + '\t' + str(round(contents[5], 7)) + '\t' + str(round(contents[6], 7)) + '\t' +  str(round(contents[7], 7)) + '\t' +  str(round(contents[8], 7)) + '\t' +  str(round(contents[9], 7)) + '\t' + str(contents[10]) + '\t' + str(contents[11]) + '\n')     
    
    if this_dic['model'].endswith('swag'): 
        assert len(contents) == 12
        with open(os.path.join(this_dic['running_path'], 'data.txt'), 'a') as f1:
            f1.write(str(contents[1]) + '\t' + str(round(contents[2]-contents[3], 2)) + '\t' + str(round(contents[4], 7)) + '\t' + str(round(contents[5], 7)) + '\t' + str(round(contents[6], 7)) + '\t' +  str(round(contents[7], 7)) + '\t' +  str(round(contents[8], 7)) + '\t' +  str(round(contents[9], 7)) + '\t' + str(contents[10]) + '\t' + str(contents[11]) + '\n')     
    
    if this_dic['taskType'] == 'multi':
        with open(os.path.join(this_dic['running_path'], 'data.txt'), 'a') as f1:
            f1.write(str(contents[1]) + '\t' + str(round(contents[2]-contents[3], 2)) + '\t' + str(round(contents[4], 7)) + '\t' + str(round(contents[5], 7)) + '\t' + str(round(contents[6], 7)) + '\t' +  str(round(contents[7], 7)) + '\t' +  str(round(contents[8], 7)) + '\t' +  str(round(contents[9], 7)) + '\t' + str(contents[10]) + '\t' + str(contents[11]) + '\n')     
        with open(os.path.join(this_dic['running_path'], 'data1.txt'), 'a') as f2:
            f2.write(str(contents[1]) + '\t' + str(round(contents[2]-contents[3], 2)) + '\t' + str(round(contents[4], 7)) + '\t' + str(round(contents[5], 7)) + '\t' + str(round(contents[6], 7)) + '\t' +  str(round(contents[7], 7)) + '\t' +  str(round(contents[8], 7)) + '\t' +  str(round(contents[9], 7)) + '\t' + str(contents[10]) + '\t' + str(contents[11]) + '\n')     
    
    if this_dic['model'] == 'VAE':
        assert len(contents) == 9
        with open(os.path.join(this_dic['running_path'], 'data.txt'), 'a') as f1:
            f1.write(str(contents[1]) + '\t' + str(round(contents[2]-contents[3], 2)) + '\t' + str(round(contents[4], 7)) + '\t' + str(round(contents[5], 7)) + '\t' + str(round(contents[6], 7)) + '\t' + str(contents[7]) + '\t' + str(contents[8]) + '\n')     
    
    if this_dic['model'] in ['1-2-GNN', 'loopybp', 'wlkernel']: # 1-2-GNN, loopybp, wlkernel
        assert len(contents) == 9
        with open(os.path.join(this_dic['running_path'], 'data.txt'), 'a') as f1:
            f1.write(str(contents[1]) + '\t' + str(round(contents[2]-contents[3], 2)) + '\t' + str(round(contents[4], 7)) + '\t' + str(round(contents[5], 7)) + '\t' + str(round(contents[6], 7)) + '\t' + str(contents[7]) + '\t' + str(contents[8]) + '\n')

def saveConfig(this_dic, name='config.json'):
    if this_dic['model'] in ['1-2-GNN', 'wlkernel', 'loopybp']:
        configs = BASIC + GNN
    if this_dic['uncertainty']:
        configs = BASIC + UQ
    if this_dic['model'] in ['VAE']:
        configs = BASIC + VAE_opts 
    if this_dic['water_interaction']:
        configs = BASIC + GNNVariants
    if this_dic['transfer_from']:
        configs = BASIC + TRANSFER
    
    with open(os.path.join(this_dic['running_path'], name), 'w') as f:
            #print(this_dic)
            json.dump({key: value for key, value in this_dic.items() if key in configs}, f)

def transferWeights(preTrainedModelDict, CurrentModelDict, CurrentModel, this_dic):
    for name, val in preTrainedModelDict.items():
        if name.startswith('conv.0'):
            trainedParam = {name: val}  
        if name.startswith('W'):
            trainedParam = {name: val}
        if name.startswith('fc'):
            trainedParam = {name: val}
            
    CurrentModelDict.update(trainedParam) # only atom embeddings -realted 
    CurrentModel.load_state_dict(CurrentModelDict)
                
    if this_dic['params'] == 'part':
        if this_dic['model'] in ['1-2-GNN']:
            for param in CurrentModel.conv.parameters(): # atom embedding layers-related weights are freezed.
                param.requires_grad = False
                        
        if this_dic['model'] in ['loopybp']:
            for param in CurrentModel.W_i.parameters(): # atom embedding layers-related weights are freezed.
                param.requires_grad = False
            for param in CurrentModel.W_h.parameters(): # atom embedding layers-related weights are freezed.
                param.requires_grad = False
            for param in CurrentModel.W_o.parameters(): # atom embedding layers-related weights are freezed.
                param.requires_grad = False
                        
        if this_dic['model'] in ['wlkernel']:
            for param in CurrentModel.fc_atom00.parameters(): # atom embedding layers-related weights are freezed.
                param.requires_grad = False
            for param in CurrentModel.fc_atom01.parameters(): # atom embedding layers-related weights are freezed.
                param.requires_grad = False
            for param in CurrentModel.fc_atom02.parameters(): # atom embedding layers-related weights are freezed.
                param.requires_grad = False
            for param in CurrentModel.fc_atom03.parameters(): # atom embedding layers-related weights are freezed.
                param.requires_grad = False
            for param in CurrentModel.fc_atom0.parameters(): # atom embedding layers-related weights are freezed.
                param.requires_grad = False
            for param in CurrentModel.fc_atom1.parameters(): # atom embedding layers-related weights are freezed.
                param.requires_grad = False
            for param in CurrentModel.fc_bond0.parameters(): # atom embedding layers-related weights are freezed.
                param.requires_grad = False
            for param in CurrentModel.fc_bond1.parameters(): # atom embedding layers-related weights are freezed.
                param.requires_grad = False
            for param in CurrentModel.fc_lei.parameters(): # atom embedding layers-related weights are freezed.
                param.requires_grad = False
            for param in CurrentModel.fc_new_lei.parameters(): # atom embedding layers-related weights are freezed.
                param.requires_grad = False
    return CurrentModel

def loadConfig(path, name='config.json'):
    with open(os.path.join(path,name), 'r') as f:
        config = json.load(f)
    return config
