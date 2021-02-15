import os, sys, random, math, json, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, ELU, Tanh, SELU
from torch import Tensor
from torch_sparse import SparseTensor
from torch_scatter import scatter_mean, scatter_add, scatter_max
import torch_geometric.transforms as T
from typing import Optional
from torch_geometric.typing import Adj

import numpy as np

################### Configuration setting names ############################
BASIC = ['dataset', 'running_path', 'model', 'gnn_type', 'normalize', 'batch_size', 'emb_dim', \
         'dropout', 'act_fn' , 'weights', 'seed', 'optimizer', 'loss', 'metrics', 'lr', 'lr_style', \
         'epochs', 'early_stopping', 'train_type', 'taskType', 'style', 'train_size', 'val_size', 'num_features', \
          'num_bond_features', 'data_path', 'efgs', 'water_interaction', 'InterByConcat', 'InterBySub', 'mol', 'pooling', 'NumParas', 'efgs_lenth', 'EFGS', 'solvent', 'interaction']
GNN = ['depths', 'num_layer', 'JK', 'NumOutLayers', 'pooling', 'num_features', 'num_i_2', 'pooling']
GNNVariants = ['efgs', 'water_interaction', 'InterByConcat', 'InterBySub', 'mol', 'num_i_2']
VAE_opts = ['vocab_path', 'vocab_name', 'vocab_size', 'numEncoLayers', 'numDecoLayers', 'numEncoders', 'numDecoders', 'varDimen', 'anneal', 'kl_weight', 'anneal_method', 'anneal_epoch']
UQ = ['depths', 'NumOutLayers', 'pooling', 'num_features', 'num_i_2', 'uncertainty', 'uncertainty_method', 'swag_start', 'num_i_2', 'atom_fdim', 'bond_fdim', 'atom_messages', 'outDim', 'weight_regularizer', 'dropout_regularizer']
TRANSFER = ['depths', 'NumOutLayers', 'pooling', 'num_features', 'num_i_2', 'transfer_from', 'pre_trained_path', 'pre_trained_model', 'params']
###############################################################################

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def heteroscedastic_loss(true, mean, log_var):
    '''
    dropout loss
    '''
    precision = torch.exp(-log_var)
    return torch.mean(torch.sum(0.5*precision * (true - mean.reshape(-1,))**2 + 0.5*log_var, 1), 0)

def unsuper_loss(recon_x, x, config):
    CELoss = F.cross_entropy(recon_x, x, ignore_index=1)
    return CELoss
    
def vae_loss(recon_x, x, mu, logvar, kl_weight, config, saveKL=False):
    #seq_size = x.size()[0]/config['batch_size']
    CELoss = F.cross_entropy(recon_x, x, ignore_index=1)  
    KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).mean().squeeze()
    #KLD /= seq_size
    if saveKL:
       return CELoss + kl_weight*KLD, CELoss, KLD
    else:
       return CELoss + kl_weight*KLD, None, None

def kl_anneal_function(anneal_function, step, k1=0.1, k2=0.2, max_value=1.0, x0=100):
    assert anneal_function in ['logistic', 'linear', 'step', 'cyclical'], 'unknown anneal_function'
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(- k1 * (step - x0))))
    elif anneal_function == 'step':
        cnt = step // x0
        step = step % x0
        if cnt > 0:
            max_value -= cnt * 0.1
            max_value = max(0.1, max_value)  
        ma = min(k2 * cnt + k2, max_value)
        mi = 0.01 + k1 * cnt
        return min(ma, mi + 2 * step * (max(ma - mi, 0)) / x0)
    elif anneal_function == 'linear':
        return min(max_value, 0.00001 + step / x0)
    elif anneal_function == 'cyclical':
        cnt = step // x0 // 5
        step = step % x0
        ma = min(k2 * cnt + k2, max_value)
        mi = k1
        return min(ma, ma * cnt + mi + 2 * step * (ma - mi) / x0)

def MaskedMSELoss(y, x, mask):
    """
    Masked mean squared error
    """
    x_masked = x[mask>0].reshape(-1, 1)
    y_masked = y[mask>0].reshape(-1, 1)
    return F.mse_loss(x_masked, y_masked)

def get_loss_fn(name):
    if name == 'l1':
        return F.l1_loss
    if name == 'l2':
        return F.mse_loss
    if name == 'maskedL2':
        return MaskedMSELoss
    if name == 'smooth_l1':
        return F.smooth_l1_loss
    if name == 'dropout':
        return heteroscedastic_loss
    if name == 'vae':
       return vae_loss
    if name == 'unsuper':
       return unsuper_loss

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
            elif this_dic['model'] in ['VAE', 'TransformerUnsuper']:
                header = 'Epoch' + '\t' + 'Time' + '\t' + 'Train Loss' + '\t' + 'Valid Loss' + '\t' + 'Test Loss' + '\t' + 'PNorm'+ '\t' + 'GNorm' + '\n'
            #elif this_dic['taskType'] == 'multi':
            #    header = 'Epoch' + '\t' + 'Time' + '\t' + 'Train RMSE' + '\t' + 'Valid RMSE' + '\t' + 'PNorm'+ '\t' + 'GNorm' + '\n'
            else:
                header = 'Epoch' + '\t' + 'Time' + '\t' + 'Train RMSE' + '\t' + 'Valid RMSE' + '\t' + 'Test RMSE' + '\t' + 'PNorm'+ '\t' + 'GNorm' + '\n'
            f.write(header)

def saveToResultsFile(this_dic, contents, name='data.txt'):

    assert os.path.exists(os.path.join(this_dic['running_path'], name))
    #torch.save(contents[0].state_dict(), os.path.join(this_dic['running_path'], 'trained_model', 'model_'+str(contents[1])+'.pt'))

    if this_dic['model'].endswith('dropout'):
        assert len(contents) == 12
        with open(os.path.join(this_dic['running_path'], 'data.txt'), 'a') as f1:
            f1.write(str(contents[1]) + '\t' + str(round(contents[2]-contents[3], 2)) + '\t' + str(round(contents[4], 7)) + '\t' + str(round(contents[5], 7)) + '\t' + str(round(contents[6], 7)) + '\t' +  str(round(contents[7], 7)) + '\t' +  str(round(contents[8], 7)) + '\t' +  str(round(contents[9], 7)) + '\t' + str(contents[10]) + '\t' + str(contents[11]) + '\n')     
    
    if this_dic['model'].endswith('swag'): 
        assert len(contents) == 12
        with open(os.path.join(this_dic['running_path'], 'data.txt'), 'a') as f1:
            f1.write(str(contents[1]) + '\t' + str(round(contents[2]-contents[3], 2)) + '\t' + str(round(contents[4], 7)) + '\t' + str(round(contents[5], 7)) + '\t' + str(round(contents[6], 7)) + '\t' +  str(round(contents[7], 7)) + '\t' +  str(round(contents[8], 7)) + '\t' +  str(round(contents[9], 7)) + '\t' + str(contents[10]) + '\t' + str(contents[11]) + '\n')     
    
    #if this_dic['taskType'] == 'multi' and this_dic['dataset'] != 'commonProperties':
    #    assert len(contents) == 8
    #    with open(os.path.join(this_dic['running_path'], 'data.txt'), 'a') as f1:
    #        f1.write(str(contents[1]) + '\t' + str(round(contents[2]-contents[3], 2)) + '\t' + str(round(contents[4], 7)) + '\t' + str(round(contents[5][0], 7)) + '\t' + str(contents[6]) + '\t' + str(contents[7]) + '\n')
    #    with open(os.path.join(this_dic['running_path'], 'data1.txt'), 'a') as f2:
    #        f2.write(str(contents[1]) + '\t' + str(round(contents[2]-contents[3], 2)) + '\t' + str(round(contents[4], 7)) + '\t' + str(round(contents[5][1], 7)) + '\t' + str(contents[6]) + '\t' + str(contents[7]) + '\n')
    
    if this_dic['model'] in ['VAE', 'TransformerUnsuper']:
        assert len(contents) == 9
        with open(os.path.join(this_dic['running_path'], 'data.txt'), 'a') as f1:
            f1.write(str(contents[1]) + '\t' + str(round(contents[2]-contents[3], 2)) + '\t' + str(round(contents[4], 7)) + '\t' + str(round(contents[5], 7)) + '\t' + str(round(contents[6], 7)) + '\t' + str(contents[7]) + '\t' + str(contents[8]) + '\n')     

    if this_dic['dataset'] == 'commonProperties': 
        assert len(contents) == 9
        with open(os.path.join(this_dic['running_path'], 'data.txt'), 'a') as f1:
            f1.write(str(contents[1]) + '\t' + str(round(contents[2]-contents[3], 2)) + '\t' + str(round(contents[4], 7)) + '\t' + str(round(contents[5], 7)) + '\t' + str(round(contents[6], 7)) + '\t' + str(contents[7]) + '\t' + str(contents[8]) + '\n')

    if this_dic['model'] in ['1-GNN', '1-2-GNN', '1-efgs-GNN', '1-2-efgs-GNN', '1-interaction-GNN', '1-interaction-GNN-naive', 'Roberta']: # 1-2-GNN, loopybp, wlkernel
        assert len(contents) == 9
        with open(os.path.join(this_dic['running_path'], 'data.txt'), 'a') as f1:
            f1.write(str(contents[1]) + '\t' + str(round(contents[2]-contents[3], 2)) + '\t' + str(round(contents[4], 7)) + '\t' + str(round(contents[5], 7)) + '\t' + str(round(contents[6], 7)) + '\t' + str(contents[7]) + '\t' + str(contents[8]) + '\n')

def saveConfig(this_dic, name='config.json'):
    if this_dic['model'] in ['1-GNN', '1-2-GNN', '1-efgs-GNN', '1-2-efgs-GNN', '1-interaction-GNN', '1-interaction-GNN-naive']:
        configs = BASIC + GNN
    elif this_dic['model'] in ['Roberta']:
        configs = BASIC 
    elif this_dic['uncertainty']:
        configs = BASIC + UQ
    elif this_dic['model'] in ['VAE', 'TransformerUnsuper']:
        configs = BASIC + VAE_opts 
    elif this_dic['water_interaction']:
        configs = BASIC + GNNVariants
    elif this_dic['transfer_from']:
        configs = BASIC + TRANSFER
    else:
        configs = BASIC
       
    with open(os.path.join(this_dic['running_path'], name), 'w') as f:
            #print(this_dic)
            json.dump({key: value for key, value in this_dic.items() if key in configs}, f)

def transferWeights(CurrentModelDict, CurrentModel, this_dic):
    list_of_files = glob.glob(os.path.join(this_dic['pre_trained_path'], 'best_model/', '*.pt'))
    latest_file = max(list_of_files, key=os.path.getctime)
    premodel_state_dict = torch.load(latest_file)
    print('Pre-trained model weights done loading from {}'.format(latest_file))
    this_dic['pre_trained_model'] = latest_file
    for name, val in premodel_state_dict.items():
        if name.startswith('conv.0'): # 1-2-GNN
            trainedParam = {name: val}  
        if name.startswith('W'): # loopybp
            trainedParam = {name: val}
        if name.startswith('fc'): # wlkernel
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
            #for param in CurrentModel.W_o.parameters(): # atom embedding layers-related weights are freezed.
            #    param.requires_grad = False
                        
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
    return CurrentModel, this_dic

def loadConfig(path, name='config.json'):
    with open(os.path.join(path,name), 'r') as f:
        config = json.load(f)
    return config

def saveModel(config, epoch, model, bestValError, valError, swag_model=None):
    if config['early_stopping']:
        if bestValError > np.sum(valError):
            patience = 0
            bestValError = np.sum(valError)
        else:
            patience += 1
            if patience > config['patience_epochs']:
                #logging.info('Early stopping! No consecutive improvements for %s epochs' % int(patience-1))
                #logging.info('Saving models...')
                torch.save(model.state_dict(), os.path.join(config['running_path'], 'best_model'))
                #logging.info('Model saved.')
                #break
    else:
        if bestValError > np.sum(valError):
            bestValError = np.sum(valError)
            #logging.info('Saving models...')
            torch.save(model.state_dict(), os.path.join(config['running_path'], 'best_model', 'model_'+str(epoch)+'.pt'))
            if config['model'].endswith('swag'):
                torch.save(swag_model.state_dict(), os.path.join(config['running_path'], 'best_model', 'swag_model_'+str(epoch)+'.pt'))
    return bestValError


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

class WLConv(torch.nn.Module):
    r"""The Weisfeiler Lehman operator from the `"A Reduction of a Graph to a
    Canonical Form and an Algebra Arising During this Reduction"
    <https://www.iti.zcu.cz/wl2018/pdf/wl_paper_translation.pdf>`_ paper, which
    iteratively refines node colorings:

    .. math::
        \mathbf{x}^{\prime}_i = \textrm{hash} \left( \mathbf{x}_i, \{
        \mathbf{x}_j \colon j \in \mathcal{N}(i) \} \right)
    """
    def __init__(self):
        super(WLConv, self).__init__()
        self.hashmap = {}

    def reset_parameters(self):
        self.hashmap = {}


    @torch.no_grad()
    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """"""
        if x.dim() > 1:
            assert (x.sum(dim=-1) == 1).sum() == x.size(0)
            x = x.argmax(dim=-1)  # one-hot -> integer.
        assert x.dtype == torch.long

        adj_t = edge_index
        if not isinstance(adj_t, SparseTensor):
            adj_t = SparseTensor(row=edge_index[1], col=edge_index[0],
                                 sparse_sizes=(x.size(0), x.size(0)))

        out = []
        _, col, _ = adj_t.coo()
        deg = adj_t.storage.rowcount().tolist()
        for node, neighbors in zip(x.tolist(), x[col].split(deg)):
            idx = hash(tuple([node] + neighbors.sort()[0].tolist()))
            if idx not in self.hashmap:
                self.hashmap[idx] = len(self.hashmap)
            out.append(self.hashmap[idx])

        return torch.tensor(out, device=x.device)


    def histogram(self, x: Tensor, batch: Optional[Tensor] = None,
                  norm: bool = False) -> Tensor:
        r"""Given a node coloring :obj:`x`, computes the color histograms of
        the respective graphs (separated by :obj:`batch`)."""

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_colors = len(self.hashmap)
        batch_size = int(batch.max()) + 1

        index = batch * num_colors + x
        out = scatter_add(torch.ones_like(index), index, dim=0,
                          dim_size=num_colors * batch_size)
        out = out.view(batch_size, num_colors)

        if norm:
            out = out.to(torch.float)
            out /= out.norm(dim=-1, keepdim=True)

        return out


    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
