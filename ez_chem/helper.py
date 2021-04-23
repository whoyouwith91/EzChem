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

def createResultsFile(this_dic, name='data.txt'):
    with open(os.path.join(this_dic['running_path'], name), 'w') as f:
            if this_dic['dataset'] in ['qm9', 'nmr/carbon', 'nmr/hydrogen', 'qm9/nmr/carbon', 'qm9/nmr/hydrogen']:
                header = 'Epoch' + '\t' + 'Time' + '\t' + 'LR' + '\t' + 'Train MAE' + '\t' + 'Valid MAE' + '\t' + 'Test MAE' + '\t' + 'PNorm'+ '\t' + 'GNorm' + '\n'
            elif this_dic['uncertainty'] and this_dic['uncertainty_method'] == 'swag':
                header = 'Epoch' + '\t' + 'Time' + '\t' + 'LR' + '\t' + 'Train RMSE' + '\t' + 'Valid RMSE' + '\t' + 'Test RMSE' + '\t' + 'Train SWAG RMSE' + '\t' + 'Valid SWAG RMSE' + '\t' + 'Test SWAG RMSE' + '\t' + 'PNorm'+ '\t' + 'GNorm' + '\n' 
            elif this_dic['uncertainty'] and this_dic['uncertainty_method'] == 'dropout':
                header = 'Epoch' + '\t' + 'Time' + '\t' + 'LR' + '\t' + 'Train Loss' + '\t' + 'Train RMSE' + '\t' + 'Valid Loss' + '\t' + 'Valid RMSE' + '\t' + 'Test Loss' + '\t' + 'Test RMSE' + '\t' + 'PNorm'+ '\t' + 'GNorm' + '\n'
            elif this_dic['model'] in ['VAE', 'TransformerUnsuper']:
                header = 'Epoch' + '\t' + 'Time' + '\t' + 'LR' + '\t' + 'Train Loss' + '\t' + 'Valid Loss' + '\t' + 'Test Loss' + '\t' + 'PNorm'+ '\t' + 'GNorm' + '\n'
            else:
                header = 'Epoch' + '\t' + 'Time' + '\t' + 'LR' + '\t' + 'Train RMSE' + '\t' + 'Valid RMSE' + '\t' + 'Test RMSE' + '\t' + 'PNorm'+ '\t' + 'GNorm' + '\n'
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

    if this_dic['model'] in ['1-GNN', '1-2-GNN', '1-efgs-GNN', '1-2-efgs-GNN', '1-interaction-GNN', 'Roberta']: # 1-2-GNN, loopybp, wlkernel
        assert len(contents) == 8
        with open(os.path.join(this_dic['running_path'], 'data.txt'), 'a') as f1:
            f1.write(str(contents[1]) + '\t' + str(round(contents[2], 2)) + '\t' + str(round(contents[3], 7)) + '\t' + \
                str(round(contents[4], 7)) + '\t' + str(round(contents[5], 7)) + '\t' + str(round(contents[6], 7)) + '\t' \
                    + str(contents[7]) + '\t' + str(contents[8]) + '\n')

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
