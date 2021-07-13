import os, sys, random, math, json, glob, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, ELU, Tanh, SELU
from torch import Tensor
from torch_sparse import SparseTensor
from torch_scatter import scatter_mean, scatter_add, scatter_max
import torch_geometric.transforms as T
from typing import Optional
from prettytable import PrettyTable
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

from typing import List, Union
from kwargs import physnet_kwargs

################### Configuration setting names ############################
data_config = ['dataset', 'model', 'style', 'data_path', 'EFGS', 'efgs_lenth', 'num_i_2', 'train_size', 'val_size', 'batch_size']
model_config = ['dataset', 'model', 'gnn_type',  'batch_size', 'emb_dim', 'act_fn' , 'weights', 'num_atom_features', 'num_tasks', 'propertyLevel', \
         'num_bond_features', 'pooling', 'NumParas', 'num_layer', 'JK', 'fully_connected_layer_sizes', 'aggregate', 'mol_features', 'tsne', \
             'residual_connect', 'resLayer', 'interaction_simpler', 'weight_regularizer', 'dropout_regularizer', 'gradCam', 'uncertainty', \
                 'uncertaintyMode', 'drop_ratio', 'energy_shift_value', 'energy_scale_value']
train_config = ['running_path', 'seed', 'num_tasks', 'propertyLevel', 'test_level', 'optimizer', 'loss', 'metrics', 'lr', 'lr_style', \
         'epochs', 'early_stopping', 'train_type', 'taskType', 'train_size', 'val_size', 'test_size', \
         'preTrainedPath', 'uncertainty', 'uncertaintyMode', 'swag_start', 'action', 'mask', 'explicit_split', 'bn']
#VAE_opts = ['vocab_path', 'vocab_name', 'vocab_size', 'numEncoLayers', 'numDecoLayers', 'numEncoders', 'numDecoders', 'varDimen', 'anneal', 'kl_weight', 'anneal_method', 'anneal_epoch']
###############################################################################

def set_seed(seed):
# define seeds for training 
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

def get_optimizer(args, model):
    # define optimizers
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    return optimizer


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

def createResultsFile(this_dic):
    ## create pretty table
    if this_dic['loss'] == 'l2':
        train_header = 'RMSE'
    else:
        train_header = 'MAE'
    if this_dic['metrics'] == 'l2':
        test_header = 'RMSE'
    else:
        test_header = 'MAE'
    header = ['Epoch', 'Time', 'LR', 'Train {}'.format(train_header), 'Valid {}'.format(test_header), 'Test {}'.format(test_header), 'PNorm', 'GNorm']
    x = PrettyTable(header)
    return x 

def saveToResultsFile(table, this_dic, name='data.txt'):

    #assert os.path.exists(os.path.join(this_dic['running_path'], name))
    #torch.save(contents[0].state_dict(), os.path.join(this_dic['running_path'], 'trained_model', 'model_'+str(contents[1])+'.pt')
    if this_dic['model'].endswith('dropout'):
        assert len(contents) == 12
        with open(os.path.join(this_dic['running_path'], 'data.txt'), 'a') as f1:
            f1.write(str(contents[1]) + '\t' + str(round(contents[2]-contents[3], 2)) + '\t' + str(round(contents[4], 7)) + '\t' + str(round(contents[5], 7)) + '\t' + str(round(contents[6], 7)) + '\t' +  str(round(contents[7], 7)) + '\t' +  str(round(contents[8], 7)) + '\t' +  str(round(contents[9], 7)) + '\t' + str(contents[10]) + '\t' + str(contents[11]) + '\n')     
    
    if this_dic['model'].endswith('swag'): 
        assert len(contents) == 12
        with open(os.path.join(this_dic['running_path'], 'data.txt'), 'a') as f1:
            f1.write(str(contents[1]) + '\t' + str(round(contents[2]-contents[3], 2)) + '\t' + str(round(contents[4], 7)) + '\t' + str(round(contents[5], 7)) + '\t' + str(round(contents[6], 7)) + '\t' +  str(round(contents[7], 7)) + '\t' +  str(round(contents[8], 7)) + '\t' +  str(round(contents[9], 7)) + '\t' + str(contents[10]) + '\t' + str(contents[11]) + '\n')     

    if this_dic['model'] in ['1-GNN', '1-2-GNN', '1-efgs-GNN', '1-2-efgs-GNN', '1-interaction-GNN', 'Roberta', 'physnet']: # 1-2-GNN, loopybp, wlkernel
        with open(os.path.join(this_dic['running_path'], 'data.txt'), 'w') as f1:
            f1.write(str(table))
        f1.close()

def saveConfig(this_dic, name='config.json'):
    all_ = {'data_config': {key:this_dic[key] for key in data_config if key in this_dic.keys()},
            'model_config':{key:this_dic[key] for key in model_config if key in this_dic.keys()},
            'train_config': {key:this_dic[key] for key in train_config if key in this_dic.keys()}}
    if this_dic['model'] == 'physnet':
        phsynet_config = {'physnet_config': physnet_kwargs}
        all_ = {**all_, **phsynet_config}
    with open(os.path.join(this_dic['running_path'], name), 'w') as f:
        json.dump(all_, f, indent=2)

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
            torch.save(model.state_dict(), os.path.join(config['running_path'], 'best_model', 'model_best.pt'))
            if config['model'].endswith('swag'):
                torch.save(swag_model.state_dict(), os.path.join(config['running_path'], 'best_model', 'swag_model_'+str(epoch)+'.pt'))
    return bestValError

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def prettyTableToPandas(f):
    with open(f, 'r') as f:
        l = f.readlines()
    
    contents = [] 
    for line in l:
        if line.startswith('+'):
            continue
        contents.append([i.strip(' ') for i in line.split('|')[1:-1]])
        
    df = pd.DataFrame(contents[1:], columns=contents[0])
    #df.colunms = [i.strip() for i in df.columns]
    df = df.astype(float)
    return df

def build_lr_scheduler(optimizer, config):
    """
    Builds a PyTorch learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing learning rate arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """
    if config['scheduler'] == 'NoamLR':
        # Learning rate scheduler
        return NoamLR(
            optimizer=optimizer,
            warmup_epochs=[config['warmup_epochs']],
            total_epochs=[config['epochs']],
            steps_per_epoch=config['train_size'] // config['batch_size'],
            init_lr=[config['init_lr']],
            max_lr=[config['max_lr']],
            final_lr=[config['final_lr']]
        )
    elif config['scheduler'] == 'decay':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                    factor=config['decay_factor'], 
                    patience=config['patience_epochs'], 
                    min_lr=0.00001)
    
    elif config['scheduler'] == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, 
                    620000, gamma=0.1)
    else:
        return None

class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where :code:`warmup_steps = warmup_epochs * steps_per_epoch`).
    Then the learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr` over the
    course of the remaining :code:`total_steps - warmup_steps` (where :code:`total_steps =
    total_epochs * steps_per_epoch`). This is roughly based on the learning rate
    schedule from `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_, section 5.3.
    """
    def __init__(self,
                 optimizer,
                 warmup_epochs,
                 total_epochs,
                 steps_per_epoch,
                 init_lr,
                 max_lr,
                 final_lr):
        """
        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after :code:`warmup_epochs`).
        :param final_lr: The final learning rate (achieved after :code:`total_epochs`).
        """
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)
    
    def get_lr(self) -> List[float]:
        """
        Gets a list of the current learning rates.

        :return: A list of the current learning rates.
        """
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
                             If None, :code:`current_step = self.current_step + 1`.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]