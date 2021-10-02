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
from torchcontrib.optim import SWA
import sklearn.metrics as metrics

from typing import List, Union
from kwargs import physnet_kwargs

################### Configuration setting names ############################
data_config = ['dataset', 'model', 'style', 'data_path', 'EFGS', 'efgs_lenth', 'num_i_2', 'batch_size']
model_config = ['dataset', 'model', 'gnn_type',  'batch_size', 'emb_dim', 'act_fn' , 'weights', 'num_atom_features', 'num_tasks', 'propertyLevel', \
         'num_bond_features', 'pooling', 'NumParas', 'num_layer', 'JK', 'fully_connected_layer_sizes', 'aggregate', 'mol_features', 'tsne', \
             'residual_connect', 'resLayer', 'interaction_simpler', 'weight_regularizer', 'dropout_regularizer', 'gradCam', 'uncertainty', \
                 'uncertaintyMode', 'drop_ratio', 'energy_shift_value', 'energy_scale_value', 'deg_value']
train_config = ['running_path', 'seed', 'num_tasks', 'propertyLevel', 'test_level', 'optimizer', 'loss', 'metrics', 'lr', 'lr_style', \
         'epochs', 'early_stopping', 'train_type', 'taskType', 'train_size', 'val_size', 'test_size', 'sample', 'sample_size', 'data_seed', \
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
    if args.optimizer == 'adamW':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.optimizer == 'SWA':
        base_opt = torch.optim.SGD(model.parameters(), lr=args.lr)
        optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.0005)
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

def MaskedL1Loss(y, x, mask):
    """
    Masked mean squared error
    """
    x_masked = x[mask>0].reshape(-1, 1)
    y_masked = y[mask>0].reshape(-1, 1)
    return F.l1_loss(x_masked, y_masked)

def get_loss_fn(name):
    if name == 'l1':
        return F.l1_loss
    if name == 'l2':
        return F.mse_loss
    if name == 'maskedL2':
        return MaskedMSELoss
    if name == 'maskedL1':
        return MaskedL1Loss
    if name == 'smooth_l1':
        return F.smooth_l1_loss
    if name == 'dropout':
        return heteroscedastic_loss
    if name == 'vae':
       return vae_loss
    if name == 'unsuper':
       return unsuper_loss
    if name =='class':
        return F.cross_entropy

def get_metrics_fn(name):
    if name == 'l1':
        return F.l1_loss
    if name == 'l2':
        return F.mse_loss
    if name == 'smooth_l1':
        return F.smooth_l1_loss
    if name == 'class':
        return metrics.accuracy_score

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
    if this_dic['loss'] in ['l2', 'maskedL2']:
        train_header = 'RMSE'
    if this_dic['loss'] in ['l1', 'maskedL1']:
        train_header = 'MAE'
    if this_dic['loss'] == 'class':
        train_header1 = 'Loss'
        train_header2 = 'Accuracy'
    if this_dic['metrics'] == 'l2':
        test_header = 'RMSE'
    if this_dic['metrics'] == 'l1':
        test_header = 'MAE'
    if this_dic['metrics'] == 'class':
        test_header = 'Accuracy'
    if this_dic['loss'] == 'class':
        header = ['Epoch', 'Time', 'LR', 'Train {}'.format(train_header1), 'Train {}'.format(train_header2), 'Valid {}'.format(test_header), 'Test {}'.format(test_header), 'PNorm', 'GNorm']
    else:
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

def semi_orthogonal_matrix(N, M, seed=None):
    if N > M:  # number of rows is larger than number of columns
        square_matrix = square_orthogonal_matrix(dim=N, seed=seed)
    else:  # number of columns is larger than number of rows
        square_matrix = square_orthogonal_matrix(dim=M, seed=seed)
    return square_matrix[:N, :M]

def getDegreeforPNA(loader):
    from torch_geometric.utils import degree
    
    deg = torch.zeros(6, dtype=torch.long)
    for data in loader:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg, str(list(deg.numpy()))

def getScaleandShift(config):
    if config['dataset'] in ['pka/dataWarrior/acidic']:
            energy_shift = torch.tensor([0.3268184327923009]) # statistics over all compounds in train set after ACSF, unit is kcal.mol
            energy_scale = torch.tensor([0.23321891601867273]) # statistics over all compounds in train set after ACSF, unit is kcal.mol
    if config['dataset'] in ['pka/dataWarrior/basic']:
            #energy_shift = torch.tensor([0.])
            #energy_scale = torch.tensor([1.])
            energy_shift = torch.tensor([0.26469873653415377]) # statistics over all compounds in train set after ACSF, unit is kcal.mol
            energy_scale = torch.tensor([0.19022209195036874]) # statistics over all compounds in train set after ACSF, unit is kcal.mol
    if config['dataset'] in ['deepchem/freesol']:
            #energy_shift = torch.tensor([-0.448323499821181]) # statistics over all compounds in train set before ACSF unit is kcal/mol
            #energy_scale = torch.tensor([0.5231881290032624]) # statistics over all compounds in train set before ACSF, unit is kcal/mol
            energy_shift = torch.tensor([-0.22380834618774426]) # statistics over all compounds in train set after ACSF, unit is kcal.mol
            energy_scale = torch.tensor([0.2554506303590151]) # statistics over all compounds in train set after ACSF, unit is kcal.mol
    if config['dataset'] in ['deepchem/delaney']:
            #energy_shift = torch.tensor([-0.12582392787107535])
            #energy_scale = torch.tensor([0.09535901863588805])
            energy_shift = torch.tensor([-0.1239565706960962])
            energy_scale = torch.tensor([0.09412915806545324])
    if config['dataset'] in ['deepchem/logp']:
            energy_shift = torch.tensor([0.04861841981877047])
            energy_scale = torch.tensor([0.03197769536926355])
    if config['dataset'] in ['mp/bradley']:
            energy_shift = torch.tensor([2.6049698263964354])
            energy_scale = torch.tensor([5.931774882494167])
    if config['dataset'] in ['qm9/nmr/allAtoms']: # loading physnet params
            energy_shift = torch.tensor([67.2858])
            energy_scale = torch.tensor([85.8406])
    if config['dataset'] in ['qm9/nmr/carbon']: # loading physnet params
            energy_shift = torch.tensor([115.9782138561384])
            energy_scale = torch.tensor([51.569003335315905])
    if config['dataset'] in ['qm9/nmr/hydrogen']: # loading physnet params
            energy_shift = torch.tensor([29.08285732440852])
            energy_scale = torch.tensor([1.9575037908857158])
    if config['dataset'] in ['qm9/u0', 'qm9/allAtoms']:
            energy_shift = torch.tensor([-4.1164152221029555])
            energy_scale = torch.tensor([0.9008408776783313])
    if config['dataset'] in ['nmr/carbon']:
            energy_shift = torch.tensor([98.23851013183594])
            energy_scale = torch.tensor([51.27542495727539])
    if config['dataset'] in ['nmr/hydrogen']:
            energy_shift = torch.tensor([4.6759105])
            energy_scale = torch.tensor([2.6481516])
    if config['dataset'] in ['sol_calc/ALL/smaller', 'sol_calc/ALL/smaller_18W', 'sol_calc/ALL/smaller_28W', 'sol_calc/ALL/smaller_38W', 'sol_calc/ALL/smaller_48W', 'sol_calc/ALL/smaller_58W']:
            energy_shift = torch.tensor([-0.37722206969568495]) # unit is kcal/mol
            energy_scale = torch.tensor([0.25533234760965845]) # unit is kcal/mol
    if config['dataset'] in ['logp_calc/ALL/smaller_58W']:
            energy_shift = torch.tensor([0.]) # unit is kcal/mol
            energy_scale = torch.tensor([1.]) # unit is kcal/mol
    if config['dataset'] in ['solALogP', 'solNMR']:
            if config['propertyLevel'] == 'multiMol':
                energy_shift = torch.tensor([-4.232369738478913]) # eV for smaller gas energy
                energy_scale = torch.tensor([0.3176761953853432]) # eV for smaller gas energy 
            elif config['propertyLevel'] in ['molecule', 'atomMol']:
                energy_shift = torch.tensor([-0.016289007907758023]) # eV for smaller solvation 
                energy_scale = torch.tensor([0.011774938538003136]) # eV for smaller solvation 
            elif config['propertyLevel'] == 'atom':
                energy_shift = torch.tensor([75.4205])
                energy_scale = torch.tensor([151.0311])
            else:
                energy_shift = torch.tensor([0.])
                energy_scale = torch.tensor([1.])
    if config['dataset'] in ['secSolu/set1', 'secSolu/set2']:
            #energy_shift = torch.tensor([-0.12582392787107535])
            #energy_scale = torch.tensor([0.09535901863588805])
            energy_shift = torch.tensor([-0.09729485081724314])
            energy_scale = torch.tensor([0.09025854835266788])
    
    config['energy_shift'], config['energy_shift_value'] = energy_shift, energy_shift.item()
    config['energy_scale'], config['energy_scale_value'] = energy_scale, energy_scale.item()
    
    return config

def getScaleandShift_from_scratch(config, loader):
    # loader: train_loader 
    train_values = []
    train_N = []
    for data in loader:
        if 'mol_y' in data:
            train_values.extend(list(data.mol_y.numpy()))
            train_N.extend(list(data.N.numpy()))
        elif 'mol_sol_wat' in data:
            train_values.extend(list(data.mol_sol_wat.numpy()))
            train_N.extend(list(data.N.numpy()))
        elif 'atom_y' or 'y' in data and 'mask' in data:
            # 'atom_y' and 'mask' are in 1-GNN graphs
            # 'y' and 'mask' are in physnet graphs
            train_values.extend(list(data.atom_y[data.mask>0].numpy()))
        else:
            pass
    #print(len(train_values), len(train_N))
    if 'atom_y' or 'y' in data and 'mask' in data:
        shift, scale = np.mean(train_values), np.std(train_values)
    else:
        assert len(train_values) == len(train_N)
        shift, scale = atom_mean_std(train_values, train_N, range(len(train_values)))
    config['energy_shift'], config['energy_shift_value'] = torch.tensor([shift]), shift
    config['energy_scale'], config['energy_scale_value'] = torch.tensor([scale]), scale
    return config 

def getElements(dataframe):
    assert 'SMILES' in dataframe.columns
    element = set()
    for smi in dataframe['SMILES']:
        mol = Chem.MolFromSmiles(smi)
        for c in [atom.GetSymbol() for atom in mol.GetAtoms()]:
            element.add(c)
        #break
    return element

def atom_mean_std(E, N, index):
    """
    calculate the mean and stand variance of Energy in the training set
    :return:
    """
    mean = 0.0
    std = 0.0
    num = len(index)
    for _i in range(num):
        i = index[_i]
        m_prev = mean
        x = E[i] / N[i]
        mean += (x - mean) / (i + 1)
        std += (x - mean) * (x - m_prev)
    std = math.sqrt(std / num)
    return mean, std