"""Optimizes hyperparameters using Bayesian optimization."""

import os, sys, math, json, argparse, logging, time, random
from prettytable import PrettyTable
import torch
from copy import deepcopy
import json
from typing import Dict, Union
import os

from hyperopt import fmin, hp, tpe
import numpy as np

from helper import *
#from model import *
from trainer import *
#from data import *

SPACE = {
    'emb_dim': hp.choice('emb_dim', [50, 100, 150, 200, 250, 300]),
    'num_layer': hp.choice('num_layer', [2, 3, 4, 5, 6, 7]),
    'NumOutLayers': hp.choice('NumOutLayers', [1, 2, 3]),
    'lr': hp.choice('lr', [0.01, 0.001, 0.005, 0.0001, 0.0005]),
    'batch_size': hp.choice('batch_size', [32, 64, 128, 256]) # [32, 64, 128, 256]
}
INT_KEYS = ['emb_dim', 'num_layer', 'NumOutLayers', 'lr', 'batch_size']

def parse_input_arguments():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--allDataPath', type=str, default='/scratch/dz1061/gcn/chemGraph/data')
    parser.add_argument('--running_path', type=str,
                        help='path to save model', default='/scratch/dz1061/gcn/chemGraph/results')    
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--model', type=str, default="1-GNN")
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--pooling', type=str, default='add')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--train_type', type=str, default='from_scratch', choices=['from_scratch', 'transfer'])
    parser.add_argument('--loss', type=str, choices=['l1', 'l2', 'smooth_l1', 'dropout', 'vae', 'unsuper'])
    parser.add_argument('--metrics', type=str, choices=['l1', 'l2'])
    parser.add_argument('--weights', type=str, choices=['he_norm', 'xavier_norm', 'he_uni', 'xavier_uni'], default='he_uni')
    parser.add_argument('--lr_style', type=str, choices=['constant', 'decay']) # now is exponential decay on valid loss
    parser.add_argument('--optimizer',  type=str, choices=['adam', 'sgd', 'swa'])
    parser.add_argument('--style', type=str, choices=['base', 'CV', 'superPreTraining', 'hpsearch'])  # if running CV
    parser.add_argument('--experiment', type=str)  # when doing experimenting, name it. 
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--uncertainty',  type=str)
    parser.add_argument('--uncertainty_method',  type=str)
    parser.add_argument('--swag_start', type=int)
    return parser.parse_args()

def grid_search(argss):

    # Run grid search
    results = []
    t = PrettyTable(['Hidden Size', 'Encoder Layers', 'Read-out Layers', 'Learning Rate', 'Batch Size', 'Mean', 'Std'])
    # Define hyperparameter optimization
    def objective(hyperparams: Dict[str, Union[int, float]]) -> float:
        # Convert hyperparams from float to int when necessary
        for key in INT_KEYS:
            if not key == 'lr': 
                hyperparams[key] = int(hyperparams[key])

        # Copy argss
        hyper_args = deepcopy(argss)
        hyper_args = AttrDict(hyper_args)

        # Update args with hyperparams
        for key, value in hyperparams.items():
            setattr(hyper_args, key, value) 
        
        args1 = vars(hyper_args)

        val_error = cv_train(args1, t)

        results.append({
            'valRMSE': val_error,
            'hyperparams': hyperparams
        })
    
        return val_error

    fmin(objective, SPACE, algo=tpe.suggest, max_evals=100)

    # Report best result
    
    best_result = min(results, key=lambda result: result['valRMSE'])
    
    if not os.path.exists(os.path.join(argss['running_path'], argss['dataset'], argss['model'], argss['gnn_type'], argss['experiment'], 'HPSearchCV')):
                    os.makedirs(os.path.join(argss['running_path'], argss['dataset'], argss['model'], argss['gnn_type'], argss['experiment'], 'HPSearchCV'))
    with open(os.path.join(argss['running_path'], argss['dataset'], argss['model'], argss['gnn_type'], argss['experiment'], 'HPSearchCV', 'hpsearch.json'), 'w') as f:
        json.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)


def main():
    args = parse_input_arguments()
    this_dic = vars(args)
    
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    #if torch.cuda.is_available():
    #    torch.cuda.manual_seed_all(0)
    #this_dic['device'] = device

    with open('/scratch/dz1061/gcn/chemGraph/configs/splitsize.json', 'r') as f:
        dataSizes = json.load(f)
    if args.dataset in ['ws', 'logp', 'sol_exp', 'mp_less', 'deepchem/freesol', 'deepchem/logp', 'deepchem/delaney']:
       this_dic['train_size'] = int(dataSizes[args.dataset]['train_size'])
       this_dic['val_size'] = int(dataSizes[args.dataset]['val_size'])
    else:
       this_dic['train_size'] = int(dataSizes[args.dataset+'/hpsearch']['train_size'])
       this_dic['val_size'] = int(dataSizes[args.dataset+'/hpsearch']['val_size'])

    this_dic['data_path'] = os.path.join(args.allDataPath, args.dataset, 'graphs', args.style, 'base', args.model)
    
    if args.dataset == 'calcSolLogP':
       this_dic['taskType'] = 'multi'
       this_dic['num_tasks'] = 3
    elif args.dataset == 'commonProperties':
       this_dic['taskType'] = 'multi'
       this_dic['num_tasks'] = 4
    else:
       this_dic['taskType'] = 'single'

    grid_search(this_dic)

if __name__ == '__main__':
    main()
