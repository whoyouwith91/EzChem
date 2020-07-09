"""Optimizes hyperparameters using Bayesian optimization."""

import os, sys, math, json, argparse, logging, time, random
from datetime import datetime
import torch
from copy import deepcopy
import json
from typing import Dict, Union
import os

from hyperopt import fmin, hp, tpe
import numpy as np

from chemprop.nn_utils import param_count
from helper import *
from model import *
from trainer import *
from data import *

SPACE = {
    'dimension': hp.quniform('dimension', low=64, high=256, q=64),
    'depths': hp.quniform('depths', low=2, high=6, q=1),
    'NumOutLayers': hp.quniform('NumOutLayers', low=1, high=3, q=1),
    'dropout': hp.quniform('dropout', low=0.0, high=0.4, q=0.05)
}
INT_KEYS = ['dimension', 'depths', 'NumOutLayers', 'dropout']

def parse_input_arguments():
    parser = argparse.ArgumentParser(description='Water solubility prediction')
    parser.add_argument('--dataset', type=str, choices=['ws', 'logp', 'mp', 'xlogp3', 'qm9', 'sol_exp', 'ccdc_sol', 'ccdc_logp', 'ccdc_sollogp', 'mp_drugs', 'mp_less'],
                        help='dataset for training (default: water solubility)')
    parser.add_argument('--allDataPath', type=str, default='/beegfs/dz1061/gcn/chemGraph/data')
    parser.add_argument('--efgs', action='store_true')
    parser.add_argument('--water_cat', action='store_true') # concatenate water embeddings
    parser.add_argument('--water_att', action='store_true') # embed water embeddings to attention layers
    parser.add_argument('--water_interaction', action='store_true')
    parser.add_argument('--InterByConcat', action='store_true')
    parser.add_argument('--InterBySub', action='store_true')
    parser.add_argument('--mol', action='store_true')
    parser.add_argument('--running_path', type=str,
                        help='path to save model', default='/beegfs/dz1061/gcn/chemGraph/results')                    
    parser.add_argument('--use_cuda', default=True,
                        help='enables CUDA training')
    parser.add_argument('--model', type=str, choices=['1-2-GNN', 'ConvGRU', '1-GNN', 'SAGE', 'GCN', 'CONVATT', 'adaConv', 'ConvSet2Set', 'NNConvGAT', 'gradcam', 'dropout', 'loopybp', 'wlkernel'])
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--nheads', type=int, default=3)
    parser.add_argument('--epi_uncer',  action='store_true')
    parser.add_argument('--pooling', type=str, choices=['add', 'mean', 'max', 'set2set'], default='add')
    parser.add_argument('--act_fn', type=str, choices=['relu', 'leaky_relu', 'elu', 'tanh', 'selu'], default='relu')
    parser.add_argument('--default', action='store_false')
    parser.add_argument('--weights', type=str, choices=['he_norm', 'xavier_norm', 'he_uni', 'xavier_uni'], default='he_uni')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--loss', type=str, choices=['l1', 'l2', 'smooth_l1', 'dropout'])
    parser.add_argument('--sgd',  action='store_true')
    parser.add_argument('--adam',  action='store_true')
    parser.add_argument('--swa',  action='store_true')
    parser.add_argument('--swag',  action='store_true')
    parser.add_argument('--swagHetero', action='store_true')
    parser.add_argument('--swag_start', type=int, default=10)
    parser.add_argument('--metrics', type=str, choices=['l1', 'l2'])
    parser.add_argument('--lr_style', type=str, choices=['constant', 'decay'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--patience_epochs', type=int, default=30)
    parser.add_argument('--train_type', type=str, default='from_scratch', choices=['from_scratch', 'transfer'])
    parser.add_argument('--transfer_from', type=str)
    parser.add_argument('--pre_trained_path', type=str)
    parser.add_argument('--pre_trained_model', type=str)
    parser.add_argument('--style', type=str, default='CV')
    return parser.parse_args()

def grid_search(argss):

    # Run grid search
    results = []

    # Define hyperparameter optimization
    def objective(hyperparams: Dict[str, Union[int, float]]) -> float:
        # Convert hyperparams from float to int when necessary
        for key in INT_KEYS:
            hyperparams[key] = int(hyperparams[key])

        # Copy argss
        hyper_args = deepcopy(argss)
        hyper_args = AttrDict(hyper_args)

        # Update args with hyperparams
        for key, value in hyperparams.items():
            setattr(hyper_args, key, value) 
        
        args1 = vars(hyper_args)

        val_error = cv_train(args1)

        results.append({
            'valRMSE': val_error,
            'hyperparams': hyperparams
        })
    
        return val_error

    fmin(objective, SPACE, algo=tpe.suggest, max_evals=20)

    # Report best result
    
    best_result = min(results, key=lambda result: result['valRMSE'])
    
    if not os.path.exists(os.path.join(argss['running_path'], argss['dataset'], argss['model'], 'HPSearchCV_new')):
                    os.makedirs(os.path.join(argss['running_path'], argss['dataset'], argss['model'], 'HPSearchCV_new'))
    with open(os.path.join(argss['running_path'], argss['dataset'], argss['model'], 'HPSearchCV_new', 'hpsearch.json'), 'w') as f:
        json.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)


def main():
    args = parse_input_arguments()
    this_dic = vars(args)
    
    this_dic['cv_path'] = os.path.join(args.allDataPath, args.dataset, 'graphs', args.style, this_dic['model'])
    with open('/beegfs/dz1061/gcn/chemGraph/configs/splitsize.json', 'r') as f:
        dataSizes = json.load(f)
    this_dic['train_size'] = int(dataSizes[args.dataset]['train_size'])
    this_dic['val_size'] = int(dataSizes[args.dataset]['val_size'])
    
    if args.dataset == 'ccdc_sollogp':
       this_dic['taskType'] = 'multi'
    else:
       this_dic['taskType'] = 'single'

    device = torch.device("cuda" if args.use_cuda else "cpu")
    this_dic['device'] = device

    grid_search(this_dic)

if __name__ == '__main__':
    main()
