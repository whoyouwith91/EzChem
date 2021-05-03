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
    'emb_dim': hp.choice('emb_dim', [50, 100, 150, 200, 250, 300, 500]),
    'num_layer': hp.choice('num_layer', [2, 3, 4, 5, 6, 7]),
    'lr': hp.choice('lr', [0.01, 0.001, 0.005, 0.0001, 0.0005]),
    'batch_size': hp.choice('batch_size', [32, 64, 128, 256]) # [32, 64, 128, 256]
}
INT_KEYS = ['emb_dim', 'num_layer', 'lr', 'batch_size']

def parse_input_arguments():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--allDataPath', type=str, default='/scratch/dz1061/gcn/chemGraph/data')
    parser.add_argument('--running_path', type=str,
                        help='path to save model', default='/scratch/dz1061/gcn/chemGraph/results')    
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--NumOutLayers', type=int, default=3) # number of readout layers
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--normalize', action='store_true')  # on target data
    parser.add_argument('--drop_ratio', type=float, default=0.0) 
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--model', type=str, default="1-GNN")
    parser.add_argument('--EFGS', action='store_true')
    parser.add_argument('--residual_connect', action='store_true')
    parser.add_argument('--resLayer', type=int, default=-1)
    parser.add_argument('--interaction', type=str, default="")
    parser.add_argument('--interaction_simpler', action='store_true')
    parser.add_argument('--pooling', type=str, default='sum')
    parser.add_argument('--aggregate', type=str, default='add')
    parser.add_argument('--degree', type=int, default=0)
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--train_type', type=str, default='from_scratch', choices=['from_scratch', 'transfer', 'hpsearch', 'finetuning'])
    parser.add_argument('--preTrainedPath', type=str)
    parser.add_argument('--OnlyPrediction', action='store_true')
    parser.add_argument('--loss', type=str, choices=['l1', 'l2', 'smooth_l1', 'dropout', 'vae', 'unsuper', 'maskedL2'])
    parser.add_argument('--metrics', type=str, choices=['l1', 'l2'])
    parser.add_argument('--weights', type=str, choices=['he_norm', 'xavier_norm', 'he_uni', 'xavier_uni'], default='he_uni')
    parser.add_argument('--lr_style', type=str, choices=['constant', 'decay']) # now is exponential decay on valid loss
    parser.add_argument('--optimizer',  type=str, choices=['adam', 'sgd', 'swa'])
    parser.add_argument('--style', type=str, choices=['base', 'CV', 'preTraining'])  # if running CV
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--experiment', type=str)  # when doing experimenting, name it. 
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--propertyLevel', type=str, default='molecule')
    
    parser.add_argument('--uncertainty',  type=str)
    parser.add_argument('--uncertaintyMode',  type=str)
    parser.add_argument('--swag_start', type=int)
    parser.add_argument('--preTrainedData', type=str)
    return parser.parse_args()

def grid_search(argss):
    # Run grid search
    results = []
    t = PrettyTable(['Hidden Size', 'Encoder Layers',  'Learning Rate', 'Batch Size', 'Mean', 'Std'])
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

        tst_error = cv_train(args1, t)

        results.append({
            'tstRMSE': tst_error,
            'hyperparams': hyperparams
        })
    
        return tst_error

    fmin(objective, SPACE, algo=tpe.suggest, max_evals=100)

    # Report best result
    best_result = min(results, key=lambda result: result['tstRMSE'])
    this_dic['running_path'] = os.path.join(args.running_path, args.dataset, args.model, args.gnn_type, args.experiment, 'HPSearch') 
    if not os.path.exists(this_dic['running_path']):
        os.makedirs(this_dic['running_path'])
    with open(os.path.join(this_dic['running_path'], 'hpsearch.json'), 'w') as f:
        json.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)

def main():
    args = parse_input_arguments()
    this_dic = vars(args)

    # define task type: multi or single
    if args.dataset == 'calcSolLogP/ALL':
       this_dic['taskType'] = 'multi'
       args.num_tasks = 3
    else:
       this_dic['taskType'] = 'single'

    # load data size info for train/validation/test because we save all of them in one single file. 
    with open('/scratch/dz1061/gcn/chemGraph/configs/splitsize.json', 'r') as f:
        dataSizes = json.load(f)
    if args.dataset in ['sol_calc/ALL', 'solOct_calc/ALL', 'calcLogP/ALL', 'calcLogPWithWater/ALL', 'calcSolLogP/ALL', 'xlogp3', 'solWithWater_calc/ALL'] and args.style == 'preTraining':
        args.dataset = args.dataset+'/COMPLETE'
    this_dic['train_size'], this_dic['val_size'] = int(dataSizes[args.dataset]['train_size']), int(dataSizes[args.dataset]['val_size'])

    # define path to load data for different training tasks
    if args.style == 'base': 
        this_dic['data_path'] = os.path.join(args.allDataPath, args.dataset, 'graphs', args.style, args.model)
    #if args.style == 'CV':
    #    this_dic['data_path'] = os.path.join(args.allDataPath, args.dataset, 'graphs', args.style, args.model, 'cv_'+str(this_dic['cv_folder'])) 
    if args.style == 'preTraining':
        this_dic['data_path'] = os.path.join(args.allDataPath, args.dataset, 'graphs/base', 'COMPLETE', args.model)

    grid_search(this_dic)

if __name__ == '__main__':
    main()
