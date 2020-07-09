import os, sys, math, json, argparse, logging, time, random
from datetime import datetime
import torch
import numpy as np
import torchcontrib.optim

from helper import *  
from model import *
from trainer import *
from data import *
from swag_model import *
from swag import *


def parse_input_arguments():
    parser = argparse.ArgumentParser(description='Water solubility prediction')
    parser.add_argument('--dataset', type=str, choices=['ws', 'logp', 'mp', 'xlogp3', 'qm9', 'sol_exp', 'ccdc_sol', 'ccdc_logp', 'ccdc_sollogp', 'mp_drugs', 'mp_less', 'sol_calc'],
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
    parser.add_argument('--depths', type=int, default=3)
    parser.add_argument('--NumOutLayers', type=int, default=3)
    parser.add_argument('--nheads', type=int, default=3)
    parser.add_argument('--dimension', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)
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
    parser.add_argument('--style', type=str)
    parser.add_argument('--experiment', type=str)
    return parser.parse_args()

def main():
    args = parse_input_arguments()
    this_dic = vars(args)
    
    this_dic['data_path'] = os.path.join(args.allDataPath, args.dataset, 'graphs', args.style, args.model) 
    args.running_path = os.path.join(args.running_path, args.dataset, args.model, args.style)
    
    with open('/beegfs/dz1061/gcn/chemGraph/configs/splitsize.json', 'r') as f:
        dataSizes = json.load(f)
    this_dic['train_size'] = int(dataSizes[args.dataset]['train_size'])
    this_dic['val_size'] = int(dataSizes[args.dataset]['val_size'])

    #logging.basicConfig(filename=os.path.join(args.running_path, 'Mylogfile_'+str(datetime.now())+'.log'), level=logging.INFO)
    #this_dic = vars(args)
    if args.dataset == 'ccdc_sollogp':
       this_dic['taskType'] = 'multi'
    else:
       this_dic['taskType'] = 'single'

    device = torch.device("cuda" if args.use_cuda else "cpu")
    this_dic['device'] = device
    best_val_error = float("inf")
    
    if not os.path.exists(os.path.join(args.running_path, 'trained_model/')):
        os.makedirs(os.path.join(args.running_path, 'trained_model/'))
    
    this_dic['data_path'] = os.path.join(this_dic['data_path'], 'shard_'+str(2))
    loader = get_data_loader(this_dic)
    std, num_features, num_bond_features, num_i_2 = loader.std, loader.num_features, loader.num_bond_features, loader.num_i_2
    this_dic['num_features'], this_dic['num_bond_features'], this_dic['num_i_2'], this_dic['std'] = int(num_features), num_bond_features, num_i_2, std
    del loader
 
    if this_dic['model'] in ['loopybp', 'wlkernel']:
        this_dic['atom_fdim'] = num_features
        this_dic['bond_fdim'] = num_bond_features
        this_dic['atom_messages'] = False
        this_dic['outDim'] = this_dic['dimension']
    if this_dic['model'] == 'dropout':
        this_dic['weight_regularizer'] = 1e-8 / len(train_loader.dataset)
        this_dic['dropout_regularizer'] = 2. / len(train_loader.dataset)
    if this_dic['model'] == '1-2-GNN' and this_dic['dataset'] == 'xlogp3':
        this_dic['num_i_2'] = 56  # for xlogp3 shards
    if this_dic['model'] == '1-2-GNN' and this_dic['dataset'] == 'sol_calc':
        this_dic['num_i_2'] = 84  # for sol_calc shards
 
    model = get_model(this_dic)
    model_ = model(this_dic).to(device)
    
    if this_dic['adam']:
        optimizer = torch.optim.Adam(model_.parameters(), lr=0.001)
    
    with open(os.path.join(this_dic['running_path'],  'data.txt'), 'w') as f:
        header = 'Epoch' + '\t' + 'Shard ID' + '\t' + 'Time' + '\t' + 'Train RMSE' + '\t' + 'Valid RMSE' + '\t' + 'Test RMSE' + '\t' + 'PNorm'+ '\t' + 'GNorm' + '\n'
        f.write(header) 

    for epoch in range(this_dic['epochs']):
        for id_ in range(100):
            this_dic['shard_ids'] = id_
            this_dic['data_path'] = os.path.join(this_dic['data_path'].split('shard_')[0], 'shard_'+str(id_))
            if not os.path.exists(this_dic['data_path']):
               continue
            loader = get_data_loader(this_dic)
            train_loader, val_loader, test_loader, std, num_features, num_bond_features, num_i_2 = loader.train_loader, loader.val_loader, loader.test_loader, loader.std, loader.num_features, loader.num_bond_features, loader.num_i_2
            
            #logging.info('Done Creating data file for train/valid/test...')
            time_tic = time.time()
            #lr = scheduler.optimizer.param_groups[0]['lr']
            loss = train(model_, optimizer, train_loader, this_dic)
            time_toc = time.time()
            
            train_error = test(model_, train_loader, this_dic)
            val_error = test(model_, val_loader, this_dic)
            test_error = test(model_, test_loader, this_dic)
            #print(train_error, val_error, test_error)
            del train_loader
            del val_loader
            del test_loader
            
            with open(os.path.join(this_dic['running_path'], 'data.txt'), 'a') as f1:
                f1.write(str(epoch) + '\t' + str(id_) + '\t' + str(round(time_toc-time_tic, 2)) + '\t' + str(round(train_error, 7)) + '\t' + str(round(val_error, 7)) + '\t' + str(round(test_error, 7)) + '\t' + str(param_norm(model_)) + '\t' + str(grad_norm(model_)) + '\n') 
            
            torch.save(model_.state_dict(), os.path.join(this_dic['running_path'], 'trained_model',  'model_shard_'+str(id_)+'.pt'))
            
            with open(os.path.join(this_dic['running_path'], 'config.json'), 'w') as f:
                #print(this_dic)
                json.dump({key: value for key, value in this_dic.items() if key not in ['device', 'std']}, f)


if __name__ == '__main__':
    main()

