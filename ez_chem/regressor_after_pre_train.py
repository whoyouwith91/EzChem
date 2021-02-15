import argparse, time, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from tqdm import tqdm
import pandas as pd
import numpy as np

from pre_train_model import *
#from pre_train_utils import *
from sklearn.metrics import roc_auc_score
##from loader import *
#from loader import MoleculeDataset
from helper import *
from data import *
from trainer import *

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
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
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--baseModel', type=str)
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)') 
    parser.add_argument('--pooling', type=str, choices=['add', 'mean', 'max', 'set2set'], default='add')
    parser.add_argument('--style', type=str, choices=['base', 'CV'])  # if running CV
    parser.add_argument('--experiment', type=str)  # when doing experimenting, name it. 
    parser.add_argument('--cv_folder', type=int) # if 
    args = parser.parse_args()
    this_dic = vars(args)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    
    config = loadConfig('/beegfs/dz1061/gcn/chemGraph/results/logp/1-2-GNN/SUMPOOL/CV_0')
    config['device'] = device
    with open('/beegfs/dz1061/gcn/chemGraph/configs/splitsize.json', 'r') as f:
        dataSizes = json.load(f)
    config['train_size'] = int(dataSizes[args.dataset]['train_size'])
    config['val_size'] = int(dataSizes[args.dataset]['val_size'])
    config['batch_size'] = 32
    config['gnn_type'] = args.gnn_type
    config['data_path'] = '/beegfs/dz1061/gcn/chemGraph/data/{}/graphs/{}/{}'.format(args.dataset, args.style, args.baseModel)
    if args.style == 'CV':
       config['data_path'] = os.path.join(config['data_path'], 'cv_'+str(args.cv_folder))
    config['JK'] = args.JK
    config['pooling'] = args.pooling
    config['baseModel'] = args.baseModel
    config['taskType'] = 'single'
    #set up dataset and transform function.
    #dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset, transform = ExtractSubstructureContextPair_rev(args.num_layer, l1, l2))
    loader = get_data_loader(config)
    train_loader, val_loader, test_loader, std, num_features, num_bond_features, num_i_2 = loader.train_loader, loader.val_loader, loader.test_loader, loader.std, loader.num_features, loader.num_bond_features, loader.num_i_2
    config['num_features'], config['num_bond_features'], config['num_i_2'], config['std'] = int(num_features), num_bond_features, num_i_2, std
 
    if not args.input_model_file == "":
        config['running_path'] = '/beegfs/dz1061/gcn/chemGraph/results/preTrain/{}/{}/{}/{}'.format(args.gnn_type, args.dataset, args.baseModel, args.experiment)
    else:
        config['running_path'] = '/beegfs/dz1061/gcn/chemGraph/results/{}/{}/{}/{}'.format(args.dataset, args.gnn_type, args.baseModel, args.experiment)
    config['uncertainty'] = False
    writer = SummaryWriter(os.path.join(config['running_path'], 'tensorboard'))
    if not os.path.exists(os.path.join(config['running_path'], 'trained_model/')): 
       os.makedirs(os.path.join(config['running_path'], 'trained_model/'))
    if not os.path.exists(os.path.join(config['running_path'], 'best_model/')):
       os.makedirs(os.path.join(config['running_path'], 'best_model/'))
    createResultsFile(config, name='data.txt')

    #set up models, one for pre-training and one for context embeddings
    best_val_error = float("inf")
    config['num_layer'] = args.num_layer
    if config['gnn_type'] == '1-2-GNN':
        model = GNN_2(config)
        if not args.input_model_file == "":
           model.from_pretrained(args.input_model_file + ".pth")
        model_ = model.to(device)
    if config['gnn_type'] == '1-GNN':
        model = GNN_1(config)
        if not args.input_model_file == "":
           model.from_pretrained(args.input_model_file + ".pth")
        model_ = model.to(device)
    
    #set up optimizer for the two GNNs
    if config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model_.parameters(), lr=config['lr'])
    config.update(this_dic)
    saveConfig(config, name='config.json')
    for epoch in range(1, args.epochs+1):
                saveContents = []
                time_tic = time.time()
                #lr = scheduler.optimizer.param_groups[0]['lr']
                loss = train(model_, optimizer, train_loader, config)
                time_toc = time.time()
                if config['dataset'] in ['mp', 'xlogp3']:
                    #train_error = np.asscalar(loss.data.cpu().numpy()) # don't test the entire train set.
                    train_error = loss
                else:
                    train_error = test(model_, train_loader, config)
                val_error = test(model_, val_loader, config)
                test_error = test(model_, test_loader, config)
                if config['lr_style'] == 'decay':
                    scheduler.step(val_error)
                saveContents.append([model_, epoch, time_toc, time_tic, train_error,  \
                        val_error, test_error, param_norm(model_), grad_norm(model_)])
                saveToResultsFile(config, saveContents[0], name='data.txt')
                best_val_error = saveModel(config, epoch, model_, best_val_error, val_error)

if __name__ == "__main__":
    #cycle_index(10,2)
    main()
