import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from tqdm import tqdm
import pandas as pd
import numpy as np

from pre_train_model import *
from pre_train_utils import *
from sklearn.metrics import roc_auc_score
##from loader import *
#from loader import MoleculeDataset
from helper import *


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
    parser.add_argument('--csize', type=int, default=3,
                        help='context size (default: 3).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--neg_samples', type=int, default=1,
                        help='number of negative contexts per positive context (default: 1)')
    parser.add_argument('--context_pooling', type=str, default="mean",
                        help='how the contexts are pooled (sum, mean, or max)')
    parser.add_argument('--mode', type=str, default = "cbow", help = "cbow or skipgram")
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default = '', help='filename to output the model')
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    l1 = args.num_layer - 1
    l2 = l1 + args.csize

    #print(args.mode)
    #print("num layer: %d l1: %d l2: %d" %(args.num_layer, l1, l2))
    
    config = loadConfig('/beegfs/dz1061/gcn/chemGraph/results/logp/1-2-GNN/TransFromALLcalcSolLoP/base/')
    config['gnn_type'] = args.gnn_type
    config['data_path'] = '/beegfs/dz1061/gcn/chemGraph/data/zinc/graphs/preTraining/'+args.gnn_type
    config['JK'] = args.JK
    #set up dataset and transform function.
    #dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset, transform = ExtractSubstructureContextPair_rev(args.num_layer, l1, l2))
    if config['gnn_type'] == 'GIN':
       num_bond_feas = 2
    else:
       num_bond_feas = 7
    dataset = knnGraph_rev(
                            root=config['data_path'],
                            transform=ExtractSubstructureContextPair_rev(args.num_layer, l1, l2, num_bond_feas),
                            pre_transform=MyPreTransform(),
                            pre_filter=MyFilter())
    
    train_loader = DataLoaderSubstructContext_rev(dataset[:80000], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoaderSubstructContext_rev(dataset[80000:], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    writer = SummaryWriter('/beegfs/dz1061/gcn/chemGraph/results/preTrain/{}/contextPred'.format(args.gnn_type))
    #set up models, one for pre-training and one for context embeddings
    config['num_layer'] = args.num_layer
    if config['gnn_type'] == 'GIN':
       model_substruct = GNNPretrain(config).to(device)
       config['num_layer'] = int(l2 - l1)
       model_context = GNNPretrain(config).to(device)
    if config['gnn_type'] == 'GCN':
       model_substruct = gnnBase(config).to(device)
       config['num_layer'] = int(l2 - l1)
       model_context = gnnBase(config).to(device)

    #set up optimizer for the two GNNs
    optimizer_substruct = optim.Adam(model_substruct.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_context = optim.Adam(model_context.parameters(), lr=args.lr, weight_decay=args.decay)
    
    config['running_path'] = '/'.join(args.output_model_file.split('/')[:-1])
    saveConfig(config, name='config.json')   
    best_valid_acc = 0
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train_loss, train_acc = train(args, model_substruct, model_context, train_loader, optimizer_substruct, optimizer_context, device)
        valid_loss, valid_acc = train(args, model_substruct, model_context, val_loader, optimizer_substruct, optimizer_context, device)
        writer.add_scalar('data/train loss', train_loss, epoch)
        writer.add_scalar('data/train acc', train_acc, epoch)
        writer.add_scalar('data/valid loss', valid_loss, epoch)
        writer.add_scalar('data/valid acc', valid_acc, epoch)
        if valid_acc > best_valid_acc:
           torch.save(model_substruct.state_dict(), args.output_model_file + "_" + str(epoch)+".pth")
           best_valid_acc = valid_acc

    if not args.output_model_file == "":
        torch.save(model_substruct.state_dict(), args.output_model_file + "_finalEpoch.pth")

if __name__ == "__main__":
    #cycle_index(10,2)
    main()
