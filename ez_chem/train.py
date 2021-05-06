import argparse, time, os 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from tensorboardX import SummaryWriter
from tqdm import tqdm
##from loader import *
#from loader import MoleculeDataset
from helper import *
from data import *
from trainer import *
from models import *

def main():
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
    parser.add_argument('--interaction_simpler', action='store_true')
    parser.add_argument('--pooling', type=str, default='sum')
    parser.add_argument('--aggregate', type=str, default='add')
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
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    this_dic = vars(args)
    # define a path to save training results: save models and data
    this_dic['running_path'] = os.path.join(args.running_path, args.dataset, args.model, args.gnn_type, args.experiment) 
    if not os.path.exists(os.path.join(args.running_path, 'trained_model/')):
        os.makedirs(os.path.join(args.running_path, 'trained_model/'))
    if not os.path.exists(os.path.join(args.running_path, 'best_model/')):
        os.makedirs(os.path.join(args.running_path, 'best_model/'))
    createResultsFile(this_dic, name='data.txt')

    # define path to load data for different training tasks
    if args.style == 'base': 
        this_dic['data_path'] = os.path.join(args.allDataPath, args.dataset, 'graphs', args.style, args.model)
    #if args.style == 'CV':
    #    this_dic['data_path'] = os.path.join(args.allDataPath, args.dataset, 'graphs', args.style, args.model, 'cv_'+str(this_dic['cv_folder'])) 
    if args.style == 'preTraining':
        this_dic['data_path'] = os.path.join(args.allDataPath, args.dataset, 'graphs/base', 'COMPLETE', args.model)

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

    # load processed data 
    loader = get_data_loader(this_dic)
    train_loader, val_loader, test_loader, std, num_features, num_bond_features, num_i_2 = loader.train_loader, loader.val_loader, loader.test_loader, loader.std, loader.num_features, loader.num_bond_features, loader.num_i_2
    this_dic['num_features'], this_dic['num_bond_features'], this_dic['num_i_2'], this_dic['std'] = int(num_features), num_bond_features, num_i_2, std
    this_dic['train_size'], this_dic['val_size'], this_dic['test_size'] = len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)

    # for special cases 
    if args.EFGS:
        this_dic['efgs_lenth'] = len(vocab)
    if args.residual_connect:
        this_dic['resLayer'] = args.resLayer
    
    # loading model
    model = get_model(this_dic)
    if this_dic['train_type'] == 'from_scratch': 
        model = init_weights(model, this_dic)
    if this_dic['train_type'] in ['finetuning', 'transfer']:
        model.from_pretrained(preTrainedPath) # load weights for encoders 
    
    if this_dic['train_type'] == 'transfer': # freeze encoder layers
        if this_dic['model'] in ['1-GNN']:
            for params in model.gnn.parameters():
                params.requires_grad = False
        if this_dic['model'] in ['1-2-GNN']:
            for params in model.gnn.parameters():
                params.requires_grad = False
    
    

    # count total # of trainable params
    this_dic['NumParas'] = count_parameters(model)
    # save out all input parameters 
    saveConfig(this_dic, name='config.json')
    
    # training parts
    model_ = model.to(device)
    optimizer = get_optimizer(config['optimizer'], model_)
    if this_dic['lr_style'] == 'decay':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=5, min_lr=0.00001)
    if this_dic['uncertainty']:
        optimizer = torchcontrib.optim.SWA(optimizer)
    
    best_val_error = float("inf")
    if args.OnlyPrediction:
        this_dic['epochs'] = 1
        model_.eval()
    for epoch in range(1, this_dic['epochs']+1):
        saveContents = []
        time_tic = time.time() # starting time
        lr = scheduler.optimizer.param_groups[0]['lr']
        if not args.OnlyPrediction:
            loss = train(model_, optimizer, train_loader, this_dic)
        else:
            loss = 0.
        time_toc = time.time() # ending time 

        if this_dic['dataset'] in ['mp', 'mp_drugs', 'xlogp3', 'calcLogP/ALL', 'sol_calc/ALL', \
             'solOct_calc/ALL', 'solWithWater_calc/ALL', 'solOctWithWater_calc/ALL', 'calcLogPWithWater/ALL', \
                 'qm9/nmr/carbon', 'qm9/nmr/hydrogen', 'qm9/nmr/allAtoms', 'calcSolLogP/ALL']:
            train_error = loss # coz train set is too large to be tested every epoch
        else:
            train_error = test(model_, train_loader, this_dic)
        val_error = test(model_, val_loader, this_dic)
        test_error = test(model_, test_loader, this_dic)
        if this_dic['lr_style'] == 'decay':
            scheduler.step(val_error)

        if not this_dic['uncertainty']:
            saveContents.append([epoch, lr, time_toc-time_tic, train_error,  \
                val_error, test_error, param_norm(model_), grad_norm(model_)])
            saveToResultsFile(this_dic, saveContents[0], name='data.txt')
            best_val_error = saveModel(this_dic, epoch, model_, best_val_error, val_error) # save model if validation error hits new lower 
        torch.save(model.state_dict(), os.path.join(this_dic['running_path'], 'trained_model', 'model_last.pt'))

if __name__ == "__main__":
    #cycle_index(10,2)
    main()

