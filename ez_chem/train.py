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
    parser.add_argument('--output_model_file', type=str, default = '', help='filename to output the model')
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--model', type=str, default="1-GNN")
    parser.add_argument('--EFGS', action='store_true')
    parser.add_argument('--residual_connect', action='store_true')
    parser.add_argument('--resLayer', type=int, default=-1)
    parser.add_argument('--solvent', type=str, default="")
    parser.add_argument('--interaction', type=str, default="")
    parser.add_argument('--pooling', type=str, default='sum')
    parser.add_argument('--aggregate', type=str, default='add')
    parser.add_argument('--degree', type=int, default=0)
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--train_type', type=str, default='from_scratch', choices=['from_scratch', 'transfer', 'hpsearch', 'finetuning'])
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
    parser.add_argument('--uncertainty_method',  type=str)
    parser.add_argument('--swag_start', type=int)
    parser.add_argument('--preTrainedData', type=str)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    this_dic = vars(args)
    
    if args.style == 'base': 
        this_dic['data_path'] = os.path.join(args.allDataPath, args.dataset, 'graphs', args.style, args.model)
    if args.style == 'CV':
        this_dic['data_path'] = os.path.join(args.allDataPath, args.dataset, 'graphs', args.style, args.model, 'cv_'+str(this_dic['cv_folder'])) 
    this_dic['running_path'] = os.path.join(args.running_path, args.dataset, args.model, args.gnn_type, args.experiment)
    if args.style == 'preTraining':
        this_dic['data_path'] = os.path.join(args.allDataPath, args.dataset, 'graphs/base', 'COMPLETE', args.model)
    if args.solvent:
        if args.style == 'preTraining':
            this_dic['data_path'] = os.path.join(args.allDataPath, args.dataset, 'graphs/base', 'COMPLETE', args.model, args.solvent)
        else:
            this_dic['data_path'] = os.path.join(args.allDataPath, args.dataset, 'graphs/base', args.model, args.solvent)
        assert this_dic['interaction']
    
    if not os.path.exists(os.path.join(args.running_path, 'trained_model/')):
        os.makedirs(os.path.join(args.running_path, 'trained_model/'))
    if not os.path.exists(os.path.join(args.running_path, 'best_model/')):
        os.makedirs(os.path.join(args.running_path, 'best_model/'))

    if args.dataset == 'calcSolLogP':
       this_dic['taskType'] = 'multi'
       args.num_tasks = 3
    elif args.dataset == 'commonProperties':
       this_dic['taskType'] = 'multi'
       args.num_tasks = 4
    else:
       this_dic['taskType'] = 'single'

    createResultsFile(this_dic, name='data.txt')

    with open('/scratch/dz1061/gcn/chemGraph/configs/splitsize.json', 'r') as f:
        dataSizes = json.load(f)
    #this_dic['train_size'] = int(dataSizes[args.dataset]['train_size'])
    #this_dic['val_size'] = int(dataSizes[args.dataset]['val_size'])
    if args.dataset in ['sol_calc/ALL', 'solOct_calc/ALL', 'logp_calc/ALL', 'calcSolLogP', 'xlogp3', 'solWithWater_calc/ALL'] and args.style == 'preTraining':
        this_dic['train_size'] = int(dataSizes[args.dataset+'/COMPLETE']['train_size'])
        this_dic['val_size'] = int(dataSizes[args.dataset+'/COMPLETE']['val_size'])
    else:
        this_dic['train_size'] = int(dataSizes[args.dataset]['train_size'])
        this_dic['val_size'] = int(dataSizes[args.dataset]['val_size'])

    this_dic['gnn_type'] = args.gnn_type
    #this_dic['data_path'] = '/beegfs/dz1061/gcn/chemGraph/data/zinc/graphs/preTraining/'+args.gnn_type
    this_dic['JK'] = args.JK
    if this_dic['pooling'] == 'set2set':
        this_dic['pooling'] = list()
        this_dic['pooling'].append('set2set')
        this_dic['pooling'].append('2')

    loader = get_data_loader(this_dic)
    train_loader, val_loader, test_loader, std, num_features, num_bond_features, num_i_2 = loader.train_loader, loader.val_loader, loader.test_loader, loader.std, loader.num_features, loader.num_bond_features, loader.num_i_2
    this_dic['num_features'], this_dic['num_bond_features'], this_dic['num_i_2'], this_dic['std'] = int(num_features), num_bond_features, num_i_2, std
    this_dic['train_size'], this_dic['val_size'], this_dic['test_size'] = len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)
    this_dic['num_layer'] = args.num_layer
    if args.gnn_type == 'pnaconv':
        deg = torch.zeros(7, dtype=torch.long)
        for data in train_loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        this_dic['degree'] = deg

    if args.style == 'best': # todo
        with open(os.path.join(args.running_path, args.dataset, args.model, args.gnn_type, 'Exp_{}_l2/seed_31'.format(args.pooling), 'hpsearch.json'), 'r') as f:
            best_config = json.load(f)
        this_dic.update(best_config)

    if args.EFGS:
        this_dic['efgs_lenth'] = len(vocab)
    if args.interaction and args.dataset in ['ws', 'mp', 'deepchem/delaney']:
        this_dic['soluteSelf'] = True
    else:
        this_dic['soluteSelf'] = False

    if args.residual_connect:
        this_dic['resLayer'] = args.resLayer
    model = get_model(this_dic)

    if this_dic['train_type'] == 'from_scratch':
        model = init_weights(model, this_dic)
    if this_dic['train_type'] in ['finetuning', 'transfer']:
        #list_of_files = glob.glob(os.path.join('/scratch/dz1061/gcn/chemGraph/results',args.preTrainedData, args.model, args.gnn_type, 'Pre_training', 'seed_31', 'best_model/', '*.pt'))
        #latest_file = max(list_of_files, key=os.path.getctime)
        #latest_file = os.path.join('/scratch/dz1061/gcn/chemGraph/results',args.preTrainedData, args.model, args.gnn_type, 'preTraining', 'Exp_layer5_dim64', 'seed_31', 'best_model/', 'bestModel.pt')
        if args.model in ['1-GNN']:
            latest_file = os.path.join('/scratch/dz1061/gcn/chemGraph/results',args.preTrainedData, '1-GNN', args.gnn_type, 'preTraining/all', 'Exp_layer5_dim256', 'seed_1', 'best_model/', 'bestModel.pt')
            model.from_pretrained(latest_file)

        if args.model == '1-2-GNN':
            #latest_file = os.path.join('/scratch/dz1061/gcn/chemGraph/results',args.preTrainedData, '1-2-GNN', args.gnn_type, 'preTraining', 'Exp_layer5_dim64', 'seed_31', 'best_model/', 'bestModel.pt')  #from sol_calc
            latest_file = os.path.join('/scratch/dz1061/gcn/chemGraph/results',args.preTrainedData, '1-2-GNN', args.gnn_type, 'newFinetuningFromws', 'Exp_layer5_dim256_lr0.001_decay', 'seed_31', 'best_model/', 'bestModel.pt') # from finetuning on itself
            #latest_file = os.path.join('/scratch/dz1061/gcn/chemGraph/results',args.preTrainedData, '1-2-GNN', args.gnn_type, 'newData', 'Exp_layer5_dim64', 'seed_31', 'best_model/', 'bestModel.pt') # from train from scratch itself
            #latest_file = os.path.join('/scratch/dz1061/gcn/chemGraph/results',args.preTrainedData, '1-2-GNN', args.gnn_type, 'preTraining/all', 'Exp_layer5_dim256', 'seed_1', 'best_model/', 'bestModel.pt')
            #conv1_file = os.path.join('/scratch/dz1061/gcn/chemGraph/results',args.preTrainedData, '1-2-GNN', args.gnn_type, 'loopFinetuningFrompka', 'Exp_layer5_dim64', 'seed_31', 'best_model/', 'convISO1.pt')
            #conv2_file = os.path.join('/scratch/dz1061/gcn/chemGraph/results',args.preTrainedData, '1-2-GNN', args.gnn_type, 'loopFinetuningFrompka', 'Exp_layer5_dim64', 'seed_31', 'best_model/', 'convISO2.pt')
            model.from_pretrained(latest_file)

        if args.model in ['1-interaction-GNN']:
            solute_latest_file = os.path.join('/scratch/dz1061/gcn/chemGraph/results', args.preTrainedData, '1-interaction-GNN', args.gnn_type, 'preTraining/all', 'Exp_layer5_dim128', 'seed_1', 'best_model/', 'bestModel_solute.pt')
            solvent_latest_file = os.path.join('/scratch/dz1061/gcn/chemGraph/results', args.preTrainedData, '1-interaction-GNN', args.gnn_type, 'preTraining/all', 'Exp_layer5_dim128', 'seed_1', 'best_model/', 'bestModel_hydrated_solute.pt')
            model.from_pretrained(solute_latest_file, solvent_latest_file)

    if this_dic['train_type'] == 'transfer': # freeze encoder layers
        if this_dic['model'] in ['1-GNN']:
            for params in model.gnn.parameters():
                params.requires_grad = False
        if this_dic['model'] in ['1-2-GNN']:
            for params in model.gnn.parameters():
                params.requires_grad = False
            #for params in model.ISO1.parameters():
            #    params.requires_grad = False
            #for params in model.ISO2.parameters():
            #    params.requires_grad = False
        if this_dic['model'] in ['1-interaction-GNN']:
            for params in model.gnn_solute.parameters():
                params.requires_grad = False
            for params in model.gnn_solvent.parameters():
                params.requires_grad = False



    this_dic['NumParas'] = count_parameters(model)
    model_ = model.to(device)
    
    if this_dic['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model_.parameters(), lr=this_dic['lr'])
    if this_dic['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model_.parameters(), lr=this_dic['lr'], momentum=0.9, weight_decay=1e-4)
    if this_dic['optimizer'] == 'swa':
        optimizer = torchcontrib.optim.SWA(optimizer)
    if this_dic['lr_style'] == 'decay':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=5, min_lr=0.00001)

    saveConfig(this_dic, name='config.json')
    best_val_error = float("inf")
    for epoch in range(1, this_dic['epochs']+1):
         saveContents = []
         time_tic = time.time()
         #lr = scheduler.optimizer.param_groups[0]['lr']
         loss = train(model_, optimizer, train_loader, this_dic)
         time_toc = time.time()

         if this_dic['dataset'] in ['mp', 'mp_drugs', 'xlogp3', 'calcLogP/ALL', 'sol_calc/ALL', 'solOct_calc/ALL', 'solWithWater_calc/ALL', 'solOctWithWater_calc/ALL', 'calcLogPWithWater/ALL', 'qm9/nmr/carbon', 'qm9/nmr/hydrogen']:
            train_error = loss
            #train_error = np.asscalar(loss.data.cpu().numpy()) # don't test the entire train set.
            #train_error = test(model_, loader.part_train_loader, this_dic)
         else:
            train_error = test(model_, train_loader, this_dic)
         val_error = test(model_, val_loader, this_dic)

         if args.style in ['base', 'preTraining']:
            test_error = test(model_, test_loader, this_dic)
         #if args.style == 'preTraining':
         #   test_error = 0.
         if this_dic['lr_style'] == 'decay':
            scheduler.step(val_error)

         if not this_dic['uncertainty']:
            saveContents.append([model_, epoch, time_toc, time_tic, train_error,  \
                val_error, test_error, param_norm(model_), grad_norm(model_)])
            saveToResultsFile(this_dic, saveContents[0], name='data.txt')
            best_val_error = saveModel(this_dic, epoch, model_, best_val_error, val_error)
         torch.save(model.state_dict(), os.path.join(this_dic['running_path'], 'trained_model', 'model_last.pt'))

if __name__ == "__main__":
    #cycle_index(10,2)
    main()

