import os, sys, math, json, argparse, logging, time, random
from datetime import datetime
import torch
import numpy as np
import torchcontrib.optim

from helper import *  
from model import *
from trainer import *
from data import *
from swag import *


def parse_input_arguments():
    parser = argparse.ArgumentParser(description='Water solubility prediction')
    '''
    The followings are very basic parameters. 
    '''
    parser.add_argument('--dataset', type=str, choices=['ws', 'logp', 'mp', 'xlogp3', 'qm9', 'sol_exp', 'ccdc_sol', 'ccdc_logp', 'ccdc_sollogp', 'mp_drugs', 'mp_less', 'zinc'],
                        help='dataset for training (default: water solubility)')
    parser.add_argument('--allDataPath', type=str, default='/beegfs/dz1061/gcn/chemGraph/data')
    parser.add_argument('--running_path', type=str,
                        help='path to save model', default='/beegfs/dz1061/gcn/chemGraph/results')                    
    parser.add_argument('--use_cuda', default=True,
                        help='enables CUDA training')
    parser.add_argument('--model', type=str, \
        choices=['1-2-GNN', 'ConvGRU', '1-GNN', 'SAGE', 'GCN', 'CONVATT', 'adaConv', 'ConvSet2Set', \
                 'NNConvGAT', 'gradcam', 'dropout', 'loopybp', 'wlkernel', 'VAE'])
    parser.add_argument('--normalize', action='store_true')  # on target data
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--depths', type=int, default=3) # atom embedding times
    parser.add_argument('--NumOutLayers', type=int, default=3) # number of readout layers
    parser.add_argument('--dimension', type=int, default=64) # hidden embedding dimension
    parser.add_argument('--dropout', type=float, default=0.0) # dropout rate
    parser.add_argument('--pooling', type=str, choices=['add', 'mean', 'max', 'set2set'], default='add')
    parser.add_argument('--act_fn', type=str, choices=['relu', 'leaky_relu', 'elu', 'tanh', 'selu'], default='relu')
    parser.add_argument('--weights', type=str, choices=['he_norm', 'xavier_norm', 'he_uni', 'xavier_uni'], default='he_uni')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--loss', type=str, choices=['l1', 'l2', 'smooth_l1', 'dropout', 'vae'])
    parser.add_argument('--optimizer',  type=str, choices=['adam', 'sgd', 'swa'])
    parser.add_argument('--metrics', type=str, choices=['l1', 'l2'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_style', type=str, choices=['constant', 'decay']) # now is exponential decay on valid loss
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--patience_epochs', type=int, default=30)
    parser.add_argument('--train_type', type=str, default='from_scratch', choices=['from_scratch', 'transfer'])
    parser.add_argument('--style', type=str, choices=['base', 'CV'])  # if running CV
    parser.add_argument('--experiment', type=str)  # when doing experimenting, name it. 
    parser.add_argument('--cv_folder', type=int) # if 

    '''
    The followings are for variants of different model tests.
    '''
    parser.add_argument('--efgs', action='store_true')
    parser.add_argument('--water_interaction', action='store_true')
    parser.add_argument('--InterByConcat', action='store_true')
    parser.add_argument('--InterBySub', action='store_true')
    parser.add_argument('--mol', action='store_true')
    
    '''
    The followings are for VAE.
    '''
    parser.add_argument('--vocab_path', type=str)
    parser.add_argument('--vocab_name', type=str)
    parser.add_argument('--numEncoLayers',type=int)
    parser.add_argument('--numDecoLayers',type=int)
    parser.add_argument('--numEncoders', type=int)
    parser.add_argument('--numDecoders', type=int)
    parser.add_argument('--varDimen', type=int)
    '''
    The followings are for UQ.
    '''
    #parser.add_argument('--swagHetero', action='store_true')
    parser.add_argument('--uncertainty',  type=str, choices=['epistemic', 'aleatoric'])
    parser.add_argument('--uncertainty_method',  type=str, choices=['dropout', 'swag'])
    parser.add_argument('--swag_start', type=int, default=10)
    
    '''
    The followings are for transfer training..
    '''
    parser.add_argument('--transfer_from', type=str)
    parser.add_argument('--pre_trained_path', type=str)
    parser.add_argument('--pre_trained_model', type=str)
    parser.add_argument('--params', type=str)
    
    return parser.parse_args()

def main():
    args = parse_input_arguments()
    this_dic = vars(args)
    
    if args.style == 'base': 
        this_dic['data_path'] = os.path.join(args.allDataPath, args.dataset, 'graphs', args.style, args.model)
    if args.style == 'CV':
        this_dic['data_path'] = os.path.join(args.allDataPath, args.dataset, 'graphs', args.style, args.model, 'cv_'+str(this_dic['cv_folder'])) 
    args.running_path = os.path.join(args.running_path, args.dataset, args.model, args.experiment)
    this_dic['running_path'] = args.running_path

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

    if True:
        if not os.path.exists(os.path.join(args.running_path, 'trained_model/')): 
            os.makedirs(os.path.join(args.running_path, 'trained_model/'))
        if not os.path.exists(os.path.join(args.running_path, 'best_model/')):
            os.makedirs(os.path.join(args.running_path, 'best_model/'))
        createResultsFile(this_dic, name='data.txt')
            
        ############################## Load Data ##############################
        # --- modifications of configuration ---
        if this_dic['train_type'] == 'transfer':
            #trans_to = this_dic['dataset']
            #this_dic['dataset'] = this_dic['transfer_from']
            #this_dic['data_path'] = os.path.join(args.allDataPath, args.transfer_from, 'graphs', args.style, args.model)
            #loader = get_data_loader(this_dic)
            #_, _, _, _, _, _, num_i_2 = loader.train_loader, loader.val_loader, loader.test_loader, loader.std, loader.num_features, loader.num_bond_features, loader.num_i_2
            #del loader
            #this_dic['num_i_2'] = num_i_2
            #this_dic['dataset'] = trans_to
            this_dic['data_path'] = os.path.join(args.allDataPath, this_dic['dataset'], 'graphs', args.style, args.model)
            loader = get_data_loader(this_dic)
            train_loader, val_loader, test_loader, std, num_features, num_bond_features, num_i_2 = loader.train_loader, loader.val_loader, loader.test_loader, loader.std, loader.num_features, loader.num_bond_features, loader.num_i_2
            preTrainedConfig = loadConfig(this_dic['pre_trained_path'], name='config.json')
            this_dic.update({key:value for key, value in preTrainedConfig.items() if key in ['dimension', 'depths', 'NumOutLayers', 'num_i_2', 'num_features', 'num_bond_features']})

        elif this_dic['model'] == 'VAE':
            assert this_dic['vocab_name']
            assert this_dic['numEncoLayers']
            assert this_dic['numDecoLayers']
            assert this_dic['numEncoders']
            assert this_dic['numDecoders']
            loader = get_data_loader(this_dic)
            train_loader, val_loader, test_loader, vocab_size = loader.train_loader, loader.val_loader, loader.test_loader, loader.vocab_size
            this_dic['vocab_size'] = vocab_size
            assert this_dic['dropout'] == 0.1 
            assert this_dic['varDimen']
            assert this_dic['dimension'] % this_dic['numEncoLayers'] == 0
        
        else:
            loader = get_data_loader(this_dic)
            train_loader, val_loader, test_loader, std, num_features, num_bond_features, num_i_2 = loader.train_loader, loader.val_loader, loader.test_loader, loader.std, loader.num_features, loader.num_bond_features, loader.num_i_2
            logging.info('Loading data ready. Standard deviation value for normalization is %s' % str(std))
            this_dic['num_features'], this_dic['num_bond_features'], this_dic['num_i_2'], this_dic['std'] = int(num_features), num_bond_features, num_i_2, std
        logging.info('Done Creating data file for train/valid/test...')
 
        ############################## Load Model ##############################
        if this_dic['model'] in ['loopybp', 'wlkernel']:
            this_dic['atom_fdim'] = num_features
            this_dic['bond_fdim'] = num_bond_features
            this_dic['atom_messages'] = False
            this_dic['outDim'] = this_dic['dimension'] 

            model = get_model(this_dic)
            model_ = model(this_dic).to(device)
            model_ = init_weights(model_, this_dic)

        if this_dic['uncertainty']:
            assert this_dic['uncertainty_method']
            if this_dic['uncertainty_method'] == 'dropout':
                this_dic['weight_regularizer'] = 1e-8 / len(train_loader.dataset)
                this_dic['dropout_regularizer'] = 2. / len(train_loader.dataset)
                this_dic['model'] = this_dic['model'] + '_dropout'

                model = get_model(this_dic)
                model_ = model(this_dic).to(device)
                model_ = init_weights(model_, this_dic)
            if this_dic['uncertainty_method'] == 'swag':
                assert this_dic['swag_start']
                this_dic['model'] = this_dic['model'] + '_swag'
                model = get_model(this_dic)
                model_ = model.base(this_dic).to(device)
                model_ = init_weights(model_, this_dic)
                swag_model = SWAG(model.base, this_dic, no_cov_mat=False, max_num_models=20)
                swag_model = swag_model.to(device)   
                #if this_dic['swagHetero']:
                #    model = knn_dropout_
                #    model_ = model.base(this_dic).to(device)
                #    swag_model = SWAG(model.base, this_dic, no_cov_mat=False, max_num_models=20)
                #    swag_model = swag_model.to(device)    
        if this_dic['model'] in ['1-2-GNN', 'VAE']:        
            model = get_model(this_dic)
            #model = init_weights(model, this_dic)
            model_ = model(this_dic).to(device)
            model_ = init_weights(model_, this_dic)
        logging.info('Done Creating model for training...')
        ########################################################################


        ############################# Claim training types #######################
        '''
            If training needs from scratch, then:
                  <train_type> --> 'from_scratch';
                  <params> --> 'all';
            If training is transfer learning, then:
                  <train_type> --> 'transfer';
                  <params> --> 'part';
        '''
            
        if this_dic['train_type'] == 'transfer': # load pretrained weights.
            logging.info('Loading pre-trained models weights')
            model_dict = model_.state_dict()
            premodel_state_dict = torch.load(os.path.join(this_dic['pre_trained_path'], 'best_model/', this_dic['pre_trained_model']))
                 
            model_ = transferWeights(premodel_state_dict, model_dict, model_, this_dic) 
        #########################################################################


 
        ############################## Optimizer #################################
        if this_dic['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model_.parameters(), lr=this_dic['lr'])
        if this_dic['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(model_.parameters(), lr=this_dic['lr'], momentum=0.9, weight_decay=1e-4)
        if this_dic['optimizer'] == 'swa':
            optimizer = torchcontrib.optim.SWA(optimizer)
        if this_dic['lr_style'] == 'decay':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=5, min_lr=0.00001)
        ##########################################################################
            

        ############################# Training Body ##############################
        saveConfig(this_dic, name='config.json')
        
        best_val_error = float("inf")
        logging.info('Starting training steps...')
        ####
        # Dropout model for UQ
        ####  
        if this_dic['model'].endswith('dropout'):
            #epi_uncer_epochs, ale_uncer_epochs = np.array([]), np.array([])
            for epoch in range(1, this_dic['epochs']+1):
                saveContents = []
                time_tic = time.time()
                train_loss, train_error, _, _, _ = train_dropout(model_, optimizer, train_loader, this_dic)
                time_toc = time.time()
                val_loss, val_error = test_dropout(model_, 1, val_loader, this_dic)
                test_loss, test_error = 0.0, 0.0 # to maintain saveout files consistency
                saveContents.append([model_, epoch, time_toc, time_tic, train_loss, train_error, val_loss, \
                    val_error, test_loss, test_error, param_norm(model_), grad_norm(model_)])
                saveToResultsFile(this_dic, saveContents[0], name='data.txt')
        
        ####
        # VAE 
        ####
        elif this_dic['model'] == 'VAE':
            for epoch in range(1, this_dic['epochs']+1):
                saveContents = []
                time_tic = time.time()
                train_loss = train_VAE(model_, optimizer, train_loader, this_dic)
                time_toc = time.time()
                val_loss = test_VAE(model_, val_loader, this_dic)
                test_loss = 0.0
                saveContents.append([model_, epoch, time_toc, time_tic, train_loss, val_loss, \
                   test_loss, param_norm(model_), grad_norm(model_)])
                saveToResultsFile(this_dic, saveContents[0], name='data.txt')        
        
        ####
        # properties prediction
        ####
        else:   # for 1-2-GNN, wlkernel, loopybp 
            for epoch in range(1, this_dic['epochs']+1):
                saveContents = []
                time_tic = time.time()
                #lr = scheduler.optimizer.param_groups[0]['lr']
                loss = train(model_, optimizer, train_loader, this_dic)
                if this_dic['optimizer'] == 'swa':
                    optimizer.update_swa()
                time_toc = time.time()
                if this_dic['dataset'] in ['mp', 'xlogp3']:
                    #train_error = np.asscalar(loss.data.cpu().numpy()) # don't test the entire train set.
                    train_error = loss
                else:
                    train_error = test(model_, train_loader, this_dic)
                val_error = test(model_, val_loader, this_dic)
                test_error = test(model_, test_loader, this_dic)
                if this_dic['lr_style'] == 'decay':
                    scheduler.step(val_error)
                if not this_dic['uncertainty']:
                    saveContents.append([model_, epoch, time_toc, time_tic, train_error,  \
                        val_error, test_error, param_norm(model_), grad_norm(model_)])
                    saveToResultsFile(this_dic, saveContents[0], name='data.txt')
                
                ####
                # SWAG for UQ
                ####
                if this_dic['model'].endswith('swag'):
                    saveContents = []
                    if epoch > this_dic['swag_start']:
                        swag_model.collect_model(model_) # swag model processing ways 
                        swag_model.sample()
                        swag_train_error = test(swag_model, train_loader, this_dic)
                        swag_val_error = test(swag_model, val_loader, this_dic)
                        swag_test_error = test(swag_model, test_loader, this_dic)
                    else:
                        swag_train_error, swag_val_error, swag_test_error = 0.0, 0.0, 0.0
                    saveContents.append([swag_model, epoch, time_toc, time_tic, train_error, val_error, test_error, \
                    swag_train_error, swag_val_error, swag_test_error, param_norm(model_), grad_norm(model_)])
                    saveToResultsFile(this_dic, saveContents[0], name='data.txt')
                
                
                ####
                # stopping criteria
                ####
                if args.early_stopping:
                    if best_val_error > np.sum(val_error):
                        patience = 0
                        best_val_error = np.sum(val_error)
                    else:
                        patience += 1
                        if patience > args.patience_epochs:
                            logging.info('Early stopping! No consecutive improvements for %s epochs' % int(patience-1))
                            logging.info('Saving models...')
                            torch.save(model_.state_dict(), os.path.join(args.running_path, 'best_model'))
                            logging.info('Model saved.')
                            break
                else:
                    if best_val_error > np.sum(val_error):
                        best_val_error = np.sum(val_error)
                        logging.info('Saving models...')
                        torch.save(model_.state_dict(), os.path.join(args.running_path, 'best_model', 'model_'+str(epoch)+'.pt'))
                        if this_dic['model'].endswith('swag'):
                            torch.save(swag_model.state_dict(), os.path.join(args.running_path, 'best_model', 'swag_model_'+str(epoch)+'.pt'))


if __name__ == '__main__':
    main()



