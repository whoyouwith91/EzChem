import argparse, time, os, warnings
import torch
from args import *
from helper import *
from data import *
from trainer import *
from models import *
from utils_functions import floating_type
from Optimizers import EmaAmsGrad
from kwargs import *

def main():
    warnings.filterwarnings("ignore")
    args = get_parser()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    this_dic = vars(args)
    this_dic['device'] = device
    # define a path to save training results: save models and data
    this_dic['running_path'] = os.path.join(args.running_path, args.dataset, args.model, args.gnn_type, args.experiment) 
    if not os.path.exists(os.path.join(args.running_path, 'trained_model/')):
        os.makedirs(os.path.join(args.running_path, 'trained_model/'))
    if not os.path.exists(os.path.join(args.running_path, 'best_model/')):
        os.makedirs(os.path.join(args.running_path, 'best_model/'))
    results = createResultsFile(this_dic) # create pretty table

    # define task type: multi or single, see helper
    this_dic = getTaskType(this_dic)
    # ------------------------------------load processed data-----------------------------------------------------------------------------
    this_dic['data_path'] = os.path.join(args.allDataPath, args.dataset, 'graphs', args.model, args.style)
    loader = get_data_loader(this_dic)
    train_loader, val_loader, test_loader, num_atom_features, num_bond_features = loader.train_loader, loader.val_loader, loader.test_loader, loader.num_features, loader.num_bond_features
    this_dic['num_atom_features'], this_dic['num_bond_features'] = int(num_atom_features), num_bond_features
    this_dic['train_size'], this_dic['val_size'], this_dic['test_size'] = len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)
    
    # use scale and shift
    if args.normalize: 
        #this_dic = getScaleandShift(this_dic)
        this_dic = getScaleandShift_from_scratch(this_dic, train_loader)
    
    #-----------------------------------loading model------------------------------------------------------------------------------------
    if args.model in ['physnet']: # loading physnet params
        this_dic['n_feature'] = this_dic['emb_dim']
        this_dic = {**this_dic, **physnet_kwargs}
    if this_dic['gnn_type'] == 'pnaconv': #
        if this_dic['dataset'].startswith('protein'): d = 8
        elif 'hydrogen' in this_dic['dataset']: d = 7
        else: d = 6
        this_dic['deg'], this_dic['deg_value'] = getDegreeforPNA(train_loader, d)

    model = get_model(this_dic)
    # model weights initializations
    if this_dic['model'] not in ['physnet']: 
    #if this_dic['train_type'] == 'from_scratch' and this_dic['model'] not in ['physnet']: 
        model = init_weights(model, this_dic)
    
    if this_dic['train_type'] in ['finetuning', 'transfer']:
        if args.normalize:
            state_dict = torch.load(os.path.join(args.preTrainedPath, 'best_model', 'model_best.pt'), map_location=torch.device('cpu'))
            #state_dict.update({key:value for key,value in model.state_dict().items() if key in ['scale', 'shift']}) # scale and shift from new train loader
            #model.load_state_dict(state_dict) 
            own_state = model.state_dict()
            for name, param in state_dict.items():
                #if name.startswith('gnn'):
                own_state[name].copy_(param)
        else:
            model.from_pretrained(os.path.join(args.preTrainedPath, 'best_model', 'model_best.pt')) # load weights for encoders 

    if this_dic['train_type'] == 'transfer': # freeze encoder layers
        if args.model in ['physnet']:
            model.freeze_prev_layers(freeze_extra=True)
        else:
            for params in model.gnn.parameters():
                params.requires_grad = False

    # count total # of trainable params
    this_dic['NumParas'] = count_parameters(model)
    # save out all input parameters 
    saveConfig(this_dic, name='config.json')
    
    # ----------------------------------------training parts----------------------------------------------------------------------------
    if args.model in ['physnet']: 
        model_ = model.type(floating_type).to(device)
    else:
        model_ = model.to(device)
    
    if args.optimizer == 'EMA': # only for physnet
        shadow_model = get_model(this_dic).to(device)
        if args.model in ['physnet']: 
            shadow_model = shadow_model.type(floating_type)
        shadow_model.load_state_dict(model_.state_dict())
        optimizer = EmaAmsGrad(model_.parameters(), shadow_model, lr=this_dic['lr'], ema=0.999)
    else: 
        optimizer = get_optimizer(args, model_)
    scheduler = build_lr_scheduler(optimizer, this_dic)
    
    best_val_error = float("inf")
    for epoch in range(this_dic['epochs']+1):
        # testing parts
        if this_dic['dataset'] in large_datasets: # see helper about large datasets
            train_error = 0. # coz train set is too large to be tested every epoch
        else:
            train_error = test_model(this_dic)(model_, train_loader, this_dic) # test on entire dataset
        if args.optimizer == 'EMA': # physnet
            shadow_model = optimizer.shadow_model
            val_error = test_model(this_dic)(shadow_model, val_loader, this_dic)
            test_error = test_model(this_dic)(shadow_model, test_loader, this_dic, onData='test')
        else:
            val_error = test_model(this_dic)(model_, val_loader, this_dic) # test on entire dataset
            test_error = test_model(this_dic)(model_, test_loader, this_dic, onData='test') # test on entire dataset

        # model saving
        if args.optimizer == 'EMA': # physnet
                best_val_error = saveModel(this_dic, epoch, shadow_model, best_val_error, val_error)
        else:
            if this_dic['loss'] in ['class']:
                val_error = 0 - val_error
            best_val_error = saveModel(this_dic, epoch, model_, best_val_error, val_error) # save model if validation error hits new lower 

        # training parts
        time_tic = time.time() # ending time 
        if this_dic['scheduler'] == 'const': lr = args.lr
        elif this_dic['scheduler'] in ['NoamLR', 'step']: lr = scheduler.get_lr()[0]
        else:lr = scheduler.optimizer.param_groups[0]['lr'] # decaying on val error

        loss = train_model(this_dic)(model_, optimizer, train_loader, this_dic, scheduler=scheduler) # training loss
        time_toc = time.time() # ending time 
        
        if this_dic['scheduler'] == 'decay':
            scheduler.step(val_error)
        
        # write out models and results
        if not this_dic['uncertainty']:
            if this_dic['dataset'] == 'solEFGs' and this_dic['propertyLevel'] == 'atom':
                contents = [epoch, round(time_toc-time_tic, 2), round(lr,7), round(train_error[0],3), round(train_error[1],3),  \
                round(val_error,3), round(test_error,3), round(param_norm(model_),2), round(grad_norm(model_),2)]
            else:
                contents = [epoch, round(time_toc-time_tic, 2), round(lr,7), round(train_error,6),  \
                round(val_error,6), round(test_error,6), round(param_norm(model_),2), round(grad_norm(model_),2)]
            results.add_row(contents) # updating pretty table 
            saveToResultsFile(results, this_dic, name='data.txt') # save instant data to directory
            
        else: 
            contents = [epoch, round(time_toc-time_tic, 2), round(lr,7), round(train_error,6),  \
                round(val_error,6), round(test_error,6), round(param_norm(model_),2), round(grad_norm(model_),2)]
            results.add_row(contents) # updating pretty table 
            saveToResultsFile(results, this_dic, name='data.txt') # save instant data to directory
            best_val_error = saveModel(this_dic, epoch, model_, best_val_error, val_error)
        
        if args.model in ['physnet']: 
            torch.save(shadow_model.state_dict(), os.path.join(this_dic['running_path'], 'trained_model', 'model_last.pt'))
        else:
            torch.save(model_.state_dict(), os.path.join(this_dic['running_path'], 'trained_model', 'model_last.pt'))

if __name__ == "__main__":
    #cycle_index(10,2)
    main()

