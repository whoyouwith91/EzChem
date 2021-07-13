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
    if args.dataset == 'solNMR' and args.propertyLevel == 'atomMol':
        args.num_tasks = 2
        this_dic['taskType'] = 'multi'
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
    train_loader, val_loader, test_loader, num_atom_features, num_bond_features, num_i_2 = loader.train_loader, loader.val_loader, loader.test_loader, loader.num_features, loader.num_bond_features, loader.num_i_2
    this_dic['num_atom_features'], this_dic['num_bond_features'], this_dic['num_i_2'] = int(num_atom_features), num_bond_features, num_i_2
    this_dic['train_size'], this_dic['val_size'], this_dic['test_size'] = len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)
    if args.model in ['physnet']:
        if args.dataset in ['qm9/nmr/allAtoms']: # loading physnet params
            energy_shift = torch.tensor([67.2858])
            energy_scale = torch.tensor([85.8406])
        if args.dataset in ['qm9/nmr/carbon']: # loading physnet params
            energy_shift = torch.tensor([115.9782138561384])
            energy_scale = torch.tensor([51.569003335315905])
        if args.dataset in ['qm9/nmr/hydrogen']: # loading physnet params
            energy_shift = torch.tensor([29.08285732440852])
            energy_scale = torch.tensor([1.9575037908857158])
        if args.dataset in ['qm9/u0']:
            energy_shift = torch.tensor([-4.1164152221029555])
            energy_scale = torch.tensor([0.9008408776783313])
        if args.dataset in ['nmr/carbon']:
            energy_shift = torch.tensor([98.23850877956372])
            energy_scale = torch.tensor([51.27542605786456])
        if args.dataset in ['nmr/hydrogen']:
            energy_shift = torch.tensor([4.707991621093338])
            energy_scale = torch.tensor([2.6513451307981577])
        if args.dataset in ['solNMR']:
            if args.test_level == 'molecule':
                energy_shift = torch.tensor([-7.3204])
                energy_scale = torch.tensor([4.7854])
            if args.test_level == 'atom':
                energy_shift = torch.tensor([75.4205])
                energy_scale = torch.tensor([151.0311])

        this_dic['energy_shift'], this_dic['energy_shift_value'] = energy_shift, energy_shift.item()
        this_dic['energy_scale'], this_dic['energy_scale_value'] = energy_scale, energy_scale.item()

    # for special cases 
    if args.EFGS:
        this_dic['efgs_lenth'] = len(vocab)
    
    # loading model
    if args.model in ['physnet']: # loading physnet params
        this_dic['n_feature'] = this_dic['emb_dim']
        this_dic = {**this_dic, **physnet_kwargs}
    model = get_model(this_dic)
    # model weights initializations
    if this_dic['train_type'] == 'from_scratch' and this_dic['model'] not in ['physnet']: 
        model = init_weights(model, this_dic)
    if this_dic['train_type'] in ['finetuning', 'transfer']:
        if args.model in ['physnet']:
            state_dict = torch.load(os.path.join(args.preTrainedPath, 'best_model', 'model_best.pt'))
            state_dict.update({key:value for key,value in model.state_dict().items() if key in ['scale', 'shift']})
            model.load_state_dict(state_dict)
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
    
    # training parts
    if args.model in ['physnet']: 
        model_ = model.type(floating_type).to(device)
    else:
        model_ = model.to(device)
    if args.optimizer == 'EMA':
        shadow_model = get_model(this_dic).to(device)
        if args.model in ['physnet']: 
            shadow_model = shadow_model.type(floating_type)
        shadow_model.load_state_dict(model_.state_dict())
        optimizer = EmaAmsGrad(model_.parameters(), shadow_model, lr=this_dic['lr'], ema=0.999)
    else:
        optimizer = get_optimizer(args, model_)
    scheduler = build_lr_scheduler(optimizer, this_dic)
    if this_dic['uncertainty']:
        optimizer = torchcontrib.optim.SWA(optimizer)
    
    best_val_error = float("inf")
    if args.OnlyPrediction:
        this_dic['epochs'] = 1
        model_.eval()
    for epoch in range(1, this_dic['epochs']+1):
        time_tic = time.time() # ending time 
        if this_dic['scheduler'] == 'const': lr = args.lr
        elif this_dic['scheduler'] in ['NoamLR', 'step']: lr = scheduler.get_lr()[0]
        else:lr = scheduler.optimizer.param_groups[0]['lr'] # decaying on val error

        if not args.OnlyPrediction:
            loss = train_model(this_dic)(model_, optimizer, train_loader, this_dic, scheduler=scheduler) # training loss
        else:
            loss = 0.
        time_toc = time.time() # ending time 

        # testing parts
        if this_dic['dataset'] in ['mp', 'mp_drugs', 'xlogp3', 'calcLogP/ALL', 'sol_calc/ALL', \
             'solOct_calc/ALL', 'solWithWater_calc/ALL', 'solOctWithWater_calc/ALL', 'calcLogPWithWater/ALL', \
                 'qm9/nmr/carbon', 'qm9/nmr/hydrogen', 'qm9/nmr/allAtoms', 'calcSolLogP/ALL', 'nmr/carbon', 'nmr/hydrogen']:
            train_error = loss # coz train set is too large to be tested every epoch
        else:
            train_error = test_model(this_dic)(model_, train_loader, this_dic) # test on entire dataset
        if args.optimizer == 'EMA':
            shadow_model = optimizer.shadow_model
            val_error = test_model(this_dic)(shadow_model, val_loader, this_dic)
            test_error = test_model(this_dic)(shadow_model, test_loader, this_dic)
        else:
            val_error = test_model(this_dic)(model_, val_loader, this_dic) # test on entire dataset
            test_error = test_model(this_dic)(model_, test_loader, this_dic) # test on entire dataset
        if this_dic['scheduler'] == 'decay':
            scheduler.step(val_error)

        # write out models and results
        if not this_dic['uncertainty']:
            #assert torch.is_tensor(train_error)
            #assert torch.is_tensor(val_error)
            #assert torch.is_tensor(test_error)
            contents = [epoch, round(time_toc-time_tic, 2), round(lr,7), round(train_error,3),  \
                round(val_error,3), round(test_error,3), round(param_norm(model_),2), round(grad_norm(model_),2)]
            results.add_row(contents) # updating pretty table 
            saveToResultsFile(results, this_dic, name='data.txt') # save instant data to directory
            if args.optimizer == 'EMA':
                best_val_error = saveModel(this_dic, epoch, shadow_model, best_val_error, val_error)
            else:
                best_val_error = saveModel(this_dic, epoch, model_, best_val_error, val_error) # save model if validation error hits new lower 
    torch.save(model_.state_dict(), os.path.join(this_dic['running_path'], 'trained_model', 'model_last.pt'))

if __name__ == "__main__":
    #cycle_index(10,2)
    main()

