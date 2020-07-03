"""Optimizes hyperparameters using Bayesian optimization."""

from copy import deepcopy
import json
from typing import Dict, Union
import os

from hyperopt import fmin, hp, tpe
import numpy as np

from chemprop.models import build_model
from chemprop.nn_utils import param_count


SPACE = {
    'dimension': hp.quniform('dimension', low=64, high=256, q=64),
    'depths': hp.quniform('depths', low=2, high=6, q=1),
    #'dropout': hp.quniform('dropout', low=0.0, high=0.4, q=0.05),
    'NumOutLayers': hp.quniform('NumOutLayers', low=1, high=3, q=1)
}
INT_KEYS = ['dimension', 'depths', 'NumOutLayers']

def parse_input_arguments():
    parser = argparse.ArgumentParser(description='Water solubility prediction')
    parser.add_argument('--dataset', type=str, choices=['ws', 'logp', 'mp', 'xlogp3', 'qm9', 'solvation_exp', 'ccdc_sol', 'ccdc_logp', 'ccdc_sollogp'],
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
    parser.add_argument('--cv', action='store_true')
    parser.add_argument('--style', type=str)
    return parser.parse_args()

def grid_search(args: HyperoptArgs):

    # Run grid search
    results = []

    # Define hyperparameter optimization
    def objective(hyperparams: Dict[str, Union[int, float]]) -> float:
        # Convert hyperparams from float to int when necessary
        for key in INT_KEYS:
            hyperparams[key] = int(hyperparams[key])

        # Copy args
        hyper_args = deepcopy(args)

        # Update args with hyperparams
        for key, value in hyperparams.items():
            setattr(hyper_args, key, value) 

        # Record results
        model = get_model(args)
        model_ = model(args).to(device)
        num_params = param_count(model_)

        if args['adam']:
            optimizer = torch.optim.Adam(model_.parameters(), lr=0.001)
        if args['sgd']:
            optimizer = torch.optim.SGD(model_.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        if args['swa']:
            optimizer = torchcontrib.optim.SWA(optimizer)
        if args['lr_style'] == 'decay':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=5, min_lr=0.00001)
            ##########################################################################
        #logger.info(f'num params: {num_params:,}')
        #logger.info(f'{mean_score} +/- {std_score} {hyper_args.metric}')

        for epoch in range(1, args['epochs']):
            #lr = scheduler.optimizer.param_groups[0]['lr']
            loss = train(model_, optimizer, train_loader, this_dic)
            if args['swa']:
                    optimizer.update_swa()
            if args['dataset'] in ['mp', 'xlogp3']:
                #train_error = np.asscalar(loss.data.cpu().numpy()) # don't test the entire train set.
                train_error = loss
            else:
                train_error = test(model_, train_loader, this_dic)
                val_error = test(model_, val_loader, this_dic)

        results.append({
            'valRMSE': val_error,
            'hyperparams': hyperparams,
            'num_params': num_params
        })
    

    fmin(objective, SPACE, algo=tpe.suggest, max_evals=20)

    # Report best result
    
    best_result = min(results, key=lambda result: result['valRMSE'])
    
    if not os.path.exists(os.path.join(args['running_path'], args.dataset, 'HPSearch', args['model'])):
                    os.makedirs(os.path.join(args['running_path'], args.dataset, 'HPSearch', args['model']))
    with open(os.path.join(args['running_path'], args.dataset, 'HPSearch', args['model']), 'w') as f:
        json.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)


def main():
    args = parse_input_arguments()
    this_dic = var(args)

    loader = get_data_loader(this_dic)
    train_loader, val_loader, test_loader, std, num_features, num_bond_features, num_i_2 = \
            loader.train_loader, loader.val_loader, loader.test_loader, loader.std, \
                loader.num_features, loader.num_bond_features, loader.num_i_2
    this_dic['num_features'], this_dic['num_bond_features'], this_dic['num_i_2'], \
            this_dic['std'] = int(num_features), num_bond_features, num_i_2, std
    this_dic['epochs'] = 200

    if this_dic['model'] in ['loopybp', 'wlkernel']:
        this_dic['atom_fdim'] = num_features
        this_dic['bond_fdim'] = num_bond_features
        this_dic['atom_messages'] = False
        this_dic['outDim'] = this_dic['dimension'] 

    grid_search(this_dic)

if __name__ == '__main__':
    main()