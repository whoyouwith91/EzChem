import argparse

def get_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of graph neural networks')
    parser.add_argument('--allDataPath', type=str, default='/scratch/dz1061/gcn/chemGraph/data')
    parser.add_argument('--running_path', type=str, help='path to save model', default='/scratch/dz1061/gcn/chemGraph/results')    
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--scheduler', type=str, default='const')
    parser.add_argument('--warmup_epochs', type=int, default=2) # for NoamLR
    parser.add_argument('--init_lr', type=float, default=0.0001) # for NoamLR
    parser.add_argument('--max_lr', type=float, default=0.001) # for NoamLR
    parser.add_argument('--final_lr', type=float, default=0.0001) # for NoamLR
    parser.add_argument('--patience_epochs', type=int, default=2) # for NoamLR
    parser.add_argument('--decay_factor', type=float, default=0.9)
    parser.add_argument('--num_layer', type=int, default=3, help='number of GNN message passing layers (default: 3).')
    parser.add_argument('--emb_dim', type=int, default=64, help='embedding dimensions (default: 64)')
    parser.add_argument('--fully_connected_layer_sizes', type=int, nargs='+') # number of readout layers
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--explicit_split', action='store_true') # TODO
    parser.add_argument('--sample', action='store_true') # 
    parser.add_argument('--data_seed', type=int, default=0)
    parser.add_argument('--sample_size', type=int, default=100) # 
    parser.add_argument('--drop_ratio', type=float, default=0.0) 
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--model', type=str, default="1-GNN")
    parser.add_argument('--ACSF', action='store_true')
    parser.add_argument('--2D', action='store_true')
    parser.add_argument('--DMPNN', action='store_true')
    parser.add_argument('--mol_features', action='store_true')
    parser.add_argument('--residual_connect', action='store_true')
    parser.add_argument('--interaction_simpler', action='store_true')
    parser.add_argument('--pooling', type=str, default='sum')
    parser.add_argument('--aggregate', type=str, default='add')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--JK', type=str, default="last", help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--action', type=str) # physnet
    parser.add_argument('--mask', action='store_true') # for nmr type; 
    parser.add_argument('--train_type', type=str, default='from_scratch', choices=['from_scratch', 'transfer', 'hpsearch', 'finetuning'])
    parser.add_argument('--preTrainedPath', type=str)
    parser.add_argument('--propertyLevel', type=str, default='molecule')
    parser.add_argument('--test_level', type=str, default='molecule')
    parser.add_argument('--loss', type=str, choices=['l1', 'l2', 'smooth_l1', 'dropout', 'vae', 'unsuper', 'maskedL1', 'maskedL2', 'class'])
    parser.add_argument('--metrics', type=str, choices=['l1', 'l2', 'class'])
    parser.add_argument('--weights', type=str, choices=['he_norm', 'xavier_norm', 'he_uni', 'xavier_uni'], default='he_uni')
    parser.add_argument('--clip', action='store_true') # clip weights or not
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--act_fn', type=str, default='relu')
    parser.add_argument('--optimizer',  type=str, choices=['adam', 'sgd', 'swa', 'EMA', 'adamW', 'SWA'])
    parser.add_argument('--style', type=str, default='base')  # if running CV
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--experiment', type=str)  # when doing experimenting, name it. 
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--gradCam', action='store_true')
    parser.add_argument('--tsne', action='store_true')
    parser.add_argument('--uncertainty',  action='store_true')
    parser.add_argument('--uncertaintyMode',  type=str)
    parser.add_argument('--weight_regularizer', type=float, default=1e-6)
    parser.add_argument('--dropout_regularizer', type=float, default=1e-5)
    parser.add_argument('--swag_start', type=int)
    
    return parser.parse_args()