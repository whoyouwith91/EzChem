import os, sys, math, json, argparse, logging, time, random, glob
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from featurization import *
from sklearn.model_selection import train_test_split

def parse_input_arguments():
    parser = argparse.ArgumentParser(description='Physicochemical prediction')
    parser.add_argument('--dataset', type=str, choices=['ws', 'logp', 'mp', 'xlogp3', 'qm9', 'solvation_exp', 'ccdc_sol', 'ccdc_logp', 'ccdc_sollogp', 'sol_calc', 'logp_calc'])
    parser.add_argument('--model', type=str, choices=['1-2-GNN', 'ConvGRU', '1-GNN', 'SAGE', 'GCN', 'CONVATT', 'adaConv', 'ConvSet2Set', 'NNConvGAT', 'gradcam', 'dropout', 'loopybp', 'wlkernel'])
    parser.add_argument('--save_path', type=str, default='/beegfs/dz1061/gcn/chemGraph/data/')
    parser.add_argument('--style', type=str, help='it is for base or different experiments')
    return parser.parse_args()


def main():
    args = parse_input_arguments()
    this_dic = vars(args)
    this_dic['data_path'] = os.path.join('/beegfs/dz1061/gcn/chemGraph/data/', args.dataset, args.style)
    
    valid_raw = pd.read_csv(os.path.join(this_dic['data_path'], 'valid.csv'), names=['SMILES', 'target'])
    test_raw = pd.read_csv(os.path.join(this_dic['data_path'], 'test.csv'), names=['SMILES', 'target'])
    
    for file_ in glob.iglob(os.path.join(this_dic['data_path'], 'train*')):
        shard_id = file_.split('.')[-2].split('_')[-1]
        shards_df = pd.read_csv(file_, names=['SMILES', 'target'])

        if this_dic['model'] not in ['loopybp', 'wlkernel']:
            all_data = pd.concat([shards_df, valid_raw, test_raw])
            examples = []
            for idx, smi, tar in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target']):
                molgraphs = {}
                        
                mol_graph = MolGraph(smi)
                molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.real_f_bonds)
                molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                molgraphs['y'] = torch.FloatTensor([tar])
                molgraphs['id'] = torch.FloatTensor([idx])
                        

                examples.append(molgraphs)
            if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'shard_'+str(shard_id), 'raw')):
                os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'shard_'+str(shard_id), 'raw'))
            torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'shard_'+str(shard_id), 'raw', 'temp.pt')) ###
    
        else:
            examples = []
            for smi, target in zip(shards_df['SMILES'], shards_df['target']):
                molgraphs = {}
        
                mol_graph = MolGraph(smi)
                molgraphs['atom_features'] = mol_graph.f_atoms
                molgraphs['bond_features'] = mol_graph.f_bonds
                molgraphs['a2b'] = mol_graph.a2b
                molgraphs['b2a'] = mol_graph.b2a
                molgraphs['b2revb'] = mol_graph.b2revb
                molgraphs['n_bonds'] = mol_graph.n_bonds
                molgraphs['n_atoms'] = mol_graph.n_atoms
                molgraphs['y'] = [target]
                molgraphs['smiles'] = smi
        
                examples.append(molgraphs)
            if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'shard_'+str(shard_id))):
                os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'shard_'+str(shard_id)))
            torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'shard_'+str(shard_id), 'train.pt')) ###

        if this_dic['model'] in ['loopybp', 'wlkernel']:
            for df, name in zip([valid_raw, test_raw], ['valid', 'test']):
                examples = []
                for smi, target in zip(df['SMILES'], df['target']):
                    molgraphs = {}

                    mol_graph = MolGraph(smi)
                    molgraphs['atom_features'] = mol_graph.f_atoms
                    molgraphs['bond_features'] = mol_graph.f_bonds
                    molgraphs['a2b'] = mol_graph.a2b
                    molgraphs['b2a'] = mol_graph.b2a
                    molgraphs['b2revb'] = mol_graph.b2revb
                    molgraphs['n_bonds'] = mol_graph.n_bonds
                    molgraphs['n_atoms'] = mol_graph.n_atoms
                    molgraphs['y'] = [target]
                    molgraphs['smiles'] = smi

                    examples.append(molgraphs)
                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'shard_'+str(shard_id))):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'shard_'+str(shard_id)))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'shard_'+str(shard_id), name+'.pt'))
            
            

if __name__ == '__main__':
    main()
