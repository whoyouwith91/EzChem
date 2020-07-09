import os, sys, math, json, argparse, logging, time, random
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from featurization import *
from sklearn.model_selection import train_test_split

def parse_input_arguments():
    parser = argparse.ArgumentParser(description='Physicochemical prediction')
    parser.add_argument('--dataset', type=str, choices=['ws', 'logp', 'mp', 'xlogp3', 'qm9', 'sol_exp', 'ccdc_sol', 'ccdc_logp', 'ccdc_sollogp', 'mp_less', 'mp_drugs'])
    parser.add_argument('--model', type=str, choices=['1-2-GNN', 'ConvGRU', '1-GNN', 'SAGE', 'GCN', 'CONVATT', 'adaConv', 'ConvSet2Set', 'NNConvGAT', 'gradcam', 'dropout', 'loopybp', 'wlkernel'])
    parser.add_argument('--save_path', type=str, default='/beegfs/dz1061/gcn/chemGraph/data/')
    parser.add_argument('--style', type=str, help='it is for base or different experiments')
    return parser.parse_args()


def main():
    args = parse_input_arguments()
    this_dic = vars(args)
    alldatapath = '/beegfs/dz1061/gcn/chemGraph/data/'
    if args.style == 'CV':
        this_dic['cv'] = True 
    else:
        this_dic['cv'] = False
    train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base', 'train.csv'), names=['SMILES', 'target'])
    valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base', 'valid.csv'), names=['SMILES', 'target'])
    test_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base', 'test.csv'), names=['SMILES', 'target'])
    
    if this_dic['model'] not in ['loopybp', 'wlkernel']:
        examples = []
        all_data = pd.concat([train_raw, valid_raw, test_raw])
        if this_dic['cv']:
            for i in range(5):
                examples = []
                this_dic['seed'] = i
                rest_data = pd.concat([train_raw, valid_raw])
                train_, valid_ = train_test_split(rest_data, test_size=valid_raw.shape[0], random_state=this_dic['seed'])
                
                #if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i))):
                #    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i)))
                #train_.to_csv(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i), 'train.csv'), index=False, header=None)
                #valid_.to_csv(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i), 'valid.csv'), index=False, header=None)
                all_data = pd.concat([train_, valid_, test_raw])

                for idx, smi, tar in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target']):
                    molgraphs = {}
                    
                    mol_graph = MolGraph(smi)
                    molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.real_f_bonds)
                    molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    molgraphs['y'] = torch.FloatTensor([tar])
                    molgraphs['id'] = torch.FloatTensor([idx])
                    

                    examples.append(molgraphs)
                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'cv_'+str(i), 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'cv_'+str(i), 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'cv_'+str(i), 'raw', 'temp.pt')) ###

        else:
            for idx, smi, tar in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target']):
                    molgraphs = {}
                    
                    mol_graph = MolGraph(smi)
                    molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.real_f_bonds)
                    molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    molgraphs['y'] = torch.FloatTensor([tar])
                    molgraphs['id'] = torch.FloatTensor([idx])
                    

                    examples.append(molgraphs)
            if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'raw')):
                os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'raw'))
            torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'raw', 'temp.pt')) ###
    
    else:
        if this_dic['cv']:
            for i in range(5):
                this_dic['seed'] = i
                all_data = pd.concat([train_raw, valid_raw])
                train_, valid_ = train_test_split(all_data, test_size=test_raw.shape[0], random_state=this_dic['seed'])

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i))):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i)))
                train_.to_csv(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i), 'train.csv'), index=False, header=None)
                valid_.to_csv(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i), 'valid.csv'), index=False, header=None)

                for df, name in zip([train_, valid_, test_raw], ['train', 'valid', 'test']):
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
                    if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'cv_'+str(i))):
                        os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'cv_'+str(i)))
                    torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'cv_'+str(i), name+'.pt')) ###
        else:
            for df, name in zip([train_raw, valid_raw, test_raw], ['train', 'valid', 'test']):
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
                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'])):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model']))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], name+'.pt')) ###


if __name__ == '__main__':
    main()
