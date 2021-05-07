import os, sys, math, json, argparse, logging, time, random
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from featurization import *
from sklearn.model_selection import train_test_split
from mordred import Calculator, descriptors
from rdkit import Chem

def parse_input_arguments():
    parser = argparse.ArgumentParser(description='Physicochemical prediction')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--format', type=str, default='graphs')
    parser.add_argument('--model', type=str)
    parser.add_argument('--usePeriodics', action=store_true)
    parser.add_argument('--save_path', type=str, default='/scratch/dz1061/gcn/chemGraph/data/')
    parser.add_argument('--style', type=str, help='it is for base or different experiments')
    return parser.parse_args()

def main():
    args = parse_input_arguments()
    this_dic = vars(args)
    alldatapath = '/scratch/dz1061/gcn/chemGraph/data/'
    
    if args.style in ['CV']:
        this_dic['cv'] = True 
    else:
        this_dic['cv'] = False

    if this_dic['dataset'] == 'calcSolLogP/ALL': # multitask
        train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'train.csv'), names=['SMILES', 'target1', 'target2', 'target3'])
        valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'valid.csv'), names=['SMILES', 'target1', 'target2', 'target3'])
        test_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'test.csv'), names=['SMILES', 'target1', 'target2', 'target3'])
    else: # single task
        if this_dic['dataset'] not in ['sol_calc/ALL', 'solOct_calc/ALL', 'logp_calc/ALL', 'xlogp3']:
           train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'train.csv'), names=['SMILES', 'target'])
           valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'valid.csv'), names=['SMILES', 'target'])
           test_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'test.csv'), names=['SMILES', 'target'])
        if this_dic['dataset'] in ['sol_calc/ALL', 'solOct_calc/ALL', 'logp_calc/ALL', 'xlogp3']:
            train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'train.csv'), names=['SMILES', 'target'])
            valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'valid.csv'), names=['SMILES', 'target'])
            test_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'test.csv'), names=['SMILES', 'target'])
 
    if this_dic['format'] == 'graphs':
        examples = []
        all_data = pd.concat([train_raw, valid_raw, test_raw])
        
        if this_dic['cv']: # 5-fold CV
            for i in range(5):
                examples = []
                this_dic['seed'] = i
                rest_data = pd.concat([train_raw, valid_raw])
                train_, valid_ = train_test_split(rest_data, test_size=valid_raw.shape[0], random_state=this_dic['seed'])
                
                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i))):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i)))
                train_.to_csv(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i), 'train.csv'), index=False, header=None)
                valid_.to_csv(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i), 'valid.csv'), index=False, header=None)

                if this_dic['dataset'] not in ['calcSolLogP/ALL']:
                    for idx, smi, tar in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target']):
                        molgraphs = {}
                    
                        mol_graph = MolGraph(smi, args.usePeriodics, this_dic['model'])
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms) 
                        molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.real_f_bonds)
                        molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                        molgraphs['y'] = torch.FloatTensor([tar])
                        molgraphs['id'] = torch.FloatTensor([idx])
                        examples.append(molgraphs)
                else: # calcSolLogP
                    for idx, smi, tar1, tar2, tar3 in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target1'], all_data['target2'], all_data['target3']):
                        molgraphs = {}

                        mol_graph = MolGraph(smi, args.usePeriodics, this_dic['model'])
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                        molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.real_f_bonds)
                        molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                        molgraphs['y0'] = torch.FloatTensor([tar1])
                        molgraphs['y1'] = torch.FloatTensor([tar2])
                        molgraphs['y2'] = torch.FloatTensor([tar3])
                        molgraphs['id'] = torch.FloatTensor([idx]) 
                        examples.append(molgraphs)

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'cv_'+str(i), 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'cv_'+str(i), 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'cv_'+str(i), 'raw', 'temp.pt')) ###

        else:
            if this_dic['dataset'] == 'calcSolLogP/ALL':
                for idx, smi, tar1, tar2, tar3 in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target1'], all_data['target2'], all_data['target3']):
                        molgraphs = {}
                        
                        mol_graph = MolGraph(smi, args.usePeriodics, this_dic['model'])
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                        molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.real_f_bonds)
                        molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                        molgraphs['y0'] = torch.FloatTensor([tar1])
                        molgraphs['y1'] = torch.FloatTensor([tar2])
                        molgraphs['y2'] = torch.FloatTensor([tar3])
                        molgraphs['id'] = torch.FloatTensor([idx])

                        examples.append(molgraphs)
                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw', 'temp.pt'))
                print('Finishing processing {} compounds'.format(all_data.shape[0]))

            else:
                for idx, smi, tar in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target']):
                    molgraphs = {}
                    
                    mol_graph = MolGraph(smi, args.usePeriodics, this_dic['model'])
                    molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.real_f_bonds)
                    molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    if this_dic['dataset'] not in ['zinc', 'nmr']:
                        molgraphs['y'] = torch.FloatTensor([tar])
                    if this_dic['dataset'] in ['nmr']:
                        mask = np.zeros((molgraphs['x'].shape[0], 1), dtype=np.float32)
                        vals = np.zeros((molgraphs['x'].shape[0], 1), dtype=np.float32)
                        for k, v in tar[0].items():
                            mask[int(k), 0] = 1.0
                            vals[int(k), 0] = v
                    if this_dic['dataset'] not in ['zinc', 'nmr']:
                        molgraphs['smiles'] = smi
                    molgraphs['id'] = torch.FloatTensor([idx])
                    examples.append(molgraphs)
                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(all_data.shape[0]))
    
    if this_dic['format'] == 'descriptors':
        calc = Calculator(descriptors, ignore_3D=True)
        examples = []
        all_data = pd.concat([train_raw, valid_raw, test_raw])
        for smi in all_data['SMILES']:
            mol = Chem.MolFromSmiles(smi)
            examples.append(calc(mol)[1:])
        if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw')):
            os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw'))
        torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw', 'temp.pt')) ###
        print('Finishing processing {} compounds'.format(all_data.shape[0]))
        

if __name__ == '__main__':
    main()

