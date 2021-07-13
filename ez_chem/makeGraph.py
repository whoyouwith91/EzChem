import os, sys, math, json, argparse, logging, time, random
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from featurization import *
from sklearn.model_selection import train_test_split
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from three_level_frag import cleavage, AtomListToSubMol, standize, mol2frag, WordNotFoundError, counter

def parse_input_arguments():
    parser = argparse.ArgumentParser(description='Physicochemical prediction')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--format', type=str, default='graphs')
    parser.add_argument('--model', type=str)
    parser.add_argument('--usePeriodics', action='store_true')
    parser.add_argument('--mol_features', action='store_true')
    parser.add_argument('--atom_classification', action='store_true')
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
    
    elif this_dic['dataset'] == 'solNMR': # multitask
        train_raw = torch.load(os.path.join(alldatapath, args.dataset, 'split', args.style, 'smaller_solNMR_train.pt'))
        valid_raw = torch.load(os.path.join(alldatapath, args.dataset, 'split', args.style, 'smaller_solNMR_valid.pt'))
        test_raw = torch.load(os.path.join(alldatapath, args.dataset, 'split', args.style, 'smaller_solNMR_test.pt'))
    else: 
        if this_dic['dataset'] not in ['sol_calc/ALL', 'solOct_calc/ALL', 'logp_calc/ALL', 'xlogp3']:
            train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'train.csv'), names=['SMILES', 'target'])
            valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'valid.csv'), names=['SMILES', 'target'])
            test_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'test.csv'), names=['SMILES', 'target'])
        if this_dic['dataset'] in ['sol_calc/ALL', 'solOct_calc/ALL', 'logp_calc/ALL', 'xlogp3']:
            train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'train.csv'), names=['SMILES', 'target'])
            valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'valid.csv'), names=['SMILES', 'target'])
            test_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'test.csv'), names=['SMILES', 'target'])
    
    if this_dic['atom_classification']: # for different datasets using different vocabulary
        efgs_vocabulary = torch.load('/scratch/dz1061/gcn/chemGraph/data/sol_calc/ALL/split/base/all_info/Frag20-EFGs.pt')
    
    if this_dic['format'] == 'graphs':
        examples = []
        if this_dic['dataset'] == 'solNMR':
            all_data = {**train_raw, **valid_raw, **test_raw}
        else:
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
                        molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                        molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                        molgraphs['y'] = torch.FloatTensor([tar])
                        molgraphs['id'] = torch.FloatTensor([idx])
                        examples.append(molgraphs)
                else: # calcSolLogP
                    for idx, smi, tar1, tar2, tar3 in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target1'], all_data['target2'], all_data['target3']):
                        molgraphs = {}

                        mol_graph = MolGraph(smi, args.usePeriodics, this_dic['model'])
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                        molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
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
                        molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
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

            if this_dic['dataset'] == 'solNMR':
                for d, value in all_data.items():
                    molgraphs = {}
                    mol = value[0][0]
                    mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])

                    molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                    molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    molgraphs['mol_y'] = torch.FloatTensor([value[0][1]])
                    molgraphs['atom_y'] = torch.FloatTensor(value[0][2])
                    molgraphs['N'] = torch.FloatTensor([mol.GetNumAtoms()])
                    examples.append(molgraphs)

                    if this_dic['atom_classification']:
                        try:
                            efg = mol2frag(mol, returnidx=True, vocabulary=list(efgs_vocabulary), toEnd=True, extra_included=True, TreatHs='include', isomericSmiles=False)
                            molgraphs['atom_efgs'] = getAtomToEFGs(efg, efgs_vocabulary)
                        except:
                            molgraphs['atom_efgs'] = None

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(all_data)))

            else:
                for idx, smi, tar in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target']):
                    molgraphs = {}
                    
                    mol_graph = MolGraph(smi, args.usePeriodics, this_dic['model'])
                    molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                    molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    if this_dic['dataset'] not in ['zinc', 'nmr']:
                        molgraphs['y'] = torch.FloatTensor([tar])
                    if this_dic['dataset'] in ['nmr']:
                        mask = np.zeros((molgraphs['x'].shape[0], 1), dtype=np.float32)
                        vals = np.zeros((molgraphs['x'].shape[0], 1), dtype=np.float32)
                        for k, v in tar[0].items():
                            mask[int(k), 0] = 1.0
                            vals[int(k), 0] = v
                        molgraphs['y'] = torch.FloatTensor(vals).flatten()
                        molgraphs['mask'] = torch.FloatTensor(mask).flatten()
                    if this_dic['dataset'] not in ['zinc', 'nmr']:
                        molgraphs['smiles'] = smi
                    if this_dic['mol_features']:
                        if this_dic['dataset'] == 'sol_calc/ALL':
                            #select_index_rdkit = [0, 36, 41, 46, 49, 50, 52, 58, 66, 78, 80, 81, 84, 95, 98, 103, 105, 112, 113]
                            # select_descriptors = ['TPSA', 'RingCount', 'NOCount', 'NumHDonors', 'VSA_EState8', 'SlogP_VSA2', 'NumAliphaticHeterocycles', \
                                                    #'MaxPartialCharge', 'VSA_EState9', 'PEOE_VSA8', 'BalabanJ', 'MolLogP', 'Kappa3', 'PEOE_VSA1', \
                                                    #'NHOHCount', 'SlogP_VSA5', 'VSA_EState10', 'SMR_VSA3', 'BCUT2D_MRHI', 'SMR_VSA1']
                            select_index_rdkit = range(200)
                            features = np.array(rdkit_2d_normalized_features_generator(smi))[select_index_rdkit]
                            molgraphs['mol_features'] = torch.FloatTensor(features)
                    if this_dic['atom_classification']:
                        efg = mol2frag(Chem.MolFromSmiles(smi), returnidx=True, vocabulary=list(efgs_vocabulary), toEnd=True, extra_included=True, TreatHs='include', isomericSmiles=False)
                        molgraphs['atom_efgs'] = getAtomToEFGs(efg, efgs_vocabulary)

                    molgraphs['id'] = torch.FloatTensor([idx])
                    examples.append(molgraphs)
                if this_dic['mol_features']:
                    new_examples = [d for d in examples if not np.isnan(np.sum(d['mol_features'].numpy()))]
                    examples = new_examples
   
                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(all_data.shape[0]))
    
    if this_dic['format'] == 'descriptors':
        descriptors = list(np.array(Descriptors._descList)[:,0])
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors)
        examples = []
        all_data = pd.concat([train_raw, valid_raw, test_raw])
        for smi,tar in zip(all_data['SMILES'], all_data['target']):
            mols = {}
            mol = Chem.MolFromSmiles(smi)
            #examples.append(calc(mol)[1:])
            mols['descriptors'] = calc.CalcDescriptors(mol)
            mols['y'] = tar
            if this_dic['mol_features']:
                if this_dic['dataset'] == 'sol_calc/ALL':
                    select_index_rdkit = [0, 36, 41, 46, 49, 50, 52, 58, 66, 78, 80, 81, 84, 95, 98, 103, 105, 112, 113]
                    # select_descriptors = ['TPSA', 'RingCount', 'NOCount', 'NumHDonors', 'VSA_EState8', 'SlogP_VSA2', 'NumAliphaticHeterocycles', \
                                            #'MaxPartialCharge', 'VSA_EState9', 'PEOE_VSA8', 'BalabanJ', 'MolLogP', 'Kappa3', 'PEOE_VSA1', \
                                            #'NHOHCount', 'SlogP_VSA5', 'VSA_EState10', 'SMR_VSA3', 'BCUT2D_MRHI', 'SMR_VSA1']
                    features = np.array(rdkit_2d_normalized_features_generator(smi))[select_index_rdkit]
                    molgraphs['mol_features'] = torch.FloatTensor(features)
            examples.append(mols)
        if this_dic['mol_features']:
            new_examples = [d for d in examples if not np.isnan(np.sum(d['mol_features'].numpy()))]
            examples = new_examples

        if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw')):
            os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw'))
        torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw', 'temp.pt')) ###
        print('Finishing processing {} compounds'.format(all_data.shape[0]))
        

if __name__ == '__main__':
    main()

