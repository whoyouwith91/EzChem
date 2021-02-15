import os, sys, math, json, argparse, logging, time, random
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from featurization import *
from sklearn.model_selection import train_test_split

def parse_input_arguments():
    parser = argparse.ArgumentParser(description='Physicochemical prediction')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--save_path', type=str, default='/scratch/dz1061/gcn/chemGraph/data/')
    parser.add_argument('--style', type=str, help='it is for base or different experiments')
    parser.add_argument('--solvent', type=str, default='')
    return parser.parse_args()


def main():
    args = parse_input_arguments()
    this_dic = vars(args)
    alldatapath = '/scratch/dz1061/gcn/chemGraph/data/'
    if args.style in ['CV']:
        this_dic['cv'] = True 
    else:
        this_dic['cv'] = False

    if this_dic['dataset'] == 'calcSolLogP':
        train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base', 'train.csv'), names=['SMILES', 'target1', 'target2', 'target3'])
        valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base', 'valid.csv'), names=['SMILES', 'target1', 'target2', 'target3'])
        if args.style == 'base/COMPLETE':
           train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base/COMPLETE', 'train.csv'), names=['SMILES', 'target1', 'target2', 'target3'])
           valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base/COMPLETE', 'valid.csv'), names=['SMILES', 'target1', 'target2', 'target3'])
        if args.style == 'hpsearch/base':
           train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'hpsearch/base', 'train.csv'), names=['SMILES', 'target1', 'target2', 'target3'])
           valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'hpsearch/base', 'valid.csv'), names=['SMILES', 'target1', 'target2', 'target3'])
    elif this_dic['dataset'] == 'commonProperties':
        train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base', 'train.csv'), names=['SMILES', 'target1', 'target2', 'target3', 'target4'])
    elif this_dic['style'] == 'external':
        if this_dic['dataset'] == 'deepchem/delaney':
            train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base', 'externalESOL.csv'.format(args.dataset)), names=['SMILES', 'target', 'InChI'])
        if this_dic['dataset'] == 'deepchem/logp':
            train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base', 'externalLipo.csv'.format(args.dataset)), names=['SMILES', 'target', 'InChI'])
        if this_dic['dataset'] == 'deepchem/freesol':
            train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base', 'externalFreesol.csv'.format(args.dataset)), names=['SMILES', 'target', 'InChI'])
    else:
        if this_dic['dataset'] not in ['sol_calc/ALL', 'solOct_calc/ALL', 'solOct_calc/ALL', 'logp_calc/ALL', 'xlogp3']:
           train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base', 'train.csv'), names=['SMILES', 'target'])
           valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base', 'valid.csv'), names=['SMILES', 'target'])
           test_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base', 'test.csv'), names=['SMILES', 'target'])
        if this_dic['dataset'] in ['sol_calc/ALL', 'solOct_calc/ALL', 'solOct_calc/ALL', 'logp_calc/ALL', 'xlogp3'] and args.style == 'base/COMPLETE':
           train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base/COMPLETE', 'train.csv'), names=['SMILES', 'target'])
           valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base/COMPLETE', 'valid.csv'), names=['SMILES', 'target'])
        if this_dic['dataset'] in ['sol_calc/ALL', 'solOct_calc/ALL', 'solOct_calc/ALL', 'logp_calc/ALL', 'xlogp3'] and args.style.startswith('hpsearch'):
           train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'hpsearch/base', 'train.csv'), names=['SMILES', 'target'])
           valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'hpsearch/base', 'valid.csv'), names=['SMILES', 'target'])
        if this_dic['dataset'] in ['sol_calc/ALL', 'solOct_calc/ALL', 'solOct_calc/ALL', 'logp_calc/ALL', 'xlogp3'] and args.style == 'base':
           train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base', 'train.csv'), names=['SMILES', 'target'])
           valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base', 'valid.csv'), names=['SMILES', 'target'])
           test_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'base', 'test.csv'), names=['SMILES', 'target'])
 
    if this_dic['model'] not in ['loopybp', 'wlkernel']:
        examples = []
        if args.solvent:
            examples_solvent = []
        if this_dic['dataset'] == 'commonProperties':
            all_data = train_raw
        elif this_dic['dataset'] in ['calcSolLogP', 'sol_calc/ALL', 'solOct_calc/ALL', 'logp_calc/ALL', 'xlogp3']:
            all_data = pd.concat([train_raw, valid_raw])
            if args.style == 'base':
                all_data = pd.concat([train_raw, valid_raw, test_raw])
        elif this_dic['style'] == 'external':
            all_data = train_raw
        else:
            all_data = pd.concat([train_raw, valid_raw, test_raw])
        if this_dic['cv']:
            for i in range(5):
                examples = []
                this_dic['seed'] = i
                rest_data = pd.concat([train_raw, valid_raw])
                if args.style == 'hpsearch':
                    train_raw, valid_raw = train_raw[:10000], valid_raw[:1000]
                    rest_data = pd.concat([train_raw, valid_raw])
                train_, valid_ = train_test_split(rest_data, test_size=valid_raw.shape[0], random_state=this_dic['seed'])
                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i))):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i)))
                train_.to_csv(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i), 'train.csv'), index=False, header=None)
                valid_.to_csv(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i), 'valid.csv'), index=False, header=None)

                if args.style == 'hpsearch':
                    all_data = pd.concat([train_, valid_])
                else:
                    all_data = pd.concat([train_, valid_, test_raw])
                if this_dic['dataset'] not in ['calcSolLogP', 'commonProperties']:
                    for idx, smi, tar in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target']):
                        molgraphs = {}
                    
                        mol_graph = MolGraph(smi, this_dic['model'])
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                        molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.real_f_bonds)
                        molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                        molgraphs['y'] = torch.FloatTensor([tar])
                        molgraphs['id'] = torch.FloatTensor([idx])
                        examples.append(molgraphs)
                else: # calcSolLogP
                    for idx, smi, tar1, tar2, tar3 in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target1'], all_data['target2'], all_data['target3']):
                        molgraphs = {}

                        mol_graph = MolGraph(smi, this_dic['model'])
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                        molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.real_f_bonds)
                        molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                        molgraphs['y0'] = torch.FloatTensor([tar1])
                        molgraphs['y1'] = torch.FloatTensor([tar2])
                        molgraphs['y2'] = torch.FloatTensor([tar3])
                        molgraphs['id'] = torch.FloatTensor([idx]) 
                        examples.append(molgraphs)

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'cv_'+str(i), 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'cv_'+str(i), 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'cv_'+str(i), 'raw', 'temp.pt')) ###

        elif this_dic['dataset'] == 'calcSolLogP':
            for idx, smi, tar1, tar2, tar3 in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target1'], all_data['target2'], all_data['target3']):
                    molgraphs = {}
                    
                    mol_graph = MolGraph(smi, this_dic['model'])
                    molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.real_f_bonds)
                    molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    molgraphs['y0'] = torch.FloatTensor([tar1])
                    molgraphs['y1'] = torch.FloatTensor([tar2])
                    molgraphs['y2'] = torch.FloatTensor([tar3])
                    molgraphs['id'] = torch.FloatTensor([idx])
                    

                    examples.append(molgraphs)
            if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'raw')):
                os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'raw'))
            torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'raw', 'temp.pt'))
        
        elif this_dic['dataset'] == 'commonProperties':
            for idx, smi, tar1, tar2, tar3, tar4 in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target1'], all_data['target2'], all_data['target3'], all_data['target4']):
                    molgraphs = {}

                    mol_graph = MolGraph(smi, this_dic['model'])
                    molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.real_f_bonds)
                    molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    molgraphs['y'] = torch.FloatTensor([tar1])
                    molgraphs['y1'] = torch.FloatTensor([tar2])
                    molgraphs['y2'] = torch.FloatTensor([tar3])
                    molgraphs['y3'] = torch.FloatTensor([tar4])
                    molgraphs['id'] = torch.FloatTensor([idx])
                    examples.append(molgraphs)
            if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'raw')):
                os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'raw'))
            torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'raw', 'temp.pt'))

        elif args.solvent == 'water': # for solvation interaction network
            for idx, smi, tar in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target']):
                    solute_graphs, solvent_graphs = {}, {}
                    
                    mol_graph = MolGraph(smi, this_dic['model'])
                    solute_graphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    solute_graphs['edge_attr'] = torch.FloatTensor(mol_graph.real_f_bonds)
                    solute_graphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    if this_dic['dataset'] not in ['zinc']:
                        solute_graphs['y'] = torch.FloatTensor([tar])
                    solute_graphs['id'] = torch.FloatTensor([idx])
                    examples.append(solute_graphs)

                    mol_graph = MolGraph('O', this_dic['model'])
                    solvent_graphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    solvent_graphs['edge_attr'] = None
                    solvent_graphs['edge_index'] = None
                    
                    examples_solvent.append(solvent_graphs)


            if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], args.solvent, 'raw')):
                os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], args.solvent, 'raw'))
            torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], args.solvent, 'raw', 'solute_temp.pt')) ###
            torch.save(examples_solvent, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], args.solvent, 'raw', 'solvent_temp.pt')) ###

            print('Finishing processing {} compounds'.format(all_data.shape[0]))
        
        elif args.solvent == 'octanol': # for solvation interaction network
            for idx, smi, tar in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target']):
                    solute_graphs, solvent_graphs = {}, {}
                    
                    mol_graph = MolGraph(smi, this_dic['model'])
                    solute_graphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    solute_graphs['edge_attr'] = torch.FloatTensor(mol_graph.real_f_bonds)
                    solute_graphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    if this_dic['dataset'] not in ['zinc']:
                        solute_graphs['y'] = torch.FloatTensor([tar])
                    solute_graphs['id'] = torch.FloatTensor([idx])
                    examples.append(solute_graphs)

                    mol_graph = MolGraph('CCCCCCCCO', this_dic['model'])
                    solvent_graphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    solvent_graphs['edge_attr'] = torch.FloatTensor(mol_graph.real_f_bonds)
                    solvent_graphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    
                    examples_solvent.append(solvent_graphs)


            if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], args.solvent, 'raw')):
                os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], args.solvent, 'raw'))
            torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], args.solvent, 'raw', 'solute_temp.pt')) ###
            torch.save(examples_solvent, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], args.solvent, 'raw', 'solvent_temp.pt')) ###

            print('Finishing processing {} compounds'.format(all_data.shape[0]))

        elif args.solvent == 'watOct': # for solvation interaction network
            for idx, smi, tar in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target']):
                    solute_graphs, solvent_graphs = {}, {}
                    
                    mol_graph = MolGraph(smi, this_dic['model'])
                    solute_graphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    solute_graphs['edge_attr'] = torch.FloatTensor(mol_graph.real_f_bonds)
                    solute_graphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    if this_dic['dataset'] not in ['zinc']:
                        solute_graphs['y'] = torch.FloatTensor([tar])
                    solute_graphs['id'] = torch.FloatTensor([idx])
                    examples.append(solute_graphs)

                    wat_mol_graph = MolGraph('O', this_dic['model'])
                    oct_mol_graph = MolGraph('CCCCCCCCO', this_dic['model'])
                    solvent_graphs['wat_x'] = torch.FloatTensor(wat_mol_graph.f_atoms)
                    solvent_graphs['wat_edge_attr'] = None
                    solvent_graphs['wat_edge_index'] = None
                    solvent_graphs['oct_x'] = torch.FloatTensor(oct_mol_graph.f_atoms)
                    solvent_graphs['oct_edge_attr'] = torch.FloatTensor(oct_mol_graph.real_f_bonds)
                    solvent_graphs['oct_edge_index'] = torch.LongTensor(np.concatenate([oct_mol_graph.at_begin, oct_mol_graph.at_end]).reshape(2,-1))
                    
                    examples_solvent.append(solvent_graphs)


            if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], args.solvent, 'raw')):
                os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], args.solvent, 'raw'))
            torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], args.solvent, 'raw', 'solute_temp.pt')) ###
            torch.save(examples_solvent, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], args.solvent, 'raw', 'solvent_temp.pt')) ###

            print('Finishing processing {} compounds'.format(all_data.shape[0]))

        else:
            
            for idx, smi, tar in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target']):
                    molgraphs = {}
                    
                    mol_graph = MolGraph(smi, this_dic['model'])
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
            if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'raw')):
                os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'raw'))
            torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'raw', 'temp.pt')) ###
            print('Finishing processing {} compounds'.format(all_data.shape[0]))
    
    else:# for Loopybp, WlKernel
        examples = []
        if this_dic['dataset'] == 'commonProperties':
            all_data = train_raw
        elif this_dic['dataset'] in ['calcSolLogP', 'sol_calc/ALL', 'logp_calc/ALL', 'xlogp3']:
            all_data = pd.concat([train_raw, valid_raw])
        elif this_dic['style'] == 'external':
            all_data = train_raw
        else:
            all_data = pd.concat([train_raw, valid_raw, test_raw])
    
        #name = ['train', 'valid', 'test']
        #dataSize = [train_.shape[0], valid_raw.shape[0], test_raw.shape[0]]
        if this_dic['cv']:
            for i in range(5):
                this_dic['seed'] = i
                rest_data = pd.concat([train_raw, valid_raw])
                if args.style == 'hpsearch':
                    train_raw, valid_raw = train_raw[:10000], valid_raw[:1000]
                    rest_data = pd.concat([train_raw, valid_raw])
                train_, valid_ = train_test_split(rest_data, test_size=valid_raw.shape[0], random_state=this_dic['seed'])

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i))):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i)))
                train_.to_csv(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i), 'train.csv'), index=False, header=None)
                valid_.to_csv(os.path.join(this_dic['save_path'], args.dataset, 'split', args.style, this_dic['model'], 'cv_'+str(i), 'valid.csv'), index=False, header=None)
                
                if args.style == 'hpsearch':
                    all_data = pd.concat([train_, valid_])
                    name = ['train', 'valid']
                    dataSize = [train_.shape[0], valid_.shape[0]]
                else:
                    all_data = pd.concat([train_, valid_, test_raw])
                    name = ['train', 'valid', 'test']
                    dataSize = [train_.shape[0], valid_.shape[0], test_raw.shape[0]]

                if this_dic['dataset'] not in ['calcSolLogP', 'commonProperties']:
                    for idx, smi, tar in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target']):
                            molgraphs = {}
        
                            mol_graph = MolGraph(smi, this_dic['model'])
                            molgraphs['atom_features'] = mol_graph.f_atoms
                            molgraphs['bond_features'] = mol_graph.f_bonds
                            molgraphs['a2b'] = mol_graph.a2b
                            molgraphs['b2a'] = mol_graph.b2a
                            molgraphs['b2revb'] = mol_graph.b2revb
                            molgraphs['n_bonds'] = mol_graph.n_bonds
                            molgraphs['n_atoms'] = mol_graph.n_atoms
                            molgraphs['y'] = [tar]
                            molgraphs['smiles'] = smi
                            molgraphs['id'] = torch.FloatTensor([idx])
        
                            examples.append(molgraphs)
                else:
                    for idx, smi, tar1, tar2 in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target1'], all_data['target2']):
                            molgraphs = {}
                            mol_graph = MolGraph(smi, this_dic['model'])
                            molgraphs['atom_features'] = mol_graph.f_atoms
                            molgraphs['bond_features'] = mol_graph.f_bonds
                            molgraphs['a2b'] = mol_graph.a2b
                            molgraphs['b2a'] = mol_graph.b2a
                            molgraphs['b2revb'] = mol_graph.b2revb
                            molgraphs['n_bonds'] = mol_graph.n_bonds
                            molgraphs['n_atoms'] = mol_graph.n_atoms
                            molgraphs['y1'] = torch.FloatTensor([tar1])
                            molgraphs['y2'] = torch.FloatTensor([tar2])
                            molgraphs['smiles'] = smi
                            molgraphs['id'] = torch.FloatTensor([idx])
        
                            examples.append(molgraphs)

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'cv_'+str(i))):
                   os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'cv_'+str(i)))
                for name_, size in zip(name, dataSize):
                    torch.save(examples[:size], os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'cv_'+str(i), name_+'.pt')) ###
                    examples = examples[size:]

        elif this_dic['dataset'] == 'calcSolLogP':
                for idx, smi, tar1, tar2, tar3 in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target1'], all_data['target2'], all_data['target3']):
                        molgraphs = {}

                        mol_graph = MolGraph(smi, this_dic['model'])
                        molgraphs['atom_features'] = mol_graph.f_atoms
                        molgraphs['bond_features'] = mol_graph.f_bonds
                        molgraphs['a2b'] = mol_graph.a2b
                        molgraphs['b2a'] = mol_graph.b2a
                        molgraphs['b2revb'] = mol_graph.b2revb
                        molgraphs['n_bonds'] = mol_graph.n_bonds
                        molgraphs['n_atoms'] = mol_graph.n_atoms
                        molgraphs['y0'] = torch.FloatTensor([tar1])
                        molgraphs['y1'] = torch.FloatTensor([tar2])
                        molgraphs['y2'] = torch.FloatTensor([tar3])
                        molgraphs['smiles'] = smi
                        molgraphs['id'] = torch.FloatTensor([idx])
        
                        examples.append(molgraphs)
                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'])):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model']))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], name+'.pt')) ###

        elif False and this_dic['dataset'] == 'commonProperties' and this_dic['taskType'] in ['multi', 'preTraining']:
                for idx, smi, tar1, tar2, tar3, tar4 in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target1'], all_data['target2'], all_data['target3'], all_data['target4']):
                        molgraphs = {}
                        mol_graph = MolGraph(smi, this_dic['model'])
                        molgraphs['atom_features'] = mol_graph.f_atoms
                        molgraphs['bond_features'] = mol_graph.f_bonds
                        molgraphs['a2b'] = mol_graph.a2b
                        molgraphs['b2a'] = mol_graph.b2a
                        molgraphs['b2revb'] = mol_graph.b2revb
                        molgraphs['n_bonds'] = mol_graph.n_bonds
                        molgraphs['n_atoms'] = mol_graph.n_atoms
                        molgraphs['y1'] = [tar1]
                        molgraphs['y2'] = [tar2]
                        molgraphs['y3'] = [tar3]
                        molgraphs['y4'] = [tar4]
                        molgraphs['smiles'] = smi
                        molgraphs['id'] = torch.FloatTensor([idx])
                        examples.append(molgraphs)

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'raw')): # todo 
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'raw')) # todo
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], 'raw', 'temp.pt')) # todo

        else:
                name = ['train', 'valid', 'test']
                dataSize = [train_raw.shape[0], valid_raw.shape[0], test_raw.shape[0]]
                for idx, smi, tar in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target']):
                        molgraphs = {}
        
                        mol_graph = MolGraph(smi)
                        molgraphs['atom_features'] = mol_graph.f_atoms
                        molgraphs['bond_features'] = mol_graph.f_bonds
                        molgraphs['a2b'] = mol_graph.a2b
                        molgraphs['b2a'] = mol_graph.b2a
                        molgraphs['b2revb'] = mol_graph.b2revb
                        molgraphs['n_bonds'] = mol_graph.n_bonds
                        molgraphs['n_atoms'] = mol_graph.n_atoms
                        molgraphs['atomBegin'] = mol_graph.at_begin
                        molgraphs['atomEnd'] = mol_graph.at_end
                        molgraphs['y'] = [tar]
                        molgraphs['smiles'] = smi
                        molgraphs['id'] = torch.FloatTensor([idx])
        
                        examples.append(molgraphs)
                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'])):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model']))
                for name_, size in zip(name, dataSize):
                    torch.save(examples[:size], os.path.join(this_dic['save_path'], args.dataset, 'graphs', args.style, this_dic['model'], name_+'.pt')) ###
                    examples = examples[size:]


if __name__ == '__main__':
    main()

