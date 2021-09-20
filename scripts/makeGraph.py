import os, sys, math, json, argparse, logging, time, random, pickle
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from featurization import *
from sklearn.model_selection import train_test_split
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdmolfiles import SDMolSupplier
from rdkit.ML.Descriptors import MoleculeDescriptors
from three_level_frag import cleavage, AtomListToSubMol, standize, mol2frag, WordNotFoundError, counter
from torch_geometric.utils.convert import to_networkx, from_networkx
from deepchem.feat import BPSymmetryFunctionInput, CoulombMatrix, CoulombMatrixEig
from deepchem.utils import conformers
from dscribe.descriptors import ACSF
from rdkit.Chem.rdmolfiles import MolToXYZFile
from ase.io import read as ase_read
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.ML.Cluster import Butina

def gen_conformers(mol, numConfs=100, maxAttempts=1000, pruneRmsThresh=0.1, useExpTorsionAnglePrefs=True, useBasicKnowledge=True, enforceChirality=True):
	ids = AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, maxAttempts=maxAttempts, pruneRmsThresh=pruneRmsThresh, useExpTorsionAnglePrefs=useExpTorsionAnglePrefs, useBasicKnowledge=useBasicKnowledge, enforceChirality=enforceChirality, numThreads=0, randomSeed=1)
	return list(ids)

def parse_input_arguments():
    parser = argparse.ArgumentParser(description='Physicochemical prediction')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--file', type=str, default=None) # for file name such as pubchem, zinc, etc in Frag20
    parser.add_argument('--format', type=str, default='graphs')
    parser.add_argument('--model', type=str)
    parser.add_argument('--usePeriodics', action='store_true')
    parser.add_argument('--mol_features', action='store_true')
    parser.add_argument('--atom_classification', action='store_true')
    parser.add_argument('--BPS', action='store_true')
    parser.add_argument('--ACSF', action='store_true')
    parser.add_argument('--physnet', action='store_true')
    parser.add_argument('--dmpnn', action='store_true')
    parser.add_argument('--xyz', type=str)
    parser.add_argument('--save_path', type=str, default='/scratch/dz1061/gcn/chemGraph/data/')
    parser.add_argument('--style', type=str, help='it is for base or different experiments')
    parser.add_argument('--task', type=str)
    parser.add_argument('--train_type', type=str)
    return parser.parse_args()

def getMol(file, id_):
    if file in ['pubchem', 'zinc']:
        path_to_sdf = '/ext3/Frag20/lessthan10/sdf/' + file # path to the singularity file overlay-50G-10M.ext3
        sdf_file = os.path.join(path_to_sdf, str(id_)+'.sdf')
    elif file in ['CCDC']:
        path_to_sdf = '/ext3/Frag20/{}/sdf'.format(file) # path to the singularity file overlay-50G-10M.ext3
        sdf_file = os.path.join(path_to_sdf, str(id_)+'_min.sdf')
    else:
        path_to_sdf = '/ext3/Frag20/{}/sdf'.format(file)
        sdf_file = os.path.join(path_to_sdf, str(id_)+'.sdf')
    #print(sdf_file)
    suppl = SDMolSupplier(sdf_file, removeHs=False)
    return suppl[0]


def main():
    args = parse_input_arguments()
    this_dic = vars(args)
    alldatapath = '/scratch/dz1061/gcn/chemGraph/data/'
    
    if args.style in ['CV']:
        this_dic['cv'] = True 
    else:
        this_dic['cv'] = False

    if 'sol_calc/ALL' in this_dic['dataset'] or 'logp_calc/ALL' in this_dic['dataset']: # multitask
        train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'train.csv'))
        valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'valid.csv'))
        test_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'test.csv'))
        #sol_calc = pd.read_csv('/scratch/dz1061/gcn/chemGraph/data/sol_calc/ALL/split/base/all_info/all_processed_new.csv')
    elif this_dic['dataset'] in ['solNMR', 'solALogP']: # multitask
        train_raw = torch.load(os.path.join(alldatapath, args.dataset, 'split', args.style, 'smaller_solNMR_train.pt'))
        valid_raw = torch.load(os.path.join(alldatapath, args.dataset, 'split', args.style, 'smaller_solNMR_valid.pt'))
        test_raw = torch.load(os.path.join(alldatapath, args.dataset, 'split', args.style, 'smaller_solNMR_test.pt'))
    
    elif this_dic['dataset'] in ['nmr/carbon', 'nmr/hydrogen']: # multitask
        train_raw = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/{}/split/base/train.pickle'.format(args.dataset), 'rb'))
        test_raw = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/{}/split/base/test.pickle'.format(args.dataset), 'rb'))
    
    else: 
        train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'train.csv'), names=['SMILES', 'target'])
        valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'valid.csv'), names=['SMILES', 'target'])
        test_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', args.style, 'test.csv'), names=['SMILES', 'target'])
    
    if this_dic['atom_classification']: # for different datasets using different vocabulary
        efgs_vocabulary = torch.load('/scratch/dz1061/gcn/chemGraph/data/sol_calc/ALL/split/base/all_info/Frag20-EFGs.pt')
    
    if this_dic['format'] == 'graphs':
        examples = [] if not args.physnet else {}
        if this_dic['dataset'] in ['solNMR', 'solALogP']:
            all_data = {**train_raw, **valid_raw, **test_raw}
        elif this_dic['dataset'] in ['nmr/carbon', 'nmr/hydrogen']:
            all_data = pd.concat([train_raw, test_raw])
        elif this_dic['dataset'] == 'qm9/u0':
            suppl = torch.load('/scratch/dz1061/gcn/chemGraph/data/qm9/raw/molObjects.pt')
            u0 = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/qm9/raw/U0_rev.pickle', 'rb'))
            nmr = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/qm9/raw/qm9_NMR.pickle', 'rb'))
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
            if 'sol_calc/ALL' in this_dic['dataset']:
                if this_dic['ACSF']:
                    acsf = ACSF(
                                species=['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S'],
                                rcut=10.0,
                                g2_params=[[1, 1], [1, 2], [1, 3]],
                                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)
                for smi, value, id_, file in zip(all_data['QM_SMILES'], all_data['CalcSol'], all_data['ID'], all_data['SourceFile']):
                    #print(smi, id_, file)
                    if not this_dic['physnet']:
                        molgraphs = {}
                    mol = Chem.MolFromSmiles(smi)
                    mol_smi = Chem.AddHs(mol)
                    #mol = getMol(file, id_)

                    if not this_dic['physnet']:
                        mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)

                    if this_dic['ACSF'] and not this_dic['physnet']:
                        if file in ['pubchem', 'zinc']:
                            path_to_xyz = '/ext3/Frag20/lessthan10/xyz' # path to the singularity file overlay-50G-10M.ext3
                        else:
                            path_to_xyz = '/ext3/Frag20/{}/xyz'.format(file)
                        file_id = file +'_' + str(int(id_)) # such as 'pubchem_100001'
                        #if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(file_id))):
                        #    MolToXYZFile(mol, os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(file_id)))
                        atoms = ase_read(os.path.join(path_to_xyz, '{}.xyz'.format(file_id))) # path to the singularity file overlay-50G-10M.ext3
                        molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms())))
                        assert mol.GetNumAtoms() == molgraphs['x'].shape[0]

                    if args.dmpnn:
                        mol_graph = MolGraph_dmpnn(mol, args.ACSF, molgraphs['x'].tolist(), args.usePeriodics, this_dic['model'])

                    if not this_dic['physnet']:
                        atomic_number = []
                        for atom in mol.GetAtoms():
                            atomic_number.append(atom.GetAtomicNum())
                        z = torch.tensor(atomic_number, dtype=torch.long)
                        molgraphs['Z'] = z
                    
                    if not this_dic['physnet']:
                        molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                        molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                        if this_dic['task'] == 'single':
                            molgraphs['mol_sol_wat'] = torch.FloatTensor([value])
                        if this_dic['task'] == 'multi': #TODO
                            molgraphs['mol_gas'] = torch.FloatTensor([value[1][0]]) # unit is eV, substracting the atom reference energy 
                            molgraphs['mol_wat'] = torch.FloatTensor([value[1][1]]) # unit is eV, substracting the atom reference energy 
                            molgraphs['mol_oct'] = torch.FloatTensor([value[1][2]]) # unit is eV, substracting the atom reference energy 
                            molgraphs['mol_sol_wat'] = torch.FloatTensor([value[1][3]])
                            molgraphs['mol_sol_oct'] = torch.FloatTensor([value[1][4]])
                            molgraphs['mol_sol_logp'] = torch.FloatTensor([value[1][5]])
                        molgraphs['atom_y'] = torch.FloatTensor([i[0] for i in rdMolDescriptors._CalcCrippenContribs(mol)])
                        molgraphs['N'] = torch.FloatTensor([mol.GetNumAtoms()])
                        if args.dmpnn:
                            molgraphs['n_atoms'] = mol_graph.n_atoms
                            molgraphs['n_bonds'] = mol_graph.n_bonds
                            molgraphs['a2b'] = mol_graph.a2b
                            molgraphs['b2a'] = mol_graph.b2a
                            molgraphs['b2revb'] = mol_graph.b2revb
                    else: # for physnet 
                        mol_sdf = getMol(file, int(id_)) # get mol object from sdf files
                        assert mol_smi.GetNumAtoms() == mol_sdf.GetNumAtoms()
                        examples[file+'_'+str(int(id_))] = [mol_sdf, value]

                    if this_dic['atom_classification']:
                        try:
                            efg = mol2frag(mol, returnidx=True, vocabulary=list(efgs_vocabulary), toEnd=True, extra_included=True, TreatHs='include', isomericSmiles=False)
                            molgraphs['atom_efgs'] = torch.tensor(getAtomToEFGs(efg, efgs_vocabulary)).view(-1).long()
                        except:
                            molgraphs['atom_efgs'] = None
                    if not this_dic['physnet']:
                        examples.append(molgraphs)

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))

            elif 'logp_calc/ALL' in this_dic['dataset']:
                if this_dic['ACSF']:
                    acsf = ACSF(
                                species=['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S'],
                                rcut=10.0,
                                g2_params=[[1, 1], [1, 2], [1, 3]],
                                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)
                for smi, value, id_, file in zip(all_data['QM_SMILES'], all_data['calcLogP'], all_data['ID'], all_data['SourceFile']):
                    molgraphs = {}
                    mol = Chem.MolFromSmiles(smi)
                    mol = Chem.AddHs(mol)
                    

                    if not this_dic['ACSF']:
                        mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    else: # ACSF
                        mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])

                    if this_dic['ACSF']:
                        if file in ['pubchem', 'zinc']:
                            path_to_xyz = '/ext3/Frag20/less_than_10/xyz' # path to the singularity file overlay-50G-10M.ext3
                        else:
                            path_to_xyz = '/ext3/Frag20/{}/xyz'.format(file)
                        file_id = file +'_' + str(int(id_)) # such as 'pubchem_100001'
                        #if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(file_id))):
                        #    MolToXYZFile(mol, os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(file_id)))
                        atoms = ase_read(os.path.join(path_to_xyz, '{}.xyz'.format(file_id))) # path to the singularity file overlay-50G-10M.ext3
                        molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms())))
                        assert mol.GetNumAtoms() == molgraphs['x'].shape[0]

                    if args.dmpnn:
                        mol_graph = MolGraph_dmpnn(mol, args.ACSF, molgraphs['x'].tolist(), args.usePeriodics, this_dic['model'])

                    atomic_number = []
                    for atom in mol.GetAtoms():
                        atomic_number.append(atom.GetAtomicNum())
                    z = torch.tensor(atomic_number, dtype=torch.long)
                    molgraphs['Z'] = z

                    molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                    molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    if this_dic['task'] == 'single':
                        molgraphs['mol_sol_logp'] = torch.FloatTensor([value])
                    if this_dic['task'] == 'multi': #TODO
                        molgraphs['mol_gas'] = torch.FloatTensor([value[1][0]]) # unit is eV, substracting the atom reference energy 
                        molgraphs['mol_wat'] = torch.FloatTensor([value[1][1]]) # unit is eV, substracting the atom reference energy 
                        molgraphs['mol_oct'] = torch.FloatTensor([value[1][2]]) # unit is eV, substracting the atom reference energy 
                        molgraphs['mol_sol_wat'] = torch.FloatTensor([value[1][3]])
                        molgraphs['mol_sol_oct'] = torch.FloatTensor([value[1][4]])
                        molgraphs['mol_sol_logp'] = torch.FloatTensor([value[1][5]])
                    molgraphs['atom_y'] = torch.FloatTensor([i[0] for i in rdMolDescriptors._CalcCrippenContribs(mol)])
                    molgraphs['N'] = torch.FloatTensor([mol.GetNumAtoms()])
                    if args.dmpnn:
                        molgraphs['n_atoms'] = mol_graph.n_atoms
                        molgraphs['n_bonds'] = mol_graph.n_bonds
                        molgraphs['a2b'] = mol_graph.a2b
                        molgraphs['b2a'] = mol_graph.b2a
                        molgraphs['b2revb'] = mol_graph.b2revb

                    if this_dic['atom_classification']:
                        try:
                            efg = mol2frag(mol, returnidx=True, vocabulary=list(efgs_vocabulary), toEnd=True, extra_included=True, TreatHs='include', isomericSmiles=False)
                            molgraphs['atom_efgs'] = torch.tensor(getAtomToEFGs(efg, efgs_vocabulary)).view(-1).long()
                        except:
                            molgraphs['atom_efgs'] = None
                    examples.append(molgraphs)

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))

            elif this_dic['dataset'] == 'solNMR':
                if this_dic['ACSF']:
                    acsf = ACSF(
                                species=['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S'],
                                rcut=10.0,
                                g2_params=[[1, 1], [1, 2], [1, 3]],
                                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)

                for d, value in all_data.items():
                    molgraphs = {}
                    mol = value[0]
                    mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])
                    
                    if not this_dic['ACSF']:
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    if this_dic['ACSF']:
                        if not os.path.exists(os.path.join(alldatapath, 'solALogP', 'split', args.style, this_dic['xyz'], '{}.xyz'.format(d))):
                            MolToXYZFile(mol, os.path.join(alldatapath, 'solALogP', 'split', args.style, this_dic['xyz'], '{}.xyz'.format(d)))
                        atoms = ase_read(os.path.join(alldatapath, 'solALogP', 'split', args.style, this_dic['xyz'], '{}.xyz'.format(d)))
                        molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms())))

                    molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                    molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    if this_dic['task'] == 'single':
                        molgraphs['mol_sol_wat'] = torch.FloatTensor([value[1][3]])
                    if this_dic['task'] == 'multi':
                        molgraphs['mol_gas'] = torch.FloatTensor([value[1][0]]) # unit is eV, substracting the atom reference energy 
                        molgraphs['mol_wat'] = torch.FloatTensor([value[1][1]]) # unit is eV, substracting the atom reference energy 
                        molgraphs['mol_oct'] = torch.FloatTensor([value[1][2]]) # unit is eV, substracting the atom reference energy 
                        molgraphs['mol_sol_wat'] = torch.FloatTensor([value[1][3]])
                        molgraphs['mol_sol_oct'] = torch.FloatTensor([value[1][4]])
                        molgraphs['mol_sol_logp'] = torch.FloatTensor([value[1][5]])
                    molgraphs['atom_y'] = torch.FloatTensor(value[2])
                    molgraphs['N'] = torch.FloatTensor([mol.GetNumAtoms()])
                    
                    atomic_number = []
                    for atom in mol.GetAtoms():
                        atomic_number.append(atom.GetAtomicNum())
                    z = torch.tensor(atomic_number, dtype=torch.long)
                    molgraphs['Z'] = z
                    
                    pos = []
                    for i in range(mol.GetNumAtoms()):
                        position = mol.GetConformer().GetAtomPosition(i) 
                        pos.append([position.x, position.y, position.z])
                    molgraphs['pos'] = pos
                    examples.append(molgraphs)

                    if this_dic['atom_classification']:
                        try:
                            efg = mol2frag(mol, returnidx=True, vocabulary=list(efgs_vocabulary), toEnd=True, extra_included=True, TreatHs='include', isomericSmiles=False)
                            molgraphs['atom_efgs'] = torch.tensor(getAtomToEFGs(efg, efgs_vocabulary)).view(-1).long()
                        except:
                            molgraphs['atom_efgs'] = None

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))

            elif this_dic['dataset'] == 'solALogP':
                if this_dic['ACSF']:
                    acsf = ACSF(
                                species=['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S'],
                                rcut=10.0,
                                g2_params=[[1, 1], [1, 2], [1, 3]],
                                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)
                for d, value in all_data.items():
                    molgraphs = {}
                    mol = value[0]

                    if this_dic['ACSF']:
                        if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(d))):
                            MolToXYZFile(mol, os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(d)))
                        atoms = ase_read(os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(d)))
                        molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms())))

                    if not args.dmpnn:
                        mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])
                    else:
                        mol_graph = MolGraph_dmpnn(mol, args.ACSF, molgraphs['x'].tolist(), args.usePeriodics, this_dic['model'])

                    if not this_dic['ACSF']:
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    
                    atomic_number = []
                    for atom in mol.GetAtoms():
                        atomic_number.append(atom.GetAtomicNum())
                    z = torch.tensor(atomic_number, dtype=torch.long)
                    molgraphs['Z'] = z

                    molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                    molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    if this_dic['task'] == 'single':
                        molgraphs['mol_sol_wat'] = torch.FloatTensor([value[1][3]])
                    if this_dic['task'] == 'multi':
                        molgraphs['mol_gas'] = torch.FloatTensor([value[1][0]]) # unit is eV, substracting the atom reference energy 
                        molgraphs['mol_wat'] = torch.FloatTensor([value[1][1]]) # unit is eV, substracting the atom reference energy 
                        molgraphs['mol_oct'] = torch.FloatTensor([value[1][2]]) # unit is eV, substracting the atom reference energy 
                        molgraphs['mol_sol_wat'] = torch.FloatTensor([value[1][3]])
                        molgraphs['mol_sol_oct'] = torch.FloatTensor([value[1][4]])
                        molgraphs['mol_sol_logp'] = torch.FloatTensor([value[1][5]])
                    molgraphs['atom_y'] = torch.FloatTensor([i[0] for i in rdMolDescriptors._CalcCrippenContribs(mol)])
                    molgraphs['N'] = torch.FloatTensor([mol.GetNumAtoms()])
                    if args.dmpnn:
                        molgraphs['n_atoms'] = mol_graph.n_atoms
                        molgraphs['n_bonds'] = mol_graph.n_bonds
                        molgraphs['a2b'] = mol_graph.a2b
                        molgraphs['b2a'] = mol_graph.b2a
                        molgraphs['b2revb'] = mol_graph.b2revb

                    if this_dic['atom_classification']:
                        try:
                            efg = mol2frag(mol, returnidx=True, vocabulary=list(efgs_vocabulary), toEnd=True, extra_included=True, TreatHs='include', isomericSmiles=False)
                            molgraphs['atom_efgs'] = torch.tensor(getAtomToEFGs(efg, efgs_vocabulary)).view(-1).long()
                        except:
                            molgraphs['atom_efgs'] = None
                    examples.append(molgraphs)

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))

            elif this_dic['dataset'] == 'qm9/u0':
                if this_dic['ACSF']:
                    acsf = ACSF(
                                species=['C', 'F', 'H', 'N', 'O'],
                                rcut=10.0,
                                g2_params=[[1, 1], [1, 2], [1, 3]],
                                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)
                for name, mol in suppl.items():
                    #name = mol.GetProp('_Name')
                    if name not in nmr.keys():
                        continue
                    molgraphs = {}
                    #mol = value[0][0]
                    mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])

                    if not this_dic['ACSF']:
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    if this_dic['BPS']:
                        bp_features = torch.FloatTensor(bp_sym(mol)[0, :mol.GetNumAtoms(), 1:])
                        molgraphs['x'] = torch.cat([molgraphs['x'], bp_features], 1)
                    if this_dic['ACSF']:
                        if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(name))):
                            MolToXYZFile(mol, os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(name)))
                        atoms = ase_read(os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(name)))
                        molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms())))
                    
                    atomic_number = []
                    for atom in mol.GetAtoms():
                        atomic_number.append(atom.GetAtomicNum())
                    z = torch.tensor(atomic_number, dtype=torch.long)
                    molgraphs['Z'] = z

                    molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                    molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    
                    molgraphs['atom_y'] = torch.FloatTensor(nmr[name])
                    molgraphs['mol_y'] = torch.FloatTensor([u0[name]])
                    #molgraphs['atom_y'] = torch.FloatTensor([i[0] for i in rdMolDescriptors._CalcCrippenContribs(mol)])
                    molgraphs['N'] = torch.FloatTensor([mol.GetNumAtoms()])
                    
                    if this_dic['atom_classification']:
                        try:
                            efg = mol2frag(mol, returnidx=True, vocabulary=list(efgs_vocabulary), toEnd=True, extra_included=True, TreatHs='include', isomericSmiles=False)
                            molgraphs['atom_efgs'] = torch.tensor(getAtomToEFGs(efg, efgs_vocabulary)).view(-1).long()
                        except:
                            molgraphs['atom_efgs'] = None
                    examples.append(molgraphs)

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))

            elif this_dic['dataset'] in ['nmr/carbon', 'nmr/hydrogen']:
                if this_dic['ACSF']:
                    acsf = ACSF(
                                species=['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S'],
                                rcut=10.0,
                                g2_params=[[1, 1], [1, 2], [1, 3]],
                                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)
                for id_, mol, tar in zip(all_data['molecule_id'], all_data['rdmol'], all_data['value']):
                    molgraphs = {}
                    mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])

                    if not this_dic['ACSF']:
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    if this_dic['ACSF']:
                        if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(id_))):
                            continue
                        atoms = ase_read(os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(id_)))
                        molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms())))
                    
                    atomic_number = []
                    for atom in mol.GetAtoms():
                        atomic_number.append(atom.GetAtomicNum())
                    z = torch.tensor(atomic_number, dtype=torch.long)
                    molgraphs['Z'] = z
                    molgraphs['N'] = torch.FloatTensor([mol.GetNumAtoms()])

                    molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                    molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    mask = np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)
                    vals = np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)
                    for k, v in tar[0].items():
                        mask[int(k), 0] = 1.0
                        vals[int(k), 0] = v
                    molgraphs['atom_y'] = torch.FloatTensor(vals).flatten()
                    molgraphs['mask'] = torch.FloatTensor(mask).flatten()
                    
                    examples.append(molgraphs)

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))

            else:
                if not this_dic['physnet']:
                    if this_dic['ACSF']:
                        if this_dic['dataset'] in ['sol_exp', 'deepchem/freesol', 'deepchem/delaney']:
                            species = ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S']
                            if args.train_type in ['FT', 'TL']:
                                species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']
                            periodic = False
                        if this_dic['dataset'] in ['deepchem/logp']:
                            species = ['B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'Se', 'Si', 'H']
                            if args.train_type in ['FT', 'TL']:
                                species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']
                            periodic = False
                        if this_dic['dataset'] == 'sol_calc/ALL':
                            species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']
                            periodic = False
                        if this_dic['dataset'] == 'mp/bradley':
                            species = ['B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'Si', 'H']
                            if args.train_type in ['FT', 'TL']:
                                species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']
                            periodic = False
                        if this_dic['dataset'] in ['pka/dataWarrior/acidic', 'pka/dataWarrior/basic']:
                            species = ['B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'Se', 'Si', 'H']
                            if args.train_type in ['FT', 'TL']:
                                species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']
                            periodic = False
                        if this_dic['dataset'] in ['secSolu/set1', 'secSolu/set2']:
                            species = ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S']
                            if args.train_type in ['FT', 'TL']:
                                species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']
                            periodic = False

                        acsf = ACSF(
                                    species=species,
                                    rcut=10.0,
                                    g2_params=[[1, 1], [1, 2], [1, 3]],
                                    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
                                    periodic=periodic)

                    for idx, smi, tar in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target']):
                        if not this_dic['physnet']:
                            molgraphs = {}
                        
                        if not Chem.MolFromSmiles(smi): # be careful with this. 
                            continue
                        if args.train_type in ['FT', 'TL'] and not set([atom.GetSymbol() for atom in Chem.MolFromSmiles(smi).GetAtoms()]) < set(['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']):
                            continue
                        #mol_graph = MolGraph(smi, args.usePeriodics, this_dic['model'])
                        if not this_dic['ACSF']:
                            mol = Chem.MolFromSmiles(smi)
                            mol = Chem.AddHs(mol)
                            mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])
                            molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                        
                        else: # ACSF
                            mol = Chem.MolFromSmiles(smi)
                            mol = Chem.AddHs(mol)
                            mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])

                        if this_dic['ACSF']:
                            if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(idx))):
                                continue
                            atoms = ase_read(os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(idx)))
                            #molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms) # in order to compare ACSF and 2D, keep same train/validation/test sets
                            molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms())))

                        if args.dmpnn:
                            mol_graph = MolGraph_dmpnn(mol, args.ACSF, molgraphs['x'].tolist(), args.usePeriodics, this_dic['model'])

                        atomic_number = []
                        for atom in mol.GetAtoms():
                            atomic_number.append(atom.GetAtomicNum())
                        z = torch.tensor(atomic_number, dtype=torch.long)
                        molgraphs['Z'] = z

                        molgraphs['N'] = torch.FloatTensor([mol.GetNumAtoms()])                    
                        molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                        molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                        molgraphs['mol_y'] = torch.FloatTensor([tar])
                        if args.dmpnn:
                            molgraphs['n_atoms'] = mol_graph.n_atoms
                            molgraphs['n_bonds'] = mol_graph.n_bonds
                            molgraphs['a2b'] = mol_graph.a2b
                            molgraphs['b2a'] = mol_graph.b2a
                            molgraphs['b2revb'] = mol_graph.b2revb
                            
                        if this_dic['dataset'] in ['nmr']:
                            mask = np.zeros((molgraphs['x'].shape[0], 1), dtype=np.float32)
                            vals = np.zeros((molgraphs['x'].shape[0], 1), dtype=np.float32)
                            for k, v in tar[0].items():
                                mask[int(k), 0] = 1.0
                                vals[int(k), 0] = v
                            molgraphs['atom_y'] = torch.FloatTensor(vals).flatten()
                            molgraphs['mask'] = torch.FloatTensor(mask).flatten()
                        
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
                
                else:
                    for idx, smi, tar in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target']):
                        # rule out some compounds 
                        if not Chem.MolFromSmiles(smi): # be careful with this. 
                            continue 
                        if not set([atom.GetSymbol() for atom in Chem.MolFromSmiles(smi).GetAtoms()]) < set(['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']):
                            continue
                        if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', args.style, 'sdf', '{}.sdf'.format(idx))):
                            continue
                        
                        mol_sdf = SDMolSupplier(os.path.join(alldatapath, args.dataset, 'split', args.style, 'sdf', '{}.sdf'.format(idx)))
                        examples[str(idx)] = [mol_sdf[0], tar]
   
                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, args.style, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))
    
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

