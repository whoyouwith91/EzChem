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

MOLFEATURES = [x[0] for x in Descriptors._descList]
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
    parser.add_argument('--Hs', action='store_true') # whether remove Hs
    parser.add_argument('--ACSF', action='store_true')
    parser.add_argument('--cutoff', type=float)
    parser.add_argument('--physnet', action='store_true')
    parser.add_argument('--dmpnn', action='store_true')
    parser.add_argument('--xyz', type=str)
    parser.add_argument('--use_Z', type=int)
    parser.add_argument('--save_path', type=str, default='/scratch/dz1061/gcn/chemGraph/data/')
    parser.add_argument('--style', type=str, help='it is for base or different experiments')
    parser.add_argument('--task', type=str)
    parser.add_argument('--train_type', type=str)

    return parser.parse_args()

def getMol(file, id_, config):
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
    suppl = SDMolSupplier(sdf_file, removeHs=config['Hs'])
    return suppl[0]


def main():
    args = parse_input_arguments()
    this_dic = vars(args)
    alldatapath = '/scratch/dz1061/gcn/chemGraph/data/'

    if 'sol_calc/ALL' in this_dic['dataset'] or 'logp_calc/ALL' in this_dic['dataset']: # multitask
        train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'train.csv'))
        valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'valid.csv'))
        test_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'test.csv'))
        #sol_calc = pd.read_csv('/scratch/dz1061/gcn/chemGraph/data/sol_calc/ALL/split/base/all_info/all_processed_new.csv')
    elif this_dic['dataset'] in ['solNMR', 'solALogP']: # multitask
        train_raw = torch.load(os.path.join(alldatapath, args.dataset, 'split', 'smaller_solNMR_train.pt'))
        valid_raw = torch.load(os.path.join(alldatapath, args.dataset, 'split', 'smaller_solNMR_valid.pt'))
        test_raw = torch.load(os.path.join(alldatapath, args.dataset, 'split', 'smaller_solNMR_test.pt'))
    
    elif this_dic['dataset'] in ['nmr/carbon', 'nmr/hydrogen']: # 
        train_raw = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/{}/split/train.pickle'.format(args.dataset), 'rb'))
        test_raw = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/{}/split/test.pickle'.format(args.dataset), 'rb'))
    
    elif this_dic['dataset'] in ['qm9/nmr/carbon', 'qm9/nmr/hydrogen']: # 
        train_raw = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/{}/split/base/train.pickle'.format(args.dataset), 'rb'))
        valid_raw = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/{}/split/base/valid.pickle'.format(args.dataset), 'rb'))
        test_raw = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/{}/split/base/test.pickle'.format(args.dataset), 'rb'))

    elif 'frag14/nmr' in this_dic['dataset']: #
        all_raw = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/{}/split/base/all_.pt'.format('frag14/nmr'), 'rb'))

    elif this_dic['dataset'] in ['sol_exp/external', 'sol_exp/external/test']:
        data = pd.read_csv('/scratch/dz1061/gcn/chemGraph/data/{}/split/base/sol_exp_external_use.csv'.format(args.dataset))
        inchi_idx = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/sol_exp/external/split/base/inchi_index.pt', 'rb'))
        inchi_mol = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/sol_exp/procssed/inchi_sdf.pt', 'rb'))
    
    elif 'freesol/plus' in this_dic['dataset'] or 'sol_exp'in this_dic['dataset']:
        train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'train.csv'))
        valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'valid.csv'))
        test_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'test.csv'))
        inchi_idx = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/sol_exp/external/split/base/inchi_index.pt', 'rb'))
        inchi_mol = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/sol_exp/procssed/inchi_sdf.pt', 'rb'))
    
    elif this_dic['dataset'] == 'pka/chembl':
        train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'train.csv'))
        valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'valid.csv'))
        test_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'test.csv'))
        inchi_idx = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/pka/chembl/split/base/inchi_idx.pt', 'rb'))
        mol_idx = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/pka/chembl/split/base/mol_idx.pt', 'rb'))
    
    else:
        train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'train.csv'))
        valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'valid.csv'))
        test_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'test.csv'))
    
    if this_dic['atom_classification']: # for different datasets using different vocabulary
        efgs_vocabulary = torch.load('/scratch/dz1061/gcn/chemGraph/data/sol_calc/ALL/split/base/all_info/Frag20-EFGs.pt')
    
    if this_dic['format'] == 'graphs':
        examples = [] if not args.physnet else {}
        if this_dic['dataset'] in ['solNMR', 'solALogP']:
            all_data = {**train_raw, **valid_raw, **test_raw}
        elif this_dic['dataset'] in ['nmr/carbon', 'nmr/hydrogen']:
            all_data = pd.concat([train_raw, test_raw])
        elif this_dic['dataset'] in ['qm9/nmr/carbon', 'qm9/nmr/hydrogen']:
            all_data = pd.concat([train_raw, valid_raw, test_raw])
        elif 'freesol/plus' in this_dic['dataset']:
            all_data = pd.concat([train_raw, valid_raw, test_raw])
        elif 'frag14/nmr' in this_dic['dataset']:
            all_data = all_raw
        elif this_dic['dataset'] == 'qm9/u0':
            suppl = torch.load('/scratch/dz1061/gcn/chemGraph/data/qm9/raw/molObjects.pt')
            u0 = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/qm9/raw/U0_rev.pickle', 'rb'))
            nmr = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/qm9/raw/qm9_NMR.pickle', 'rb'))
        elif this_dic['dataset'] in ['sol_exp/external', 'sol_exp/external/test']:
            all_data = data 
        else:
           all_data = pd.concat([train_raw, valid_raw, test_raw]).reset_index(drop=True)
        
        if False:
            pass
        else:
            if 'sol_calc/ALL' in this_dic['dataset']:
                if this_dic['ACSF']:
                    acsf = ACSF(
                                species=['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S'],
                                rcut=args.cutoff,
                                g2_params=[[1, 1], [1, 2], [1, 3]],
                                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)
                for smi, value, id_, file in zip(all_data['QM_SMILES'], all_data['CalcSol'], all_data['ID'], all_data['SourceFile']):
                    #print(smi, id_, file)
                    if not this_dic['physnet']:
                        molgraphs = {}
                    #mol = Chem.MolFromSmiles(smi)
                    #mol_smi = mol = Chem.AddHs(mol)
                    mol = getMol(file, int(id_), this_dic)

                    if not this_dic['physnet']:
                        mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])
                        if not this_dic['ACSF']:
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
                        mol_sdf = mol # get mol object from sdf files
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

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))

            elif 'logp_calc/ALL' in this_dic['dataset']:
                if this_dic['ACSF']:
                    acsf = ACSF(
                                species=['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S'],
                                rcut=10.0,
                                g2_params=[[1, 1], [1, 2], [1, 3]],
                                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)
                for smi, value, id_, file in zip(all_data['QM_SMILES'], all_data['calcLogP'], all_data['ID'], all_data['SourceFile']):
                    if not this_dic['physnet']:
                        molgraphs = {}
                    #mol = Chem.MolFromSmiles(smi)
                    #mol_smi = mol = Chem.AddHs(mol)
                    mol = getMol(file, int(id_))

                    if not this_dic['physnet']:
                        mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])
                        if not this_dic['ACSF']:
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

                    atomic_number = []
                    for atom in mol.GetAtoms():
                        atomic_number.append(atom.GetAtomicNum())
                    z = torch.tensor(atomic_number, dtype=torch.long)
                    molgraphs['Z'] = z

                    molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                    molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    if this_dic['task'] == 'single':
                        molgraphs['mol_sol_wat'] = torch.FloatTensor([value]) # change later because inconvienece in dataProcess.py 
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

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))

            elif 'frag14/nmr' in this_dic['dataset']:
                if 'carbon' in this_dic['dataset']: use_Z = 6
                elif 'hydrogen' in this_dic['dataset']: use_Z = 1
                else: use_Z = 'all'

                if this_dic['ACSF']:
                    acsf = ACSF(
                                species=['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S'],
                                rcut=args.cutoff,
                                g2_params=[[1, 1], [1, 2], [1, 3]],
                                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)
                for file_id, value in all_data.items():
                    #print(smi, id_, file)
                    if not this_dic['physnet']:
                        molgraphs = {}
                    #mol = Chem.MolFromSmiles(smi)
                    #mol_smi = mol = Chem.AddHs(mol)
                    file, id_ = file_id.split('_')
                    mol = getMol(file, int(id_), this_dic)

                    if not this_dic['physnet']:
                        mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])
                        if not this_dic['ACSF']:
                            molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)

                    if this_dic['ACSF'] and not this_dic['physnet']:
                        if file in ['pubchem', 'zinc']:
                            path_to_xyz = '/ext3/Frag20/lessthan10/xyz' # path to the singularity file overlay-50G-10M.ext3
                        else:
                            path_to_xyz = '/ext3/Frag20/{}/xyz'.format(file)
                        file_id = file +'_' + str(int(id_)) # such as 'pubchem_100001'
                        if not os.path.exists(os.path.join(path_to_xyz, '{}.xyz'.format(file_id))):
                            MolToXYZFile(mol, os.path.join(path_to_xyz, '{}.xyz'.format(file_id)))
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
                        molgraphs['N'] = torch.FloatTensor([mol.GetNumAtoms()])

                        molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                        molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                        
                        #elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
                        if use_Z in [6, 1]:
                            mask = [float(i == use_Z) for i in atomic_number]
                        else:
                            mask = [1.0] * len(atomic_number)
                        #mask = np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)
                        vals = np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)
                        for k, v in zip(range(mol.GetNumAtoms()), value):
                            #mask[int(k), 0] = m
                            vals[int(k), 0] = v
                        molgraphs['atom_y'] = torch.FloatTensor(vals).flatten()
                        molgraphs['mask'] = torch.FloatTensor(mask).flatten()
                        
                        examples.append(molgraphs)
                    
                    else: # for physnet 
                        mol_sdf = mol # get mol object from sdf files
                        assert mol_smi.GetNumAtoms() == mol_sdf.GetNumAtoms()
                        examples[file+'_'+str(int(id_))] = [mol_sdf, value]

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw', 'temp.pt')) ###
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
                        if not os.path.exists(os.path.join(alldatapath, 'solALogP', 'split', this_dic['xyz'], '{}.xyz'.format(d))):
                            MolToXYZFile(mol, os.path.join(alldatapath, 'solALogP', 'split', this_dic['xyz'], '{}.xyz'.format(d)))
                        atoms = ase_read(os.path.join(alldatapath, 'solALogP', 'split', this_dic['xyz'], '{}.xyz'.format(d)))
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

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw', 'temp.pt')) ###
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
                        if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(d))):
                            MolToXYZFile(mol, os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(d)))
                        atoms = ase_read(os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(d)))
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

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw', 'temp.pt')) ###
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
                        if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(name))):
                            MolToXYZFile(mol, os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(name)))
                        atoms = ase_read(os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(name)))
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

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))

            elif this_dic['dataset'] in ['nmr/carbon', 'nmr/hydrogen']: #
                if this_dic['ACSF']:
                    if args.train_type in ['FT']:
                        species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']
                    else:
                        species = ['C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']
                    acsf = ACSF(
                                species=species,
                                rcut=args.cutoff,
                                g2_params=[[1, 1], [1, 2], [1, 3]],
                                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)
                for id_, mol, tar in zip(all_data['molecule_id'], all_data['rdmol'], all_data['value']):
                    molgraphs = {}
                    mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])

                    if not this_dic['ACSF']:
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    if this_dic['ACSF']:
                        if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(id_))):
                            continue
                        atoms = ase_read(os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(id_)))
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

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style, 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style, 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style, 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))
            
            elif this_dic['dataset'] in ['qm9/nmr/carbon', 'qm9/nmr/hydrogen']:
                if this_dic['ACSF']:
                    acsf = ACSF(
                                species=['C', 'F', 'H', 'N', 'O'],
                                rcut=args.cutoff,
                                g2_params=[[1, 1], [1, 2], [1, 3]],
                                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)
                for id_, mol, tar in zip(all_data['molecule_id'], all_data['rdmol'], all_data['values']):
                    molgraphs = {}
                    mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])

                    if not this_dic['ACSF']:
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    if this_dic['ACSF']:
                        path_to_xyz = '/ext3/qm9/xyz/QMXYZ'
                        #if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(file_id))):
                        #    MolToXYZFile(mol, os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(file_id)))
                        atoms = ase_read(os.path.join(path_to_xyz, '{}.xyz'.format(id_))) # path to the singularity file overlay-50G-10M.ext3
                        molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms())))
                    
                    if not this_dic['physnet']:
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
                    else:
                        mol_sdf = mol
                        examples[id_] = [mol_sdf, tar]

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))


            elif this_dic['dataset'] in ['sol_exp/external', 'sol_exp/external/test']:
                species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']
                if this_dic['ACSF']:
                    acsf = ACSF(
                            species=species,
                            rcut=args.cutoff,
                            g2_params=[[1, 1], [1, 2], [1, 3]],
                            g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
                            periodic=False)
                
                for inchi, tar, set_ in zip(all_data['InChI'], all_data['sol'], all_data['set']):
                        #if uncertainty > 0.1: 
                        #   continue  
                        #if tar < -5.0:
                        #    continue
                        if set_ not in ['Test']:
                            continue
                        if inchi not in inchi_mol:
                            continue 
                        idx = inchi_idx[inchi] # see jupyter notebook for why having this 
                        mol = inchi_mol[inchi] # mol is from sdf file.\

                        if not set([atom.GetSymbol() for atom in mol.GetAtoms()]) < set(['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']):
                            continue

                        if not this_dic['physnet']:
                            molgraphs = {}
                        
                        #if not Chem.MolFromInchi(inchi): # be careful with this. 
                        #    continue
                        
                        if not this_dic['ACSF']:
                            #mol = Chem.MolFromInchi(smi)
                            #mol = Chem.AddHs(mol)
                            mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])
                            molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                        
                        else: # ACSF
                            #mol = Chem.MolFromInchi(smi)
                            #mol = Chem.AddHs(mol)
                            mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])

                        if this_dic['ACSF']:
                            #if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(idx))):
                            #    continue
                            atoms = ase_read(os.path.join('/scratch/dz1061/gcn/chemGraph/data/sol_exp/procssed/MMFFXYZ_rev', '{}.xyz'.format(idx)))
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
                        
                        molgraphs['id'] = torch.FloatTensor([idx])
                        examples.append(molgraphs)

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))

            elif 'freesol/plus' in this_dic['dataset']:
                species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']
                if this_dic['ACSF']:
                    acsf = ACSF(
                            species=species,
                            rcut=args.cutoff,
                            g2_params=[[1, 1], [1, 2], [1, 3]],
                            g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
                            periodic=False)
                
                for inchi, tar in zip(all_data['InChI'], all_data['sol']):
                        
                        if inchi not in inchi_mol:
                            continue 
                        idx = inchi_idx[inchi] # see jupyter notebook for why having this 
                        mol = inchi_mol[inchi] # mol is from sdf file.\

                        if not set([atom.GetSymbol() for atom in mol.GetAtoms()]) < set(['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']):
                            continue

                        if not this_dic['physnet']:
                            molgraphs = {}
                        
                        #if not Chem.MolFromInchi(inchi): # be careful with this. 
                        #    continue
                        
                        if not this_dic['ACSF']:
                            #mol = Chem.MolFromInchi(smi)
                            #mol = Chem.AddHs(mol)
                            mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])
                            molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                        
                        else: # ACSF
                            #mol = Chem.MolFromInchi(smi)
                            #mol = Chem.AddHs(mol)
                            mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])

                        if this_dic['ACSF']:
                            #if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(idx))):
                            #    continue
                            atoms = ase_read(os.path.join('/scratch/dz1061/gcn/chemGraph/data/sol_exp/procssed/MMFFXYZ_rev', '{}.xyz'.format(idx)))
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
                        
                        molgraphs['id'] = torch.FloatTensor([idx])
                        examples.append(molgraphs)

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))

            elif this_dic['dataset'] in ['pka/chembl']:
                if this_dic['ACSF']:
                    species = ['N', 'O', 'C', 'Cl', 'F', 'S', 'Br', 'I', 'P', 'B', 'H']
                    if args.train_type in ['FT', 'TL']:
                        species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']
                    periodic = False

                if this_dic['ACSF']:
                    acsf = ACSF(
                            species=species,
                            rcut=args.cutoff,
                            g2_params=[[1, 1], [1, 2], [1, 3]],
                            g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
                            periodic=False)
                
                for inchi, tar, node_id in zip(all_data['InChI'], all_data['pka'], all_data['idx']):
                        
                        #if inchi not in inchi_mol:
                        #    continue 
                        idx = inchi_idx[inchi] # see jupyter notebook for why having this 
                        mol = mol_idx[idx] # mol is from sdf file.\

                        #if not set([atom.GetSymbol() for atom in mol.GetAtoms()]) < set(['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']):
                        #    continue

                        if not this_dic['physnet']:
                            molgraphs = {}
                        
                        #if not Chem.MolFromInchi(inchi): # be careful with this. 
                        #    continue
                        
                        if not this_dic['ACSF']:
                            #mol = Chem.MolFromInchi(smi)
                            #mol = Chem.AddHs(mol)
                            mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])
                            molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                        
                        else: # ACSF
                            #mol = Chem.MolFromInchi(smi)
                            #mol = Chem.AddHs(mol)
                            mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])

                        if this_dic['ACSF']:
                            #if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(idx))):
                            #    continue
                            atoms = ase_read(os.path.join('/scratch/dz1061/gcn/chemGraph/data/pka/chembl/split/base/MMFFXYZ', '{}.xyz'.format(idx)))
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
                            
                        mask = np.zeros((molgraphs['x'].shape[0], 1), dtype=np.float32)
                        mask[int(node_id), 0] = 1.0
                        molgraphs['mask'] = torch.FloatTensor(mask).flatten()
                        
                        molgraphs['id'] = torch.FloatTensor([idx])
                        examples.append(molgraphs)

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, athis_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))

            else:
                if not this_dic['physnet']:
                    if this_dic['ACSF']:
                        if this_dic['dataset'] in ['sol_exp', 'deepchem/freesol', 'deepchem/delaney']:
                            inchi_idx = pickle.load(open(os.path.join(alldatapath, '{}/split/inchi_index.pt'.format(args.dataset)), 'rb'))
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
                            if this_dic['dataset'] == 'secSolu/set1':
                                inchi_idx = torch.load('/scratch/dz1061/gcn/chemGraph/data/secSolu/set1/split/base/inchi_idx_set1.pt')
                        if this_dic['dataset'] in ['bbbp']:
                            species = ['N', 'O', 'C', 'Cl', 'F', 'S', 'Br', 'I', 'P', 'B', 'H']
                            if args.train_type in ['FT', 'TL']:
                                species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']
                            periodic = False                       

                        acsf = ACSF(
                                    species=species,
                                    rcut=args.cutoff,
                                    g2_params=[[1, 1], [1, 2], [1, 3]],
                                    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
                                    periodic=periodic)

                    for idx, inchi, tar in zip(range(all_data.shape[0]), all_data['InChI'], all_data['target']):
                        if not this_dic['physnet']:
                            molgraphs = {}
                        if not Chem.MolFromInchi(inchi): # be careful with this. Though we already checked before featurizing. 
                           continue
                        if args.train_type in ['FT', 'TL'] and not \
                        set([atom.GetSymbol() for atom in mol.GetAtoms()]) < set(['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']):
                            continue
                        #if this_dic['dataset'] in ['secSolu/set1', 'secSolu/set2']:
                        #    if smi not in inchi_idx: # smi is actually index
                        #        continue
                        #    idx = inchi_idx[smi]
                        #else:
                        mol = Chem.MolFromInchi(inchi)
                        mol = Chem.AddHs(mol)
                        mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])
                        if not this_dic['ACSF']:
                            #mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])
                            molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)

                        if this_dic['ACSF']:
                            id_ = inchi_idx[inchi]
                            if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(id_))):
                                continue
                            atoms = ase_read(os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(id_)))
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
                        
                        molgraphs['InChI'] = inchi
                        if this_dic['mol_features']:
                            molgraphs['mol_features'] = torch.FloatTensor(all_data.loc[idx, MOLFEATURES].tolist())
                        if this_dic['atom_classification']:
                            efg = mol2frag(Chem.MolFromSmiles(smi), returnidx=True, vocabulary=list(efgs_vocabulary), toEnd=True, extra_included=True, TreatHs='include', isomericSmiles=False)
                            molgraphs['atom_efgs'] = getAtomToEFGs(efg, efgs_vocabulary)

                        molgraphs['id'] = torch.FloatTensor([idx])
                        examples.append(molgraphs)
                
                else:
                    for idx, smi, tar in zip(range(all_data.shape[0]), all_data['SMILES'], all_data['target']):
                        # rule out some compounds 
                        if not Chem.MolFromSmiles(smi): # be careful with this. 
                            continue 
                        #if not set([atom.GetSymbol() for atom in Chem.MolFromSmiles(smi).GetAtoms()]) < set(['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']):
                        #    continue
                        if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', 'sdf', '{}.sdf'.format(idx))):
                            continue
                        
                        mol_sdf = SDMolSupplier(os.path.join(alldatapath, args.dataset, 'split', 'sdf', '{}.sdf'.format(idx)))
                        examples[str(idx)] = [mol_sdf[0], tar]
                
                if this_dic['mol_features']:
                    args.style = args.style + '_mol_features'
                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style, 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format,this_dic['model'], args.style, 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style, 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))
        

if __name__ == '__main__':
    main()

