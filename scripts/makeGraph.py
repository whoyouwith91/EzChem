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
from rdkit.Chem.rdmolfiles import SDMolSupplier, MolFromPDBFile
from rdkit.ML.Descriptors import MoleculeDescriptors
#from three_level_frag import cleavage, AtomListToSubMol, standize, mol2frag, WordNotFoundError, counter
from deepchem.feat import BPSymmetryFunctionInput, CoulombMatrix, CoulombMatrixEig
from dscribe.descriptors import ACSF
from rdkit.Chem.rdmolfiles import MolToXYZFile
from ase.io import read as ase_read
from rdkit import Chem
from prody import *
import mdtraj as md
import itertools, operator

def parse_input_arguments():
    parser = argparse.ArgumentParser(description='Physicochemical prediction')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--QMD', action='store_true')
    parser.add_argument('--solvent', type=str) # if it's HF, B3LYP or mPW. 
    parser.add_argument('--use_crippen', action='store_true')
    parser.add_argument('--use_tpsa', action='store_true')
    parser.add_argument('--file', type=str, default=None) # for file name such as pubchem, zinc, etc in Frag20
    parser.add_argument('--format', type=str, default='graphs')
    parser.add_argument('--model', type=str)
    parser.add_argument('--usePeriodics', action='store_true')
    parser.add_argument('--mol_features', action='store_true')
    parser.add_argument('--removeHs', action='store_true') # whether remove Hs
    parser.add_argument('--ACSF', action='store_true')
    parser.add_argument('--cutoff', type=float)
    parser.add_argument('--dmpnn', action='store_true')
    parser.add_argument('--xyz', type=str)
    parser.add_argument('--use_Z', type=int) # for frag14/nmr filtering on heavy atoms
    parser.add_argument('--save_path', type=str, default='/scratch/dz1061/gcn/chemGraph/data/')
    parser.add_argument('--style', type=str, help='it is for base or different experiments')
    parser.add_argument('--task', type=str)
    parser.add_argument('--train_type', type=str)
    parser.add_argument('--atom_type', type=str, nargs='+')
    parser.add_argument('--residue_embed', action='store_true')

    return parser.parse_args()

def getMol(file, id_, config):
    if config['xyz'] == 'MMFFXYZ':
        data = 'Frag20'
        format_ = '.sdf' 
    elif config['xyz'] == 'QMXYZ':
        data = 'Frag20_QM'
        format_ = '.opt.sdf' # optimized by DFT
    else:
        pass

    if file in ['pubchem', 'zinc']:
        path_to_sdf = '/ext3/{}/lessthan10/sdf/'.format(data) + file # path to the singularity file overlay-50G-10M.ext3
        sdf_file = os.path.join(path_to_sdf, str(id_)+format_)
    elif file in ['CCDC']:
        path_to_sdf = '/ext3/{}/{}/sdf'.format(data, file) # path to the singularity file overlay-50G-10M.ext3
        if config['xyz'] == 'MMFFXYZ':
            sdf_file = os.path.join(path_to_sdf, str(id_)+'_min.sdf')
        else:
            sdf_file = os.path.join(path_to_sdf, str(id_)+'.opt.sdf')
    else:
        path_to_sdf = '/ext3/{}/{}/sdf'.format(data, file)
        sdf_file = os.path.join(path_to_sdf, str(id_)+format_)
    #print(sdf_file)
    
    suppl = SDMolSupplier(sdf_file, removeHs=config['removeHs'])
    return suppl[0]

def GetSequence(protein):
    seq = {}
    for i,j in zip(protein.getSequence(), protein.getResnums()):
        seq[j] = i
    return ''.join(list(seq.values()))

def residueEncoder(residue, allAA=['W', 'S', 'E', 'M', 'P', 'L', 'Q', 'C', 'R', 'G', 'D', 'N', 'H', 'I', 'F', 'Y', 'V', 'A', 'K', 'T']):
    residue_embed = np.zeros((len(residue), len(allAA)))
    for idx, aa in enumerate(residue):
        residue_embed[idx, :][allAA.index(aa)] = 1
        #residue_embed[allAA.index(aa)] = 1
    return residue_embed

def softplus_inverse(x):
    """
    Private function called by rbf_expansion(D, K=64, cutoff=10)
    """
    return torch.log(-torch.expm1(-x)) + x

def _cutoff_fn(D, cutoff):
    """
    Private function called by rbf_expansion(D, K=64, cutoff=10)
    """
    x = D / cutoff
    x3 = x ** 3
    x4 = x3 * x
    x5 = x4 * x

    result = 1 - 6 * x5 + 15 * x4 - 10 * x3
    return result

def gaussian_rbf(D, centers, widths, cutoff, return_dict=False):
    """
    The rbf expansion of a distance
    Input D: matrix that contains the distance between to atoms
          K: Number of generated distance features
    Output: A matrix containing rbf expanded distances
    """

    rbf = _cutoff_fn(D, cutoff) * torch.exp(-widths * (torch.exp(-D) - centers) ** 2)
    if return_dict:
        return {"rbf": rbf}
    else:
        return rbf

import math 
centers = softplus_inverse(torch.linspace(1.0, math.exp(-10.), 64))
centers = torch.nn.functional.softplus(centers)

widths = [softplus_inverse((0.5 / ((1.0 - torch.exp(-torch.as_tensor(10.))) / 64)) ** 2)] * 64
widths = torch.as_tensor(widths).type(torch.double)
widths = torch.nn.functional.softplus(widths)

def main():
    args = parse_input_arguments()
    this_dic = vars(args)
    alldatapath = '/scratch/dz1061/gcn/chemGraph/data/'

    if 'sol_calc' in this_dic['dataset'] or 'logp_calc' in this_dic['dataset']: # multitask
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
        if args.QMD: 
            #nmr_calc_data = torch.load('/scratch/dz1061/gcn/chemGraph/data/nmr/carbon/split/NMR.pt')
            nmr_calc_data_normalize = torch.load('/scratch/dz1061/gcn/chemGraph/data/{}/split//QM_nmr_gjf/B3LYP_6-31G_d/{}/nmr_calc_norm.pt'.format(args.dataset, args.solvent))
            #nmr_calc_data_ANI_normalize = torch.load('/scratch/dz1061/gcn/chemGraph/data/{}/split/nmr_calc_data_norm_mPW_ANI.pt'.format(args.dataset))
    elif 'protein/nmr' in this_dic['dataset']: # 
        train_raw = pd.read_pickle(os.path.join(alldatapath, args.dataset, 'split', 'train.pt'))
        test_raw = pd.read_pickle(os.path.join(alldatapath, args.dataset, 'split', 'test.pt'))
    
    elif this_dic['dataset'] in ['qm9/nmr/carbon', 'qm9/nmr/hydrogen']: # 
        train_raw = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/{}/split/base/train.pickle'.format(args.dataset), 'rb'))
        valid_raw = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/{}/split/base/valid.pickle'.format(args.dataset), 'rb'))
        test_raw = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/{}/split/base/test.pickle'.format(args.dataset), 'rb'))

    elif 'frag14/nmr' in this_dic['dataset']: #
        all_raw = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/{}/split/base/all_.pt'.format('frag14/nmr'), 'rb'))
    
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
    
    #if this_dic['atom_classification']: # for different datasets using different vocabulary
    #    efgs_vocabulary = torch.load('/scratch/dz1061/gcn/chemGraph/data/sol_calc/ALL/split/base/all_info/Frag20-EFGs.pt')
    
    if this_dic['format'] == 'graphs':
        examples = []
        if this_dic['dataset'] in ['solNMR', 'solALogP']:
            all_data = {**train_raw, **valid_raw, **test_raw}
        elif this_dic['dataset'] in ['nmr/carbon', 'nmr/hydrogen']:
            all_data = pd.concat([train_raw, test_raw])
            if 'use_comp' in args.style:
                all_data = all_data[all_data['use_comparison']]
        elif 'protein/nmr' in this_dic['dataset']:
            all_data = pd.concat([train_raw, test_raw])
        elif this_dic['dataset'] in ['qm9/nmr/carbon', 'qm9/nmr/hydrogen']:
            all_data = pd.concat([train_raw, valid_raw, test_raw])
        elif 'frag14/nmr' in this_dic['dataset']:
            all_data = all_raw
        elif this_dic['dataset'] == 'qm9/u0':
            suppl = torch.load('/scratch/dz1061/gcn/chemGraph/data/qm9/raw/molObjects.pt')
            u0 = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/qm9/raw/U0_rev.pickle', 'rb'))
            nmr = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/qm9/raw/qm9_NMR.pickle', 'rb'))
        else:
           all_data = pd.concat([train_raw, valid_raw, test_raw]).reset_index(drop=True)
        
        if False:
            pass
        else:
            if 'sol_calc' in this_dic['dataset'] or 'logp_calc' in this_dic['dataset']:
                if this_dic['task'] == 'multi':
                    atom_ref = np.load(open('/scratch/dz1061/gcn/chemGraph/data/qm9/raw/atomref.B3LYP_631Gd.10As.npz', 'rb'))
                if this_dic['ACSF']:
                    acsf = ACSF(
                                species=['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S'],
                                rcut=args.cutoff,
                                g2_params=[[1, 1], [1, 2], [1, 3]],
                                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)
                if 'sol_calc' in this_dic['dataset']: values = all_data['CalcSol'].tolist()
                else: values = all_data['calcLogP'].tolist()
                
                c_test = 0
                for locid, smi, value, id_, file, split in zip(range(all_data.shape[0]), all_data['QM_SMILES'], values, all_data['ID'], all_data['SourceFile'], all_data['split']):
                    #print(smi, id_, file)
                    molgraphs = {}
                    #mol = Chem.MolFromSmiles(smi)
                    #mol_smi = mol = Chem.AddHs(mol)
                    mol = getMol(file, int(id_), this_dic)

                    mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model']) # 2d or 3d
                    if not this_dic['ACSF']:
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms) # 2d

                    if this_dic['ACSF']: #3d
                        folder = 'Frag20' if args.xyz == 'MMFFXYZ' else 'Frag20_QM'
                        if file in ['pubchem', 'zinc']:
                            path_to_xyz = '/ext3/{}/lessthan10/xyz'.format(folder) # path to the singularity file overlay-50G-10M.ext3
                        else:
                            path_to_xyz = '/ext3/{}/{}/xyz'.format(folder, file)
                        file_id = str(file) +'_' + str(int(id_)) # such as 'pubchem_100001'
                        #if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(file_id))):
                        #    MolToXYZFile(mol, os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(file_id)))
                        atoms = ase_read(os.path.join(path_to_xyz, '{}.xyz'.format(file_id))) # path to the singularity file overlay-50G-10M.ext3
                        molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms())))
                        assert mol.GetNumAtoms() == molgraphs['x'].shape[0]
                        if args.use_crippen:
                            molgraphs['x'] = torch.cat([molgraphs['x'], torch.FloatTensor(rdMolDescriptors._CalcCrippenContribs(mol))[:,0].view(-1,1)], dim=-1)

                    if args.dmpnn: # dpmnn
                        mol_graph = MolGraph_dmpnn(mol, args.ACSF, molgraphs['x'].tolist(), args.usePeriodics, this_dic['model'])

                    if True: # 2d or 3d
                        atomic_number = []
                        for atom in mol.GetAtoms():
                            atomic_number.append(atom.GetAtomicNum())
                        z = torch.tensor(atomic_number, dtype=torch.long)
                        molgraphs['Z'] = z

                        if this_dic['task'] == 'multi':
                            total_ref = 0
                            for z_ in atomic_number:
                                total_ref += atom_ref['atom_ref'][:,1][z_]

                        molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                        molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                        if this_dic['task'] == 'single':
                            molgraphs['mol_y'] = torch.FloatTensor([value])
                        elif this_dic['task'] == 'multi': #
                            molgraphs['mol_gas'] = torch.FloatTensor([all_data.loc[locid, 'gasEnergy'] *  27.2114 - total_ref]) # unit is eV, substracting the atom reference energy 
                            molgraphs['mol_wat'] = torch.FloatTensor([all_data.loc[locid, 'watEnergy'] * 27.2114 - total_ref]) # unit is eV, substracting the atom reference energy 
                            molgraphs['mol_oct'] = torch.FloatTensor([all_data.loc[locid, 'octEnergy'] * 27.2114 - total_ref]) # unit is eV, substracting the atom reference energy 
                            molgraphs['mol_sol_wat'] = torch.FloatTensor([all_data.loc[locid, 'CalcSol']]) * 0.043 # unit is eV
                            molgraphs['mol_sol_oct'] = torch.FloatTensor([all_data.loc[locid, 'CalcOct']]) * 0.043 # unit is eV
                            molgraphs['mol_sol_logp'] = torch.FloatTensor([all_data.loc[locid, 'calcLogP']])
                        #molgraphs['atom_y'] = torch.FloatTensor([i[0] for i in rdMolDescriptors._CalcCrippenContribs(mol)])
                        molgraphs['N'] = torch.FloatTensor([mol.GetNumAtoms()])
                        if args.dmpnn:
                            molgraphs['n_atoms'] = mol_graph.n_atoms
                            molgraphs['n_bonds'] = mol_graph.n_bonds
                            molgraphs['a2b'] = mol_graph.a2b
                            molgraphs['b2a'] = mol_graph.b2a
                            molgraphs['b2revb'] = mol_graph.b2revb
                    #else: # for physnet 
                    #    mol_sdf = mol # get mol object from sdf files
                    #    assert mol_smi.GetNumAtoms() == mol_sdf.GetNumAtoms()
                    #    examples[file+'_'+str(int(id_))] = [mol_sdf, value]

                    #if this_dic['atom_classification']:
                    #    try:
                    #        efg = mol2frag(mol, returnidx=True, vocabulary=list(efgs_vocabulary), toEnd=True, extra_included=True, TreatHs='include', isomericSmiles=False)
                    #        molgraphs['atom_efgs'] = torch.tensor(getAtomToEFGs(efg, efgs_vocabulary)).view(-1).long()
                    #    except:
                    #        molgraphs['atom_efgs'] = None
                    examples.append(molgraphs)
                    if split == 'test': c_test +=1 

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style, 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style, 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style,'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds, among them {} compounds in test set'.format(len(examples), c_test))

            elif 'frag14/nmr' in this_dic['dataset']:
                
                if 'carbon' in this_dic['dataset']: 
                    use_Z = 6
                    refTMS = 186.96
                    #f = lambda x: (185.3785 - x) / 1.0330
                    f = lambda x: (refTMS - x) / 1.
                elif 'hydrogen' in this_dic['dataset']: 
                    use_Z = 1
                    refTMS = 31.76 # to be determined
                    #f = lambda x: (31.8493 - x) / 1.0355
                    f = lambda x: (refTMS - x) / 1.
                else: use_Z = 'all'

                if this_dic['xyz'] == 'MMFFXYZ': data = 'Frag20'
                elif this_dic['xyz'] == 'QMXYZ': data = 'Frag20_QM'
                
                if this_dic['ACSF']:
                    acsf = ACSF(
                                species=['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S'],
                                rcut=args.cutoff,
                                g2_params=[[1, 1], [1, 2], [1, 3]],
                                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)
                for file_id, value in all_data.items():
                    #print(smi, id_, file)
                    molgraphs = {}
                    #mol = Chem.MolFromSmiles(smi)
                    #mol_smi = mol = Chem.AddHs(mol)
                    file, id_ = file_id.split('_')
                    mol = getMol(file, int(id_), this_dic)

                    mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])
                    if not this_dic['ACSF']:
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)

                    if this_dic['ACSF']:
                        if file in ['pubchem', 'zinc']:
                            path_to_xyz = '/ext3/{}/lessthan10/xyz'.format(data) # path to the singularity file overlay-50G-10M.ext3
                        else:
                            path_to_xyz = '/ext3/{}/{}/xyz'.format(data, file)
                        file_id = file +'_' + str(int(id_)) # such as 'pubchem_100001'
                        if not os.path.exists(os.path.join(path_to_xyz, '{}.xyz'.format(file_id))):
                            MolToXYZFile(mol, os.path.join(path_to_xyz, '{}.xyz'.format(file_id)))
                        atoms = ase_read(os.path.join(path_to_xyz, '{}.xyz'.format(file_id))) # path to the singularity file overlay-50G-10M.ext3
                        molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms())))
                        assert mol.GetNumAtoms() == molgraphs['x'].shape[0]

                    if args.dmpnn:
                        mol_graph = MolGraph_dmpnn(mol, args.ACSF, molgraphs['x'].tolist(), args.usePeriodics, this_dic['model'])

                    if True:
                        atomic_number = []
                        for atom in mol.GetAtoms():
                            atomic_number.append(atom.GetAtomicNum())
                        z = torch.tensor(atomic_number, dtype=torch.long)
                        molgraphs['Z'] = z
                        molgraphs['N'] = torch.FloatTensor([mol.GetNumAtoms()])

                        molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                        molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                        
                        if args.dmpnn:
                            molgraphs['n_atoms'] = mol_graph.n_atoms
                            molgraphs['n_bonds'] = mol_graph.n_bonds
                            molgraphs['a2b'] = mol_graph.a2b
                            molgraphs['b2a'] = mol_graph.b2a
                            molgraphs['b2revb'] = mol_graph.b2revb

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
                        molgraphs['atom_y'] = torch.FloatTensor(f(vals.flatten()))
                        molgraphs['mask'] = torch.FloatTensor(mask).flatten()
                        
                    examples.append(molgraphs)
                    
                    ##else: # for physnet 
                    #    mol_sdf = mol # get mol object from sdf files
                    #    assert mol_smi.GetNumAtoms() == mol_sdf.GetNumAtoms()
                    #    examples[file+'_'+str(int(id_))] = [mol_sdf, value]

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style, 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style, 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style, 'raw', 'temp.pt')) ###
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
                    if id_ in [8453, 4742, 8967, 8966, 9614, 8464, 8977, 15897, 8731, 8631, 18238, 16069, \
                                17996, 20813, 9173, 9558, 9559, 8791, 9564, 9567, 9824, 14945, 18273, 8050]: # non overlapping check with CHESHIRE
                        continue
                    molgraphs = {}
                    mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])
                    #if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(id_))):
                    #    continue
                    if args.QMD and str(id_) not in nmr_calc_data_normalize: continue # only for QM descriptors 

                    if not this_dic['ACSF']:
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    if this_dic['ACSF']:
                        if not os.path.exists(os.path.join(alldatapath, args.dataset, 'sdf_300confs/minimum/', this_dic['xyz'], '{}.xyz'.format(id_))):
                            continue
                        #atoms = ase_read(os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(id_)))
                        atoms = ase_read(os.path.join(alldatapath, args.dataset, 'sdf_300confs/minimum/', this_dic['xyz'], '{}.xyz'.format(id_)))
                        molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms())))
                    if args.QMD: 
                        #shielding_tensors = torch.FloatTensor(nmr_calc_data_normalize[str(id_)]).view(-1,1)
                        shielding_tensors = torch.FloatTensor(nmr_calc_data_normalize[str(id_)]).view(-1,3)
                        molgraphs['x'] = torch.cat([molgraphs['x'], shielding_tensors], dim=-1)
                    if args.dmpnn: # dpmnn
                        mol_graph = MolGraph_dmpnn(mol, args.ACSF, molgraphs['x'].tolist(), args.usePeriodics, this_dic['model'])
                    

                    atomic_number = []
                    for atom in mol.GetAtoms():
                        atomic_number.append(atom.GetAtomicNum())
                    z = torch.tensor(atomic_number, dtype=torch.long)
                    molgraphs['Z'] = z
                    molgraphs['N'] = torch.FloatTensor([mol.GetNumAtoms()])

                    molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                    molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    
                    if args.dmpnn:
                        molgraphs['n_atoms'] = mol_graph.n_atoms
                        molgraphs['n_bonds'] = mol_graph.n_bonds
                        molgraphs['a2b'] = mol_graph.a2b
                        molgraphs['b2a'] = mol_graph.b2a
                        molgraphs['b2revb'] = mol_graph.b2revb

                    mask = np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)
                    vals = np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)
                    for k, v in tar[0].items():
                        mask[int(k), 0] = 1.0
                        vals[int(k), 0] = v
                    molgraphs['atom_y'] = torch.FloatTensor(vals).flatten()
                    molgraphs['mask'] = torch.FloatTensor(mask).flatten()
                    molgraphs['ID'] = torch.LongTensor([id_])
                    examples.append(molgraphs)

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style, 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style, 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style, 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))
            
            elif 'protein/nmr' in this_dic['dataset']: #
                if this_dic['ACSF']:
                    if args.train_type in ['FT']:
                        species = ['C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']
                    else:
                        species = ['C', 'H', 'N', 'O', 'S', 'P', 'Se'] # in original XRD pdb, there is MSE in pdbs such as A023, which contains Se. However, no Se in alphafold2 pdbs.
                    acsf = ACSF(
                                species=species,
                                rcut=args.cutoff,
                                g2_params=[[1, 1], [1, 2], [1, 3]],
                                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)
                train_pdbs = 0
                test_pdbs = 0
                for id_, fileID, tar, atomIdx in zip(range(all_data.shape[0]), all_data['pdb'], all_data['value'], all_data['atoms']):
                    #atom_type = 'CA'
                    if not set(args.atom_type) < set(atomIdx.values()): continue # whether chemical shifts contains the corresponding atom types
                    if fileID.startswith('R'): train_pdbs +=1 
                    if fileID.startswith('A'): test_pdbs +=1
                    
                    molgraphs = {}
                    mol = MolFromPDBFile(os.path.join(alldatapath, args.dataset, 'split/pdb',  '{}.pdb'.format(fileID)), removeHs=False, sanitize=False)
                    mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])
                    

                    if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', 'xyz', '{}.xyz'.format(fileID))):
                        continue
                    if not this_dic['ACSF']: # 2D 
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    if this_dic['ACSF']:
                        if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', 'xyz', '{}.xyz'.format(fileID))):
                            continue
                        atoms = ase_read(os.path.join(alldatapath, args.dataset, 'split', 'xyz', '{}.xyz'.format(fileID)))
                        molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms())))

                    if args.dmpnn: # dpmnn
                        mol_graph = MolGraph_dmpnn(mol, args.ACSF, molgraphs['x'].tolist(), args.usePeriodics, this_dic['model'])
                        
                    atomic_number = []
                    for atom in mol.GetAtoms():
                        atomic_number.append(atom.GetAtomicNum())
                    z = torch.tensor(atomic_number, dtype=torch.long)
                    molgraphs['Z'] = z
                    molgraphs['N'] = torch.FloatTensor([mol.GetNumAtoms()])

                    molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                    molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    
                    if args.residue_embed:
                        
                        res_edge_index = []
                        res_attr = []
                        structure = parsePDB(os.path.join(alldatapath, args.dataset, 'split/pdb',  '{}.pdb'.format(fileID)))
                        seq = GetSequence(structure)
                        molgraphs['x_residue'] = torch.FloatTensor(residueEncoder(seq)) # one-hot embedding for residues 
                        atom_res = [[e[0] for e in d[1]] for d in itertools.groupby(enumerate(structure.getResnums()), key=operator.itemgetter(1))]
                        temp = []
                        for idx, atoms in enumerate(atom_res):
                            temp.append(len(atoms))
                        molgraphs['atom_res_map'] = torch.LongTensor(temp)

                        pdb = md.load_pdb(os.path.join(alldatapath, args.dataset, 'split/pdb',  '{}.pdb'.format(fileID)))
                        for res_idx in range(pdb.n_residues):
                            c = 0
                            for res_2_idx in range(res_idx + 1, pdb.n_residues):
                                if c > 2: break
                                c += 1
                                distance_matr = md.compute_contacts(pdb, contacts=[[res_idx, res_2_idx]])
                                res_edge_index.append(distance_matr[1][0])
                                res_attr.append(gaussian_rbf(torch.as_tensor(distance_matr[0][0].item()), centers, widths, 10.))
                                c += 1
                        molgraphs['res_edge_index'] = torch.LongTensor(np.array(res_edge_index).transpose())
                        molgraphs['res_edge_attr'] = torch.stack(res_attr)


                    if args.dmpnn:
                        molgraphs['n_atoms'] = mol_graph.n_atoms
                        molgraphs['n_bonds'] = mol_graph.n_bonds
                        molgraphs['a2b'] = mol_graph.a2b
                        molgraphs['b2a'] = mol_graph.b2a
                        molgraphs['b2revb'] = mol_graph.b2revb
                    
                    mask = np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)
                    vals = np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)
                    for k, v in tar.items():
                        if atomIdx[k] in args.atom_type: # only select specific atom types
                            mask[int(k), 0] = 1.0
                            vals[int(k), 0] = v
                    molgraphs['atom_y'] = torch.FloatTensor(vals).flatten()
                    molgraphs['mask'] = torch.FloatTensor(mask).flatten()
                    
                    molgraphs['ID'] = torch.LongTensor([id_])
                    examples.append(molgraphs)

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style, 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style, 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style, 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds with train samples of {} and test samples of {}'.format(len(examples), train_pdbs, test_pdbs))

            

            else:
                if True:
                    inchi_idx = pickle.load(open(os.path.join(alldatapath, '{}/split/inchi_index.pt'.format(args.dataset)), 'rb')) # inchi-index pair to xyz files
                    if this_dic['ACSF']:
                        if this_dic['dataset'] in ['ws', 'logp', 'mp_drugs', 'sol_exp']:
                            if this_dic['dataset'] == 'ws':
                                inchi_sdf = pickle.load(open(os.path.join(alldatapath, '{}/split/inchi_sdf.pt'.format(args.dataset)), 'rb'))
                            species = ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'B']
                            if args.train_type in ['FT', 'TL']:
                                species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']
                            periodic = False
                        if this_dic['dataset'] in ['deepchem/freesol', 'deepchem/delaney']:
                            inchi_idx = pickle.load(open(os.path.join(alldatapath, '{}/split/inchi_index.pt'.format(args.dataset)), 'rb'))
                            species = ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S'] # no B 
                            if args.train_type in ['FT', 'TL']:
                                species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S'] # no I
                            periodic = False
                        if this_dic['dataset'] in ['deepchem/logp']:
                            species = ['B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'Se', 'Si', 'H']
                            if args.train_type in ['FT', 'TL']:
                                species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']
                            periodic = False
                        if this_dic['dataset'] == 'mp/bradley':
                            species = ['Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'Si', 'H']
                            #if args.train_type in ['FT', 'TL']:
                            #    species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']
                            periodic = False
                        if this_dic['dataset'] in ['pka/dataWarrior/acidic', 'pka/dataWarrior/basic']:
                            species = ['B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'Si', 'H']
                            if args.train_type in ['FT', 'TL']:
                                species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']
                            periodic = False

                        acsf = ACSF(
                                    species=species,
                                    rcut=args.cutoff,
                                    g2_params=[[1, 1], [1, 2], [1, 3]],
                                    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
                                    periodic=periodic)

                    for inchi, tar in zip(all_data['InChI'], all_data['target']):
                        molgraphs = {}
                        if not Chem.MolFromInchi(inchi): # be careful with this. Though we already checked before featurizing. 
                           continue
                        
                        idx = inchi_idx[inchi]
                        if this_dic['dataset'] == 'logp': idx = idx[0] # ['dc_1434'] for example
                        #mol = Chem.MolFromSmiles(smiles)
                        if this_dic['dataset'] == 'ws' and not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', 'sdf', '{}.sdf'.format(inchi_sdf[inchi]))): continue 
                        if this_dic['dataset'] != 'ws' and not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', 'sdf', '{}.sdf'.format(idx))): continue 
                        if this_dic['dataset'] == 'ws': 
                            mol = SDMolSupplier(os.path.join(alldatapath, args.dataset, 'split', 'sdf', '{}.sdf'.format(inchi_sdf[inchi])), removeHs=args.removeHs)[0]
                        else:
                            mol = SDMolSupplier(os.path.join(alldatapath, args.dataset, 'split', 'sdf', '{}.sdf'.format(idx)), removeHs=args.removeHs)[0]
                        if args.train_type in ['FT', 'TL'] and not \
                        set([atom.GetSymbol() for atom in mol.GetAtoms()]) < set(['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']):
                            continue
                        
                        mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])
                        if not this_dic['ACSF']:
                            #mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])
                            molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)

                        if this_dic['ACSF']:
                            
                            if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(idx))):
                                continue
                            atoms = ase_read(os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(idx)))
                            #molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms) # in order to compare ACSF and 2D, keep same train/validation/test sets
                            molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms())))
                            if args.use_crippen:
                                molgraphs['x'] = torch.cat([molgraphs['x'], torch.FloatTensor(rdMolDescriptors._CalcCrippenContribs(mol))[:,0].view(-1,1)], dim=-1)
                            if args.use_tpsa:
                                molgraphs['x'] = torch.cat([molgraphs['x'], torch.FloatTensor(rdMolDescriptors._CalcTPSAContribs(mol)).view(-1,1)], dim=-1)

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
                            
                        molgraphs['InChI'] = inchi
                        if this_dic['mol_features']:
                            molgraphs['mol_features'] = torch.FloatTensor(all_data.loc[idx, MOLFEATURES].tolist())
                        #if this_dic['atom_classification']:
                        #    efg = mol2frag(Chem.MolFromSmiles(smi), returnidx=True, vocabulary=list(efgs_vocabulary), toEnd=True, extra_included=True, TreatHs='include', isomericSmiles=False)
                        #    molgraphs['atom_efgs'] = getAtomToEFGs(efg, efgs_vocabulary)

                        molgraphs['id'] = idx
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
                
                #if this_dic['mol_features']:
                #    args.style = args.style + '_mol_features'
                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style, 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format,this_dic['model'], args.style, 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], args.style, 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))
        

if __name__ == '__main__':
    main()

