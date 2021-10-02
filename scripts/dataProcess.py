import os, pickle, sys, time
from typing import List, Tuple, Union
import torch
import torch.nn.functional as F
from torch_geometric.data import (InMemoryDataset,Data)
from torch_geometric.data import Batch
from k_gnn import TwoMalkin
from featurization import *

from three_level_frag import cleavage, AtomListToSubMol, standize, mol2frag, WordNotFoundError, counter
from ifg import identify_functional_groups
from torch_geometric.utils.convert import to_networkx, from_networkx
from networkx.linalg.graphmatrix import adjacency_matrix
from networkx.algorithms.centrality.betweenness import betweenness_centrality
from sklearn.metrics.pairwise import euclidean_distances

from torch.utils.data import DataLoader, Dataset, Sampler
from random import Random
from tqdm import tqdm 
import numpy as np
import pandas as pd
from typing import Dict, Iterator, List, Optional, Union, Tuple

#vocab = pickle.load(open('/scratch/dz1061/gcn/datasets/EFGS/vocab/ours/ours_vocab.pt', 'rb'))

def smiles2gdata(data):
    mol = Chem.MolFromSmiles(data.smiles)
    #mol = Chem.AddHs(mol)

    a,b ,c, d = mol2frag(mol, returnidx=True, vocabulary=list(vocab), toEnd=True, extra_included=True, TreatHs='include', isomericSmiles=False)
    ass_idx = {x:i for i,t in enumerate(c+d) for x in t}
    ei2 = []

    for bond in mol.GetBonds():
        if ass_idx[bond.GetBeginAtomIdx()] == ass_idx[bond.GetEndAtomIdx()]: continue
        groupA, groupB = ass_idx[bond.GetBeginAtomIdx()], ass_idx[bond.GetEndAtomIdx()]
        ei2.extend([[groupA, groupB],[groupB, groupA]])
    if ei2:
        data.edge_index_3 = torch.LongTensor(ei2).transpose(0,1)
    else:
        data.edge_index_3 = torch.LongTensor()

    vocab_index = torch.LongTensor([list(vocab).index(x) for x in a + b])
    data.assignment_index_3 = torch.LongTensor([[key,value] for key, value in ass_idx.items()]).transpose(0,1)
    data.iso_type_3 = F.one_hot(vocab_index, num_classes=len(vocab)).to(torch.float)

    del data.smiles

    return data

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


class MyPreTransform(object):
    def __call__(self, data):
        x = data.x
        data.x = data.x[:, :10] # 10 because first 10 bits are element types
        data = TwoMalkin()(data)
        #data 1= ConnectedThreeMalkin()(data)
        data.x = x
        
        return data

class MyPreTransform_EFGS(object):
    def __call__(self, data):
        x = data.x
        data.x = data.x[:, :10]
        data = TwoMalkin()(data)
        data.x = x
        data = smiles2gdata(data)

        return data

class MyPreTransform_centrality(object):
    def __call__(self, data):
        G = to_networkx(data)
        centrality = torch.FloatTensor(list(betweenness_centrality(G, k=int(data.N.numpy()[0])).values())).view(-1, 1)
        data.x = torch.cat((data.x, centrality), dim=1)
        return data

class MyPreTransform_twoHop(object):
    def __call__(self, data):
        G = to_networkx(data)
        adj = adjacency_matrix(G).toarray()
        adj2 = adj.dot(adj)
        np.fill_diagonal(adj2, 0)
        data.edge_index_twoHop = torch.tensor(np.array(np.where(adj2 == 1))).long()

        D = euclidean_distances(data.pos)
        rbf = gaussian_rbf(torch.tensor(D).view(-1,1), centers, widths, cutoff, return_dict=True)['rbf'].view(D.shape[0], -1, 64)
        data.edge_attr_twoHop = rbf[data.edge_index_twoHop[0,], data.edge_index_twoHop[1,], :]
        return data

class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes > 1  # Remove graphs with less than 1 nodes.

#------------------Naive-------------------------------------
class knnGraph(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return 'processed.pt'
    
    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d['x'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                mol_y=d['mol_y'],
                Z=d['Z'],
                N=d['N']
                #smiles=d['smiles'],
                #ids=d['id']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#------------------Naive-------------------------------------

#------------------Atom-level and mol-level-------------------------------------
class knnGraph_atom(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph_atom, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return 'processed.pt'
    
    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d['x'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                atom_y=d['atom_y'],
                mol_sol_wat=d['mol_sol_wat'],
                Z=d['Z'],
                N=d['N']
                #smiles=d['smiles'],
                #ids=d['id']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#------------------Atom-level and mol-level-------------------------------------

#------------------solEFGs-------------------------------------
class knnGraph_solEFGs(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph_solEFGs, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return 'processed.pt'
    
    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d['x'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                mol_y=d['mol_y'],
                atom_y=d['atom_efgs']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#------------------SolEFGs-------------------------------------

#------------------solNMR-------------------------------------
class knnGraph_solNMR(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph_solNMR, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return 'processed.pt'
    
    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d['x'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                atom_y=d['atom_y'],
                mol_sol_wat=d['mol_sol_wat'],
                N=d['N'],
                pos=d['pos'],
                Z=d['Z']
                #smiles=d['smiles'],
                #ids=d['id']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#------------------solNMR-------------------------------------

#------------------solALogP-------------------------------------
class knnGraph_solALogP(InMemoryDataset): # TODO
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph_solALogP, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return 'processed.pt'
    
    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d['x'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                atom_y=d['atom_y'],
                mol_sol_wat=d['mol_y'],
                N=d['N'],
                #smiles=d['smiles'],
                #ids=d['id']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#------------------Naive-------------------------------------

#------------------With Mol features-------------------------------------
class knnGraph_mol(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph_mol, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return '1-2-whole.pt'
    
    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d['x'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                y=d['y'],
                features=d['mol_features'],
                smiles=d['smiles'],
                ids=d['id']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#------------------With Mol features-------------------------------------

#------------------NMR--------------------------------------
class knnGraph_nmr(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph_nmr, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return 'processed.pt'
    
    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d['x'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                atom_y=d['atom_y'],
                mask=d['mask'],
                Z=d['Z'],
                N=d['N']      
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#------------------NMR--------------------------------------

#-----------------------PhysNet-------------------------------
class physnet(InMemoryDataset):
    #raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
    #           'molnet_publish/qm9.zip')
    #raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    #processed_url = 'https://pytorch-geometric.com/datasets/qm9_v2.zip'

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(physnet, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return 'processed.pt'

    def process(self):

        #with open(self.raw_paths[1], 'rb') as f: # modify this function TODO
        #    data = pickle.load(f)
        #all_ = pd.concat([data['train_df'], data['test_df']])
        #print(all_['molecule_id'].tolist())
        
        #suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
        #                           sanitize=False)

        data_list = []
        raw_data_list = torch.load(self.raw_paths[0])
        for _, value in raw_data_list.items():
            mol = value[0]
            #name = mol.GetProp('_Name')
            #print(name)
            N = mol.GetNumAtoms()
            #print(N)
            N_ = torch.tensor(N).view(-1)

            #pos = mol.GetItemText(i).split('\n')[4:4 + N]
            pos = []
            for i in range(N):
                position = mol.GetConformer().GetAtomPosition(i) 
                pos.append([position.x, position.y, position.z])
            pos = torch.tensor(pos, dtype=torch.float)

            atomic_number = []
            for atom in mol.GetAtoms():
                atomic_number.append(atom.GetAtomicNum())
            z = torch.tensor(atomic_number, dtype=torch.long)
            
            #mask = np.zeros((N, 1), dtype=np.float32)
            #vals = np.zeros((N, 1), dtype=np.float32)
            #for k,v in enumerate(value[0][2]):
            #    mask[int(k), 0] = 1.0
            #    vals[int(k), 0] = v
            mol_y = torch.FloatTensor([value[1]]).flatten()
            #atom_y = torch.FloatTensor(value[0][2]).flatten()
            data = Data(R=pos, Z=z, mol_sol_wat=mol_y, N=N_)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            data_edge = self.pre_transform(data, edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                         cutoff=10.0, boundary_factor=100., use_center=True, mol=None, cal_3body_term=False,
                                         bond_atom_sep=False, record_long_range=True)
            data_list.append(data_edge)

        torch.save(self.collate(data_list), self.processed_paths[0])
#---------------------------------------------PhysNet------------------------------------------------

#---------------------------------------------PhysNet_NMR------------------------------------------------
class physnet_nmr(InMemoryDataset):
    #raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
    #           'molnet_publish/qm9.zip')
    #raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    #processed_url = 'https://pytorch-geometric.com/datasets/qm9_v2.zip'

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(physnet_nmr, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['nmr.sdf', 'nmrshiftdb.pickle']

    @property
    def processed_file_names(self):
        return 'processed.pt'

    def process(self):

        with open(self.raw_paths[1], 'rb') as f: # modify this function TODO
            data = pickle.load(f)
        all_ = pd.concat([data['train_df'], data['test_df']])
        #print(all_['molecule_id'].tolist())
        
        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            name = mol.GetProp('_Name')
            #print(name)
            N = mol.GetNumAtoms()
            #print(N)
            N_ = torch.tensor(N).view(-1)

            pos = suppl.GetItemText(i).split('\n')[4:4 + N]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]
            pos = torch.tensor(pos, dtype=torch.float)

            atomic_number = []
            for atom in mol.GetAtoms():
                atomic_number.append(atom.GetAtomicNum())
            z = torch.tensor(atomic_number, dtype=torch.long)
            
            #print('here')
            if int(name) in all_['molecule_id'].tolist():
                #print('here')
                spectra_numbers = all_[all_['molecule_id'] == int(name)]['value'].shape[0]
                #print(spectra_numbers)
                if spectra_numbers > 1:
                    print('multiple spectra found for %s!' % name)
                for i in range(spectra_numbers):
                    mask = np.zeros((N, 1), dtype=np.float32)
                    vals = np.zeros((N, 1), dtype=np.float32)
                    #print(i)
                    tar = all_[all_['molecule_id'] == int(name)]['value'].values[i][0]
                    #print(tar)
                    for k, v in tar.items():
                        mask[int(k), 0] = 1.0
                        vals[int(k), 0] = v
                    y = torch.FloatTensor(vals).flatten()
                    #print(y)
                    data = Data(R=pos, Z=z, atom_y=y, mask=torch.FloatTensor(mask).view(-1), N=N_, idx=int(name))
                    #print(data)

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    data_edge = self.pre_transform(data, edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                         cutoff=10.0, boundary_factor=100., use_center=True, mol=None, cal_3body_term=False,
                                         bond_atom_sep=False, record_long_range=True)
                    #print(data_edge[0].keys())
                    data_list.append(data_edge)
                #if i > 10:
            #    break

        torch.save(self.collate(data_list), self.processed_paths[0])
#---------------------------------------------PhysNet_NMR------------------------------------------------

#---------------------------------------------QM9-NMR------------------------------------------------
class QM9_nmr(InMemoryDataset):
    #raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
    #           'molnet_publish/qm9.zip')
    #raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    #processed_url = 'https://pytorch-geometric.com/datasets/qm9_v2.zip'

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(QM9_nmr, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['gdb9.sdf', 'qm9_NMR.pickle']

    @property
    def processed_file_names(self):
        return 'data_nmr.pt'

    def process(self):

        with open(self.raw_paths[1], 'rb') as f: # modify this function TODO
            d = pickle.load(f)  
        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            name = mol.GetProp('_Name')
            N = mol.GetNumAtoms()
            N_ = torch.tensor(N).view(-1)

            pos = suppl.GetItemText(i).split('\n')[4:4 + N]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]
            pos = torch.tensor(pos, dtype=torch.float)

            atomic_number = []
            for atom in mol.GetAtoms():
                atomic_number.append(atom.GetAtomicNum())
            z = torch.tensor(atomic_number, dtype=torch.long)
            
            if name in d.keys():
                y = torch.tensor(d[name])  
                data = Data(R=pos, Z=z, y=y, N=N_, idx=i)
                #print(data)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                data_edge = self.pre_transform(data, edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                     cutoff=10.0, boundary_factor=100., use_center=True, mol=None, cal_3body_term=False,
                                     bond_atom_sep=False, record_long_range=True)
                #print(data_edge.N.shape)
                data_list.append(data_edge)
            #if i > 10:
            #    break

        torch.save(self.collate(data_list), self.processed_paths[0])
#-------------------------------------------QM9-NMR--------------------------------------------------

#-------------------------------------------MultiTask------------------------------------------------
class knnGraph_multi(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph_multi, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return 'processed.pt'
    
    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d['x'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                mol_gas=d['mol_gas'],
                mol_wat=d['mol_wat'],
                mol_oct=d['mol_oct'],
                mol_sol_wat=d['mol_sol_wat'],
                mol_sol_oct=d['mol_sol_oct'],
                mol_logp=d['mol_sol_logp'],
                N=d['N'],
                Z=d['Z']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#--------------------------------------------MultiTask-----------------------------------------------

#-------------------------------------------DMMPNN------------------------------------------------
class knnGraph_dmpnn(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph_dmpnn, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return 'processed.pt'
    
    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d['x'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                mol_sol_wat=d['mol_sol_wat'],
                N=d['N'],
                Z=d['Z'],
                a2b=d['a2b'],
                b2a=d['b2a'],
                b2revb=d['b2revb'],
                n_atoms=d['n_atoms'],
                n_bonds=d['n_bonds']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#--------------------------------------------DMPNN-----------------------------------------------

class knnGraph_dmpnn_exp(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph_dmpnn_exp, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return 'processed.pt'
    
    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d['x'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                mol_y=d['mol_y'],
                N=d['N'],
                Z=d['Z'],
                a2b=d['a2b'],
                b2a=d['b2a'],
                b2revb=d['b2revb'],
                n_atoms=d['n_atoms'],
                n_bonds=d['n_bonds']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#--------------------------------------------DMPNN-----------------------------------------------

#--------------------------------------------SingleTask----------------------------------------------
class knnGraph_single(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph_single, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return 'processed.pt'
    
    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d['x'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                mol_sol_wat=d['mol_sol_wat'],
                N=d['N'],
                Z=d['Z']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#---------------------------------------------SingleTask---------------------------------------------

#--------------------------------------------SingleTask----------------------------------------------
class knnGraph_logp(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph_logp, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return 'processed.pt'
    
    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d['x'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                mol_sol_wat=d['mol_sol_logp'], # stick with mol_sol_wat first 
                N=d['N'],
                Z=d['Z']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#---------------------------------------------SingleTask---------------------------------------------

#-------------------------------------------------EFGS-----------------------------------------------
class knnGraph_EFGS(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph_EFGS, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return '1-2-whole.pt'

    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d['x'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                y=d['y'],
                ids=d['id'],
                smiles=d['smiles']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#-------------------------------------------EFGS-----------------------------------------------------

#-------------------------------------Interaction----------------------------------------------------
def collate_WithWater(data_list):
        keys = data_list[0].keys
        assert 'batch' not in keys

        batch = Batch()
        for key in keys:
            batch[key] = []
        batch.batch = []
        batch.num_nodes = []
        batch.solute_length_matrix = []

        if 'hyd_solute_x' in keys:
            batch.hyd_solute_batch = []
            batch.hyd_solute_num_nodes = []

        if 'hyd_solute_edge_index' in keys:
            keys.remove('hyd_solute_edge_index')
        if 'edge_index' in keys:
            keys.remove('edge_index')

        props = [
            'edge_index_2', 'assignment_index_2', 'edge_index_3',
            'assignment_index_3', 'assignment_index_2to3'
        ]
        keys = [x for x in keys if x not in props]

        cumsum_1 = N_1 = cumsum_2 = N_2 = cumsum_3 = N_3 = 0

        for i, data in enumerate(data_list):
            for key in keys:
                batch[key].append(data[key])

            N_1 = data.x.shape[0]
            #print(N_1)
            batch.edge_index.append(data.edge_index + cumsum_1)
            batch.batch.append(torch.full((N_1, ), i, dtype=torch.long))
            batch.num_nodes.append(N_1)

            if 'hyd_solute_x' in data:
                N_2 = data.hyd_solute_x.shape[0]
                batch.hyd_solute_num_nodes.append(N_2)
                batch.hyd_solute_batch.append(torch.full((N_2, ), i, dtype=torch.long))
                if 'hyd_solute_edge_index' in data:
                    batch.hyd_solute_edge_index.append(data.hyd_solute_edge_index + cumsum_2)

            cumsum_1 += N_1
            cumsum_2 += N_2

        keys = [x for x in batch.keys if x not in ['batch', 'hyd_solute_batch', 'solute_length_matrix', 'hyd_solute_length_matrix']]

        for key in keys:
            if torch.is_tensor(batch[key][0]):
                batch[key] = torch.cat(
                    batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        batch.batch = torch.cat(batch.batch, dim=-1)
        if 'hyd_solute_x' in data:
            batch.hyd_solute_batch = torch.cat(batch.hyd_solute_batch, dim=-1)
        return batch.contiguous()

class knnGraph_WithWater(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph_WithWater, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['solute_temp.pt', 'hydrated_solute_temp.pt']

    @property
    def processed_file_names(self):
        return '1-GNN-withWater.pt'

    def download(self):
        pass

    def process(self):
        raw_solute_data = torch.load(self.raw_paths[0])
        raw_hydrated_solute_data = torch.load(self.raw_paths[1])

        data_list = [
            Data(
                x=solute['x'],
                edge_index=solute['edge_index'],
                edge_attr=solute['edge_attr'],
                mol_y=solute['y'],
                hyd_solute_x=hyd_solute['x'],
                hyd_solute_edge_index=hyd_solute['edge_index'],
                hyd_solute_edge_attr=hyd_solute['edge_attr'],
                hyd_solute_mask=hyd_solute['mask']

                ) for solute, hyd_solute in zip(raw_solute_data, raw_hydrated_solute_data)
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class DataLoader_WithWater(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        super(DataLoader_WithWater, self).__init__(dataset, collate_fn=collate_WithWater, **kwargs)
#---------------------------------------- Interaction ----------------------------------------------


def collate_dmpnn(data_list):
    """
    A :class:`BatchMolGraph` represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a :class:`MolGraph` plus:

    * :code:`atom_fdim`: The dimensionality of the atom feature vector.
    * :code:`bond_fdim`: The dimensionality of the bond feature vector (technically the combined atom/bond features).
    * :code:`a_scope`: A list of tuples indicating the start and end atom indices for each molecule.
    * :code:`b_scope`: A list of tuples indicating the start and end bond indices for each molecule.
    * :code:`max_num_bonds`: The maximum number of bonds neighboring an atom in this batch.
    * :code:`b2b`: (Optional) A mapping from a bond index to incoming bond indices.
    * :code:`a2a`: (Optional): A mapping from an atom index to neighboring atom indices.
    """
    #keys = ['N', 'Z', 'mol_y']
    keys = ['N', 'Z', 'mol_sol_wat']
    atom_fdim = 50
    bond_fdim = 50+7
    batch = Batch()
    #batch.batch = []
    
    
    for key in keys:
        batch[key] = []
     # Start n_atoms and n_bonds at 1 b/c zero padding
    n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
    n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
    a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
    b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

    # All start with zero padding so that indexing with zero padding returns zeros
    f_atoms = [[0] * atom_fdim]  # atom features
    f_bonds = [[0] * bond_fdim]  # combined atom/bond features
    a2b = [[]]  # mapping from atom index to incoming bond indices
    b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
    b2revb = [0]  # mapping from bond index to the index of the reverse bond
    #batch['Z'] = [0]
    for i, mol_graph in enumerate(data_list):
        
        batch['N'].append(mol_graph.N)
        batch['Z'].append(mol_graph.Z)
        batch['mol_sol_wat'].append(mol_graph.mol_sol_wat)
        #batch['mol_y'].append(mol_graph.mol_y)
        
        #batch.batch.append(torch.full((mol_graph.N.long().item(), ), i, dtype=torch.long))
        f_atoms.extend(mol_graph.x)
        f_bonds.extend(mol_graph.edge_attr)
        
        for a in range(mol_graph.n_atoms):
            a2b.append([b + n_bonds for b in mol_graph.a2b[a]])

        for b in range(mol_graph.n_bonds):
            b2a.append(n_atoms + mol_graph.b2a[b])
            b2revb.append(n_bonds + mol_graph.b2revb[b])

        a_scope.append((n_atoms, mol_graph.n_atoms.item()))
        b_scope.append((n_bonds, mol_graph.n_bonds.item()))
        n_atoms += mol_graph.n_atoms.item()
        #print(a_scope)
        n_bonds += mol_graph.n_bonds.item()

    max_num_bonds = max(1, max(
            len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

    f_atoms = torch.FloatTensor(f_atoms)
    f_bonds = torch.FloatTensor(f_bonds)
    a2b = torch.LongTensor([a2b[a] + [0] * (max_num_bonds - len(a2b[a])) for a in range(n_atoms)])
    b2a = torch.LongTensor(b2a)
    b2revb = torch.LongTensor(b2revb)
    
    for key in keys:
        if torch.is_tensor(batch[key][0]):
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
    
    batch.x = f_atoms
    batch.edge_attr = f_bonds
    batch.a2b = a2b
    batch.b2a = b2a
    batch.b2revb = b2revb
    batch.a_scope = a_scope
    batch.b_scope = b_scope
    
    return batch.contiguous()

class DataLoader_dmpnn(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        super(DataLoader_dmpnn, self).__init__(dataset, collate_fn=collate_dmpnn, **kwargs)