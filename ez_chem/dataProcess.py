import os, pickle
from collections import Counter
from typing import List, Tuple, Union
import torch
import torch.nn.functional as F
from torch_geometric.data import (InMemoryDataset, download_url, extract_tar,
                                  Data)
from torch_geometric.data import Batch
from k_gnn import TwoMalkin, ConnectedThreeMalkin
from featurization import *

from three_level_frag import cleavage, AtomListToSubMol, standize, mol2frag, WordNotFoundError, counter
from ifg import identify_functional_groups
import numpy as np

vocab = pickle.load(open('/scratch/dz1061/gcn/datasets/EFGS/vocab/ours/ours_vocab.pt', 'rb'))

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

class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes > 1  # Remove graphs with less than 1 nodes.

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
                ids=d['id']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

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
                mask=d['mask'],
                ids=d['id']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#------------------NMR--------------------------------------

#------------------MultiTask--------------------------------------
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
                y=d['y0'],
                y1=d['y1'],
                y2=d['y2'],
                ids=d['id']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#------------------MultiTask--------------------------------------

#------------------EFGS-------------------------------------------
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
#----------------EFGS--------------------------------------------

#----------------Interaction----------------------------------------------------
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
                y=solute['y'],
                ids=solute['id'],
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
#------------------------ Interaction ----------------------------------------------

#------------------------Other functions-------------------------------------------
class MoleculeDataset(Dataset):
    """A MoleculeDataset contains a list of molecules and their associated features and targets."""

    def __init__(self, data):
        """
        Initializes a MoleculeDataset, which contains a list of MoleculeDatapoints (i.e. a list of molecules).
        :param data: A list of MoleculeDatapoints.
        """
        self._data = data
        self._scaler = None
        self._batch_graph = None
        self.num_graphs = len(self._data)

    def smiles(self):
        """
        Returns the smiles strings associated with the molecules.
        :return: A list of smiles strings.
        """
        return [d['smiles'] for d in self._data]

    def batch_graph(self, cache: bool = False):
        """
        Returns a BatchMolGraph with the graph featurization of the molecules.
        :param cache: Whether to store the graph featurizations in the global cache.
        :return: A BatchMolGraph.
        """

        return BatchMolGraph(self._data)


    def targets(self):
        """
        Returns the targets associated with each molecule.
        :return: A list of lists of floats containing the targets.
        """
        return [d['y'] for d in self._data]
    

    def __len__(self):
        """
        Returns the length of the dataset (i.e. the number of molecules).
        :return: The length of the dataset.
        """
        return len(self._data)

    def __getitem__(self, item):
        """
        Gets one or more MoleculeDatapoints via an index or slice.
        :param item: An index (int) or a slice object.
        :return: A MoleculeDatapoint if an int is provided or a list of MoleculeDatapoints if a slice is provided.
        """
        return self._data[item]


class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.
    A BatchMolGraph contains the attributes of a MolGraph plus:
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs):
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim()

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph['atom_features'])
            f_bonds.extend(mol_graph['bond_features'])

            for a in range(mol_graph['n_atoms']):
                a2b.append([b + self.n_bonds for b in mol_graph['a2b'][a]])

            for b in range(mol_graph['n_bonds']):
                b2a.append(self.n_atoms + mol_graph['b2a'][b])
                b2revb.append(self.n_bonds + mol_graph['b2revb'][b])

            self.a_scope.append((self.n_atoms, mol_graph['n_atoms']))
            self.b_scope.append((self.n_bonds, mol_graph['n_bonds']))
            self.n_atoms += mol_graph['n_atoms']
            self.n_bonds += mol_graph['n_bonds']

        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages
        self.edge_index = torch.LongTensor(np.concatenate([mol_graph['atomBegin'], mol_graph['atomEnd']]).reshape(2,-1))

    def get_components(self, atom_messages: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                                                   torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                                                   List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.
        :param atom_messages: Whether to use atom messages instead of bond messages. This changes the bond features
        to contain only bond features rather than a concatenation of atom and bond features.
        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        if atom_messages:
            f_bonds = self.f_bonds[:, :get_bond_fdim(atom_messages=atom_messages)]
        else:
            f_bonds = self.f_bonds

        return self.f_atoms, f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.
        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.
        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a

def construct_molecule_batch(data):
            """
            Constructs a MoleculeDataset from a list of MoleculeDatapoints while also constructing the BatchMolGraph.
            :param data: A list of MoleculeDatapoints.
            :return: A MoleculeDataset with all the MoleculeDatapoints and a BatchMolGraph graph featurization.
            """
            data = MoleculeDataset(data)
            data.batch_graph(cache=True)  # Forces computation and caching of the BatchMolGraph for the molecules

            return data