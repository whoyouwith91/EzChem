#import os, pickle, sys, time
#from typing import List, Tuple, Union
import torch
from torch_geometric.data import (InMemoryDataset,Data)
from torch_geometric.data import Batch
#from k_gnn import TwoMalkin
from featurization import *
#from torch.utils.data import DataLoader, Dataset, Sampler

#from three_level_frag import cleavage, AtomListToSubMol, standize, mol2frag, WordNotFoundError, counter
#from ifg import identify_functional_groups
#from torch_geometric.utils.convert import to_networkx, from_networkx
#from networkx.linalg.graphmatrix import adjacency_matrix
#from networkx.algorithms.centrality.betweenness import betweenness_centrality
#from sklearn.metrics.pairwise import euclidean_distances

#------------------Naive-------------------------------------
class GraphDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
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
                N=d['N'],
                #smiles=d['smiles'],
                ids=d['id']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#------------------Naive-------------------------------------


class GraphDataset_atom(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(GraphDataset_atom, self).__init__(root, transform, pre_transform, pre_filter)
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
                N=d['N'],
                id=d['ID']      
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class physnet_nmr(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(physnet_nmr, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['nmrshiftdb.pickle', 'nmr_calc.pt']

    @property
    def processed_file_names(self):
        return 'processed.pt'

    def process(self):
        import pandas as pd
        import numpy as np
        import pickle, os
        from ase.io import read as ase_read
        
        with open(self.raw_paths[0], 'rb') as f: # modify this function TODO
            data = pickle.load(f)
        all_data = pd.concat([data['train_df'], data['test_df']])
        
        nmr_calc_data_normalize = torch.load(self.raw_paths[1])

        data_list = []
        for id_, mol, tar in zip(all_data['molecule_id'], all_data['rdmol'], all_data['value']):
            if id_ in [8453, 4742, 8967, 8966, 9614, 8464, 8977, 15897, 8731, 8631, 18238, 16069, \
                                17996, 20813, 9173, 9558, 9559, 8791, 9564, 9567, 9824, 14945, 18273, 8050]: # non overlapping check with CHESHIRE
                        continue
            #if str(id_) not in nmr_calc_data_normalize: continue # only for QM descriptors 

            N = mol.GetNumAtoms()
            N_ = torch.tensor(N).view(-1)
            
            if not os.path.exists('/scratch/dz1061/gcn/chemGraph/data/nmr/carbon/sdf_300confs/minimum/MMFFXYZ/{}.xyz'.format(id_)):
                continue
            atoms = ase_read('/scratch/dz1061/gcn/chemGraph/data/nmr/carbon/sdf_300confs/minimum/MMFFXYZ/{}.xyz'.format(id_))
            pos = atoms.get_positions()
            pos = torch.tensor(pos, dtype=torch.float)

            atomic_number = []
            for atom in mol.GetAtoms():
                atomic_number.append(atom.GetAtomicNum())
            z = torch.tensor(atomic_number, dtype=torch.long)
            
            mask = np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)
            vals = np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)
            for k, v in tar[0].items():
                mask[int(k), 0] = 1.0
                vals[int(k), 0] = v
            atom_y = torch.FloatTensor(vals).flatten()
            mask = torch.FloatTensor(mask).flatten()

            data = Data(R=pos, Z=z, atom_y=atom_y.flatten(), mask=mask.view(-1), N=N_, idx=id_)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            data_edge = self.pre_transform(data, edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                        cutoff=10.0, boundary_factor=100., use_center=True, mol=None, cal_3body_term=False,
                                        bond_atom_sep=False, record_long_range=True)
            data_list.append(data_edge)

        torch.save(self.collate(data_list), self.processed_paths[0])

class physnet_frag14(InMemoryDataset):
    
    #raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
    #           'molnet_publish/qm9.zip')
    #raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    #processed_url = 'https://pytorch-geometric.com/datasets/qm9_v2.zip'

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(physnet_frag14, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['frag14_nmr_params.pt']

    @property
    def processed_file_names(self):
        return 'processed.pt'

    def process(self):
        import pandas as pd
        import numpy as np
        
        with open(self.raw_paths[0], 'rb') as f: # modify this function TODO
            data = torch.load(f)

        data_list = []
        i = 0
        for id_, tar in data.items():
            if not tar: continue  
            if not os.path.exists('/vast/dz1061/frag14/opt_xyz/{}.xyz'.format(id_)):
                continue
            #name = mol.GetProp('_Name')
            #print(name)
            
            #print(id_)
            atoms = ase_read('/vast/dz1061/frag14/opt_xyz/{}.xyz'.format(id_))
            N = atoms.get_global_number_of_atoms()
            #print(N)
            N_ = torch.tensor(N).view(-1)
            pos = atoms.get_positions()
            pos = torch.tensor(pos, dtype=torch.float)
            z = torch.tensor(mol.get_atomic_numbers(), dtype=torch.long)
            
            atom_iso = torch.FloatTensor([v[0] for v in tar])
            atom_ani = torch.FloatTensor([v[1] for v in tar])
            atom_eigens = torch.FloatTensor([v[2] for v in tar])
            
                    #print(y)
            data = Data(R=pos, Z=z, atom_iso=atom_iso, atom_ani=atom_ani, atom_eigens=atom_eigens, N=N_)
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
            i += 1

        torch.save(self.collate(data_list), self.processed_paths[0])
        
class GraphDataset_atom_residue(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(GraphDataset_atom_residue, self).__init__(root, transform, pre_transform, pre_filter)
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
                x_res=d['x_residue'],
                atom_res_map=d['atom_res_map'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                edge_index_res=d['res_edge_index'],
                edge_attr_res=d['res_edge_attr'],
                atom_y=d['atom_y'],
                mask=d['mask'],
                Z=d['Z'],
                N=d['N'],
                id=d['ID']      
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

#-------------------------------------------MultiTask------------------------------------------------
class GraphDataset_multi(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(GraphDataset_multi, self).__init__(root, transform, pre_transform, pre_filter)
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

class GraphDataset_dmpnn_mol(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(GraphDataset_dmpnn_mol, self).__init__(root, transform, pre_transform, pre_filter)
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

class GraphDataset_dmpnn_atom(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(GraphDataset_dmpnn_atom, self).__init__(root, transform, pre_transform, pre_filter)
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
class GraphDataset_single(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(GraphDataset_single, self).__init__(root, transform, pre_transform, pre_filter)
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
    if 'atom_y' in data_list[0]:
        keys = ['N', 'Z',  'atom_y', 'mask']
    if 'mol_y' in data_list[0]:
        keys = ['N', 'Z',  'mol_y']
    
    atom_fdim = data_list[0]['x'].shape[1]
    bond_fdim = atom_fdim+7
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
        if 'atom_y' in mol_graph:
            #print('here')
            batch['atom_y'].append(mol_graph.atom_y)
            batch['mask'].append(mol_graph.mask)
            #keys.remove('mol_y')
        elif 'mol_y' in mol_graph:
            batch['mol_y'].append(mol_graph.mol_y)
            #keys.remove('atom_y')
        batch['N'].append(mol_graph.N)
        batch['Z'].append(mol_graph.Z)
        
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
        #print(key)
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