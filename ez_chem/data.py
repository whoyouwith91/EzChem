import os, pickle
from collections import Counter
from typing import List, Tuple, Union
import torch
from torch.utils.data import DataLoader as loopyLoader, Dataset
import torch.nn.functional as F
from torch_geometric.data import (InMemoryDataset, download_url, extract_tar,
                                  Data)
from torch_geometric.data import Batch
from k_gnn import TwoMalkin, ConnectedThreeMalkin
from k_gnn import DataLoader
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
        data.x = data.x[:, :10]
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
        return data.num_nodes > 1  # Remove graphs with less than 6 nodes.


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

class knnGraph_multi1(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph_multi1, self).__init__(root, transform, pre_transform, pre_filter)
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
                y1=d['y1'],
                y2=d['y2'],
                y3=d['y3'],
                ids=d['id']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

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

#------------------EFGS--------------------------------------
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
def get_len_matrix(len_list):
    len_list = np.array(len_list)
    max_nodes = np.sum(len_list)
    curr_sum = 0
    len_matrix = []
    for l in len_list:
        curr = np.zeros(max_nodes)
        curr[curr_sum:curr_sum + l] = 1
        len_matrix.append(curr)
        curr_sum += l
    return torch.FloatTensor(len_matrix)

def collate(data_list):
        keys = data_list[0].keys
        assert 'batch' not in keys

        batch = Batch()
        for key in keys:
            batch[key] = []
        batch.batch = []
        batch.num_nodes = []
        batch.solute_length_matrix = []

        if 'solvent_x' in keys:
            batch.solvent_batch = []
            batch.solvent_num_nodes = []
            batch.solvent_length_matrix = []

        if 'solvent_edge_index' in keys:
            keys.remove('solvent_edge_index')
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

            if 'solvent_x' in data:
                N_2 = data.solvent_x.shape[0]
                batch.solvent_num_nodes.append(N_2)
                batch.solvent_batch.append(torch.full((N_2, ), i, dtype=torch.long))
                if 'solvent_edge_index' in data:
                    batch.solvent_edge_index.append(data.solvent_edge_index + cumsum_2)

            cumsum_1 += N_1
            cumsum_2 += N_2

        keys = [x for x in batch.keys if x not in ['batch', 'solvent_batch', 'solute_length_matrix', 'solvent_length_matrix']]
        batch.solute_length_matrix = get_len_matrix(batch.num_nodes)
        batch.solvent_length_matrix = get_len_matrix(batch.solvent_num_nodes)

        for key in keys:
            if torch.is_tensor(batch[key][0]):
                batch[key] = torch.cat(
                    batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        batch.batch = torch.cat(batch.batch, dim=-1)
        if 'solvent_x' in data:
            batch.solvent_batch = torch.cat(batch.solvent_batch, dim=-1)
        #
        #

        return batch.contiguous()

class knnGraph_interaction(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph_interaction, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['solute_temp.pt', 'solvent_temp.pt']

    @property
    def processed_file_names(self):
        return '1-interaction-GNN.pt'

    def download(self):
        pass

    def process(self):
        raw_solute_data = torch.load(self.raw_paths[0])
        raw_solvent_data = torch.load(self.raw_paths[1])

        data_list = [
            Data(
                x=solute['x'],
                edge_index=solute['edge_index'],
                edge_attr=solute['edge_attr'],
                y=solute['y'],
                ids=solute['id'],
                solvent_x=solvent['x'],
                solvent_edge_index=solvent['edge_index'],
                solvent_edge_attr=solvent['edge_attr']

                ) for solute, solvent in zip(raw_solute_data, raw_solvent_data)
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def collate_logp(data_list):
        keys = data_list[0].keys
        assert 'batch' not in keys

        batch = Batch()
        for key in keys:
            batch[key] = []
        batch.batch = []
        batch.num_nodes = []
        batch.solute_length_matrix = []

        if 'wat_x' in keys:
            batch.wat_batch = []
            batch.wat_num_nodes = []
            batch.wat_length_matrix = []

        if 'oct_x' in keys:
            batch.oct_batch = []
            batch.oct_num_nodes = []
            batch.oct_length_matrix = []

        if 'wat_edge_index' in keys:
            keys.remove('wat_edge_index')
        if 'oct_edge_index' in keys:
            keys.remove('oct_edge_index')
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

            if 'wat_x' in data:
                N_2 = data.wat_x.shape[0]
                batch.wat_num_nodes.append(N_2)
                batch.wat_batch.append(torch.full((N_2, ), i, dtype=torch.long))
                if 'wat_edge_index' in data:
                    batch.wat_edge_index.append(data.wat_edge_index + cumsum_2)

            if 'oct_x' in data:
                N_2 = data.oct_x.shape[0]
                batch.oct_num_nodes.append(N_2)
                batch.oct_batch.append(torch.full((N_2, ), i, dtype=torch.long))
                if 'oct_edge_index' in data:
                    batch.oct_edge_index.append(data.oct_edge_index + cumsum_2)

            cumsum_1 += N_1
            cumsum_2 += N_2

        keys = [x for x in batch.keys if x not in ['batch', 'wat_batch', 'oct_batch', 'solute_length_matrix', 'wat_length_matrix', 'oct_length_matrix']]
        batch.solute_length_matrix = get_len_matrix(batch.num_nodes)
        batch.wat_length_matrix = get_len_matrix(batch.wat_num_nodes)
        batch.oct_length_matrix = get_len_matrix(batch.oct_num_nodes)

        for key in keys:
            if torch.is_tensor(batch[key][0]):
                batch[key] = torch.cat(
                    batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        batch.batch = torch.cat(batch.batch, dim=-1)
        if 'wat_x' in data:
            batch.wat_batch = torch.cat(batch.wat_batch, dim=-1)
        if 'oct_x' in data:
            batch.oct_batch = torch.cat(batch.oct_batch, dim=-1)
        #
        #

        return batch.contiguous()

class knnGraph_interaction_logp(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph_interaction_logp, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['solute_temp.pt', 'solvent_temp.pt']

    @property
    def processed_file_names(self):
        return '1-interaction-GNN.pt'

    def download(self):
        pass

    def process(self):
        raw_solute_data = torch.load(self.raw_paths[0])
        raw_solvent_data = torch.load(self.raw_paths[1])

        data_list = [
            Data(
                x=solute['x'],
                edge_index=solute['edge_index'],
                edge_attr=solute['edge_attr'],
                y=solute['y'],
                ids=solute['id'],
                wat_x=solvent['wat_x'],
                wat_edge_index=solvent['wat_edge_index'],
                wat_edge_attr=solvent['wat_edge_attr'],
                oct_x=solvent['oct_x'],
                oct_edge_index=solvent['oct_edge_index'],
                oct_edge_attr=solvent['oct_edge_attr']

                ) for solute, solvent in zip(raw_solute_data, raw_solvent_data)
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

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

class DataLoader_interaction(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        super(DataLoader_interaction, self).__init__(dataset, collate_fn=collate, **kwargs)

class DataLoader_interaction_logp(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        super(DataLoader_interaction_logp, self).__init__(dataset, collate_fn=collate_logp, **kwargs)

class DataLoader_WithWater(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        super(DataLoader_WithWater, self).__init__(dataset, collate_fn=collate_WithWater, **kwargs)
#------------------------ Interaction ----------------------------------------------

class get_data_loader():
    def __init__(self, config):
        self.config = config
        self.name = config['dataset']
        self.model = config['model']
        if self.config['train_type'] != 'hpsearch':
           self.train_size, self.val_size = config['train_size'], config['val_size']

        if self.name == None:
           raise ValueError('Please specify one dataset you want to work on!')
        if self.model in ['1-GNN', '1-2-GNN', '1-2-GNN_dropout', '1-2-GNN_swag', '1-2-efgs-GNN', '1-efgs-GNN', '1-interaction-GNN']:
           self.train_loader, self.val_loader, self.test_loader, self.std, self.num_features, self.num_bond_features, self.num_i_2 = self.knn_loader()
        if self.model in ['loopybp', 'wlkernel', 'loopybp_dropout', 'loopybp_swag', 'wlkernel_dropout', 'wlkernel_swag']:
           self.train_loader, self.val_loader, self.test_loader, self.std, self.num_features, self.num_bond_features, self.num_i_2 = self.bpwlkernel_loader()
        if self.config['dataset'] == 'commonProperties':
           self.train_loader, self.num_features, self.num_bond_features, self.num_i_2 = self.knn_loader()
        if self.model in ['VAE', 'TransformerUnsuper']:
           self.tokenizer = MolTokenizer(vocab_file=os.path.join(self.config['vocab_path'], self.config['vocab_name']))
           
           pad_idx = self.tokenizer.pad_token_id
           unk_idx = self.tokenizer.unk_token_id
           eos_idx = self.tokenizer.eos_token_id
           init_idx = self.tokenizer.bos_token_id   
           
           self.vocab_size = self.tokenizer.vocab_size
           TEXT = torchtext.data.Field(use_vocab=False, \
                                       tokenize=self.tokenizer.encode, \
                                       pad_token=pad_idx, \
                                       unk_token=unk_idx, \
                                       eos_token=eos_idx, \
                                       init_token=init_idx, \
                                       batch_first=True)
           IDS = torchtext.data.Field(use_vocab=False, dtype=torch.long, sequential=False)
           self.fields = [("id", IDS), ("SRC", TEXT), ("TRG", TEXT)]

           self.train_loader, self.val_loader, self.test_loader = self.vaeLoader()
    
    def knn_loader(self):

        if self.config['model'] in ['1-2-GNN', '1-2-GNN_dropout']:
            if self.config['taskType'] != 'multi': 
               dataset = knnGraph(root=self.config['data_path'], pre_transform=MyPreTransform(), pre_filter=MyFilter())
            elif self.config['dataset'] == 'commonProperties':
               dataset = knnGraph_multi1(root=self.config['data_path'], pre_transform=MyPreTransform())
            else:
               dataset = knnGraph_multi(root=self.config['data_path'], pre_transform=MyPreTransform(), pre_filter=MyFilter())
            dataset.data.iso_type_2 = torch.unique(dataset.data.iso_type_2, True, True)[1]
            num_i_2 = dataset.data.iso_type_2.max().item() + 1
            #if self.config['train_type'] in ['transfer', 'finetuning']:
            #   num_i_2 = self.config['num_i_2']
            dataset.data.iso_type_2 = F.one_hot(dataset.data.iso_type_2, num_classes=num_i_2).to(torch.float)
        elif self.config['model'] in ['1-2-efgs-GNN', '1-efgs-GNN']:
            if self.config['taskType'] != 'multi':
               dataset = knnGraph_EFGS(root=self.config['data_path'], pre_transform=MyPreTransform_EFGS(), pre_filter=MyFilter())
            elif self.config['dataset'] == 'commonProperties':
               dataset = knnGraph_EFGS_multi1(root=self.config['data_path'], pre_transform=MyPreTransform_EFGS())
            else:
               dataset = knnGraph_EFGS_multi(root=self.config['data_path'], pre_transform=MyPreTransform_EFGS(), pre_filter=MyFilter())
            dataset.data.iso_type_2 = torch.unique(dataset.data.iso_type_2, True, True)[1]
            num_i_2 = dataset.data.iso_type_2.max().item() + 1
            #if self.config['train_type'] in ['transfer', 'finetuning']:
            #   num_i_2 = self.config['num_i_2']
            dataset.data.iso_type_2 = F.one_hot(dataset.data.iso_type_2, num_classes=num_i_2).to(torch.float)
        elif self.config['model'] in ['1-interaction-GNN']:
            if self.config['dataset'] in ['sol_exp', 'ws', 'deepchem/delaney', 'deepchem/freesol', 'attentiveFP/delaney',  'attentiveFP/freesol', 'sol_calc/ALL', 'solOct_calc/ALL']:
                dataset = knnGraph_interaction(root=self.config['data_path'])
                num_features = dataset[0]['x'].shape[1]
                num_bond_features = dataset[0]['edge_attr'].shape[1]
                my_split_ratio = [self.train_size, self.val_size]  #my dataseet
                test_dataset = dataset[my_split_ratio[0]+my_split_ratio[1]:]
                rest_dataset = dataset[:my_split_ratio[0]+my_split_ratio[1]]
                train_dataset = rest_dataset[:my_split_ratio[0]]
                val_dataset = rest_dataset[my_split_ratio[0]:]

                test_loader = DataLoader_interaction(test_dataset, batch_size=self.config['batch_size'], num_workers=0)
                val_loader = DataLoader_interaction(val_dataset, batch_size=self.config['batch_size'], num_workers=0)
                train_loader = DataLoader_interaction(train_dataset, batch_size=self.config['batch_size'], num_workers=0, shuffle=True)
                #part_train_loader = DataLoader_interaction(train_dataset[:10000], batch_size=self.config['batch_size'], num_workers=0)

                return train_loader,  val_loader, test_loader, None, num_features, num_bond_features, None
            elif self.config['dataset'] in ['logp', 'deepchem/logp']:
                dataset = knnGraph_interaction_logp(root=self.config['data_path'])
                num_features = dataset[0]['x'].shape[1]
                num_bond_features = dataset[0]['edge_attr'].shape[1]
                my_split_ratio = [self.train_size, self.val_size]  #my dataseet
                test_dataset = dataset[my_split_ratio[0]+my_split_ratio[1]:]
                rest_dataset = dataset[:my_split_ratio[0]+my_split_ratio[1]]
                train_dataset = rest_dataset[:my_split_ratio[0]]
                val_dataset = rest_dataset[my_split_ratio[0]:]

                test_loader = DataLoader_interaction_logp(test_dataset, batch_size=self.config['batch_size'], num_workers=0)
                val_loader = DataLoader_interaction_logp(val_dataset, batch_size=self.config['batch_size'], num_workers=0)
                train_loader = DataLoader_interaction_logp(train_dataset, batch_size=self.config['batch_size'], num_workers=0, shuffle=True)
                #part_train_loader = DataLoader_interaction_logp(train_dataset[:10000], batch_size=self.config['batch_size'], num_workers=0, shuffle=True)
                return train_loader,  val_loader, test_loader, None, num_features, num_bond_features, None
            elif self.config['dataset'] in ['solWithWater', 'solWithWater_calc/ALL', 'logpWithWater']:
                dataset = knnGraph_WithWater(root=self.config['data_path'])
                num_features = dataset[0]['x'].shape[1]
                num_bond_features = dataset[0]['edge_attr'].shape[1]
                my_split_ratio = [self.train_size, self.val_size]  #my dataseet
                test_dataset = dataset[my_split_ratio[0]+my_split_ratio[1]:]
                rest_dataset = dataset[:my_split_ratio[0]+my_split_ratio[1]]
                train_dataset = rest_dataset[:my_split_ratio[0]]
                val_dataset = rest_dataset[my_split_ratio[0]:]

                test_loader = DataLoader_WithWater(test_dataset, batch_size=self.config['batch_size'], num_workers=0)
                val_loader = DataLoader_WithWater(val_dataset, batch_size=self.config['batch_size'], num_workers=0)
                train_loader = DataLoader_WithWater(train_dataset, batch_size=self.config['batch_size'], num_workers=0, shuffle=True)

                return train_loader,  val_loader, test_loader, None, num_features, num_bond_features, None

        else: # 1-GNN
            if self.config['dataset'] == 'commonProperties':
               dataset = knnGraph_multi1(root=self.config['data_path'])
            elif self.config['dataset'] == 'calcSolLogP':
               dataset = knnGraph_multi(root=self.config['data_path'])
            elif self.config['dataset'] in ['qm9/nmr/carbon', 'qm9/nmr/carbon/smaller', 'qm9/nmr/hydrogen', 'qm9/nmr/allAtoms', 'nmr/hydrogen', 'nmr/carbon']:
               dataset = knnGraph_nmr(root=self.config['data_path'])
            else:
               dataset = knnGraph(root=self.config['data_path'])
            num_i_2 = None
        num_features = dataset.num_features
        num_bond_features = dataset[0]['edge_attr'].shape[1]
        
        if self.config['dataset'] == 'commonProperties':
            train_loader = DataLoader(dataset, batch_size=self.config['batch_size'], num_workers=0, shuffle=True)
            return train_loader, num_features, num_bond_features, num_i_2
        #elif self.config['train_type'] == 'hpsearch':
        #    train_loader = DataLoader(dataset[:10000], batch_size=self.config['batch_size'], num_workers=0, shuffle=True)
        #    val_loader = DataLoader(dataset[10000:], batch_size=self.config['batch_size'], num_workers=0, shuffle=True)
        #    return train_loader, val_loader, None, 1.0, num_features, num_bond_features, num_i_2
        else:
            my_split_ratio = [self.train_size, self.val_size]  #my dataseet
            if self.config['normalize']:
                mean = dataset.data.y[:my_split_ratio[0]+my_split_ratio[1]].mean(dim=0)
                std = dataset.data.y[:my_split_ratio[0]+my_split_ratio[1]].std(dim=0)
                dataset.data.y = (dataset.data.y - mean) / std
            else:
                std = torch.FloatTensor([1.0])  # for test function: not converting back by multiplying std
                test_dataset = dataset[my_split_ratio[0]+my_split_ratio[1]:]
                rest_dataset = dataset[:my_split_ratio[0]+my_split_ratio[1]]
            #rest_dataset = rest_dataset.shuffle()  ## can be used on CV
            train_dataset = rest_dataset[:my_split_ratio[0]]
            val_dataset = rest_dataset[my_split_ratio[0]:]
            #test_dataset = dataset[my_split_ratio[0]+my_split_ratio[1]:]

            test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], num_workers=0)
            train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], num_workers=0, shuffle=True)
            #part_train_loader = DataLoader(train_dataset[:10000], batch_size=self.config['batch_size'], num_workers=0)

            return train_loader, val_loader, test_loader, std, num_features, num_bond_features, num_i_2

    def bpwlkernel_loader(self):
        train_graphs = torch.load(os.path.join(self.config['data_path'], 'train.pt'))
        valid_graphs = torch.load(os.path.join(self.config['data_path'], 'valid.pt'))
        if self.config['train_type'] != 'hpsearch':
           test_graphs = torch.load(os.path.join(self.config['data_path'], 'test.pt'))

        train_loader = loopyLoader(
                                 dataset=train_graphs,
                                 batch_size=self.config['batch_size'],
                                 num_workers=0,
                                 collate_fn=construct_molecule_batch,
                                 shuffle=True)
        valid_loader = loopyLoader(
                                 dataset=valid_graphs,
                                 batch_size=self.config['batch_size'],
                                 num_workers=0,
                                 collate_fn=construct_molecule_batch,
                                 shuffle=False)
        if self.config['train_type'] != 'hpsearch':
            test_loader = loopyLoader(
                                 dataset=test_graphs,
                                 batch_size=self.config['batch_size'],
                                 num_workers=0,
                                 collate_fn=construct_molecule_batch,
                                 shuffle=False)
        num_features = get_atom_fdim()
        num_bond_features = get_bond_fdim()
        std = torch.FloatTensor([1.])
        if self.config['train_type'] != 'hpsearch':
           return train_loader, valid_loader, test_loader, std, num_features, num_bond_features, None
        else:
           return train_loader, valid_loader, None, std, num_features, num_bond_features, None

    def vaeLoader(self):
        train_data, valid_data, test_data = data.TabularDataset.splits(
                path=self.config['data_path'],
                train='train.csv',
                validation='valid.csv',
                test='test.csv',
                format="csv",
                fields=self.fields,)

        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                (train_data, valid_data, test_data),
                batch_size=self.config['batch_size'],
                shuffle=True,
                sort=False,
                device=self.config['device'])
        
        return train_iterator, valid_iterator, test_iterator


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
