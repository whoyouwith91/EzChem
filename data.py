import os
from typing import List, Tuple, Union
import torch
from torch.utils.data import DataLoader as loopyLoader, Dataset
import torch.nn.functional as F
from torch_geometric.data import (InMemoryDataset, download_url, extract_tar,
                                  Data)
from k_gnn import TwoMalkin, ConnectedThreeMalkin
from k_gnn import DataLoader
from featurization import *

import torchtext
from torchtext import data
from torchtext import datasets
from torchtext.data import TabularDataset

#from tokenizers.implementations import ByteLevelBPETokenizer
#from tokenizers.processors import BertProcessing
from transformers import PreTrainedTokenizer
# from knn
class MyPreTransform(object):
    def __call__(self, data):
        x = data.x
        data.x = data.x[:, :10]
        data = TwoMalkin()(data)
        #data 1= ConnectedThreeMalkin()(data)
        data.x = x
        
        return data

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

class get_data_loader():
    def __init__(self, config):
        self.config = config
        self.name = config['dataset']
        self.model = config['model']
        self.train_size, self.val_size = config['train_size'], config['val_size']

        if self.name == None:
           raise ValueError('Please specify one dataset you want to work on!')
        if self.model in ['1-2-GNN', '1-2-GNN_dropout', '1-2-GNN_swag']:
           self.train_loader, self.val_loader, self.test_loader, self.std, self.num_features, self.num_bond_features, self.num_i_2 = self.knn_loader()
        if self.model in ['loopybp', 'wlkernel', 'loopybp_dropout', 'loopybp_swag', 'wlkernel_dropout', 'wlkernel_swag']:
           self.train_loader, self.val_loader, self.test_loader, self.std, self.num_features, self.num_bond_features, self.num_i_2 = self.bpwlkernel_loader()
        if self.model in ['VAE']:
           self.tokenizer = MolTokenizer(vocab_file=os.path.join(self.config['vocab_path'], self.config['vocab_name']))
           
           pad_idx = self.tokenizer.pad_token_id
           unk_idx = self.tokenizer.unk_token_id
           eos_idx = self.tokenizer.eos_token_id
            
           
           self.vocab_size = self.tokenizer.vocab_size
           TEXT = torchtext.data.Field(use_vocab=False, tokenize=self.tokenizer.encode, pad_token=pad_idx, unk_token=unk_idx, \
                            eos_token=eos_idx, batch_first=True)
           IDS = torchtext.data.Field(use_vocab=False, dtype=torch.long, sequential=False)
           self.fields = [("id", IDS), ("SRC", TEXT), ("TRG", TEXT)]

           self.train_loader, self.val_loader, self.test_loader = self.vaeLoader()
    
    def knn_loader(self):

        if self.config['model'] in ['1-2-GNN', 'NNConvGAT', 'gradcam', '1-2-GNN_dropout']:
            dataset = knnGraph(root=self.config['data_path'], pre_transform=MyPreTransform())
            dataset.data.iso_type_2 = torch.unique(dataset.data.iso_type_2, True, True)[1]
            num_i_2 = dataset.data.iso_type_2.max().item() + 1
            dataset.data.iso_type_2 = F.one_hot(dataset.data.iso_type_2, num_classes=num_i_2).to(torch.float)
        else:
           dataset = knnGraph(root=self.config['data_path'])
           num_i_2 = None
        num_features = dataset.num_features
        num_bond_features = dataset[0]['edge_attr'].shape[1]
        
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

        return train_loader, val_loader, test_loader, std, num_features, num_bond_features, num_i_2

    def bpwlkernel_loader(self):
        train_graphs = torch.load(os.path.join(self.config['data_path'], 'train.pt'))
        valid_graphs = torch.load(os.path.join(self.config['data_path'], 'valid.pt'))
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
        test_loader = loopyLoader(
                                 dataset=test_graphs,
                                 batch_size=self.config['batch_size'],
                                 num_workers=0,
                                 collate_fn=construct_molecule_batch,
                                 shuffle=False)
        num_features = get_atom_fdim()
        num_bond_features = get_bond_fdim()
        std = torch.FloatTensor([1.])
        return train_loader, valid_loader, test_loader, std, num_features, num_bond_features, None
    
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

class MolTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a Molecular tokenizer. Based on SMILES.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.

    Args:
        vocab_file (:obj:`string`, `optional`, defaults to ''):
            File containing the vocabulary (torchtext.vocab.Vocab class).
        source_files (:obj:`string`, `optional`, defaults to ''):
            File containing source data files, vocabulary would be built based on the source file(s).
        unk_token (:obj:`string`, `optional`, defaults to '<unk>'):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to '<s>'):
            string: a beginning of sentence token.
        pad_token (:obj:`string`, `optional`, defaults to "<blank>"):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (:obj:`string`, `optional`, defaults to '</s>'):
            string: an end of sentence token.
        **kwargs：
            Arguments passed to `~transformers.PreTrainedTokenizer`
    """

    def __init__(
        self,
        vocab_file='',
        source_files='',
        unk_token='<unk>',
        bos_token='<s>',
        pad_token="<blank>",
        eos_token='</s>',
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            pad_token=pad_token,
            eos_token=eos_token,
            **kwargs)

        self.create_vocab(vocab_file=vocab_file, source_files=source_files)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def merge_vocabs(self, vocabs, vocab_size=None):
        """
        Merge individual vocabularies (assumed to be generated from disjoint
        documents) into a larger vocabulary.
        Args:
            vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
            vocab_size: `int` the final vocabulary size. `None` for no limit.
        Return:
            `torchtext.vocab.Vocab`
        """
        merged = sum([vocab.freqs for vocab in vocabs], Counter())
        return torchtext.vocab.Vocab(merged,
                                     specials=[self.unk_token, self.pad_token,
                                               self.bos_token, self.eos_token],
                                     max_size=vocab_size)

    def create_vocab(self, vocab_file=None, source_files=None):
        """
        Create a vocabulary from current vocabulary file or from source file(s).
        Args:
            vocab_file (:obj:`string`, `optional`, defaults to ''):
                File containing the vocabulary (torchtext.vocab.Vocab class).
            source_files (:obj:`string`, `optional`, defaults to ''):
                File containing source data files, vocabulary would be built based on the source file(s).
        """
        if (not vocab_file) and (not source_files):
            self.vocab = []
        if vocab_file:
            if not os.path.isfile(vocab_file):
                raise ValueError(
                    "Can't find a vocabulary file at path '{}'.".format(vocab_file)
                )
            else:
                self.vocab = torch.load(vocab_file)

        if source_files:
            if isinstance(source_files,str):
                if not os.path.isfile(source_files):
                    raise ValueError(
                        "Can't find a source file at path '{}'.".format(source_files)
                    )
                else:
                    source_files = [source_files]
            counter = {}
            vocabs = {}
            for i, source_file in enumerate(source_files):
                counter[i] = Counter()
                with open(source_file) as rf:
                    for line in rf:
                        items = self._tokenize(line.strip())
                        counter[i].update(items)
                specials = list(OrderedDict.fromkeys(
                    tok for tok in [self.unk_token, self.pad_token, self.bos_token, self.eos_token]))
                vocabs[i] = torchtext.vocab.Vocab(counter[i], specials=specials)
            self.vocab = self.merge_vocabs([vocabs[i] for i in range(len(source_files))])
            
    def get_vocab(self):
        return self.vocab

    def _tokenize(self, text):
        """
        Tokenize a SMILES molecule or reaction
        """
        import re
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(text)]
        assert text == ''.join(tokens)
        return tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        assert isinstance(self.vocab, torchtext.vocab.Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.stoi[token]

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        assert isinstance(self.vocab, torchtext.vocab.Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.itos[index]

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = "".join(tokens).strip()
        return out_string

    def save_vocabulary(self, vocab_path, vocab_name):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.

        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.

        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, vocab_name)
        else:
            vocab_file = vocab_path
        torch.save(self.vocab, os.path.join(vocab_file, vocab_name))
