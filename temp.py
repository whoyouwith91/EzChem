import torch
import torchtext
from torchtext import data
from torchtext import datasets
from torchtext.data import TabularDataset

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

class get_data_loader():
    def __init__(self, config):
        self.config = config
        self.name = config['dataset']
        self.model = config['model']
        self.train_size, self.val_size = config['train_size'], config['val_size']

        if self.name == None:
           raise ValueError('Please specify one dataset you want to work on!')
        if self.model not in ['loopybp', 'wlkernel']:
           self.train_loader, self.val_loader, self.test_loader, self.std, self.num_features, self.num_bond_features, self.num_i_2 = self.knn_loader()
        if self.model in ['loopybp', 'wlkernel']:
           self.train_loader, self.val_loader, self.test_loader, self.std, self.num_features, self.num_bond_features, self.num_i_2 = self.bpwlkernel_loader()
        if self.model in ['VAE']:
           self.tokenizer = ByteLevelBPETokenizer(
                        os.path.join(self.config['vocab'], "vocab.json"),
                        os.path.join(self.config['vocab'], "merges.txt"),)
           self.tokenizer._tokenizer.post_processor = BertProcessing(
                        ("</s>", self.tokenizer.token_to_id("</s>")),
                        ("<s>", self.tokenizer.token_to_id("<s>")),)

           self.tokenizer.enable_truncation(max_length=512)
           pad_idx = self.tokenizer.token_to_id('<pad>')
           unk_idx = self.tokenizer.token_to_id('<unk>')
           eos_idx = self.tokenizer.token_to_id('</s>')
            
           ### let's defined a customized tokenizer
           def custom_tokenizer(text):
                return self.tokenizer.encode(text).ids

           TEXT = torchtext.data.Field(use_vocab=False, tokenize=custom_tokenizer, pad_token=pad_idx, unk_token=unk_idx, \
                            eos_token=eos_idx, batch_first=True, fix_length=self.tokenizer.get_vocab_size())
           self.fields = [("SRC", TEXT), ("TRG", TEXT)]

           self.train_loader, self.val_loader, self.test_loader = self.vaeLoader()



    def knn_loader(self):

        if self.config['model'] in ['1-2-GNN', 'NNConvGAT', 'gradcam', 'dropout']:
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

        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], num_workers=4)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], num_workers=4, shuffle=True)

        return train_loader, val_loader, test_loader, std, num_features, num_bond_features, num_i_2

    def bpwlkernel_loader(self):
        train_graphs = torch.load(os.path.join(self.config['data_path'], 'train.pt'))
        valid_graphs = torch.load(os.path.join(self.config['data_path'], 'valid.pt'))
        test_graphs = torch.load(os.path.join(self.config['data_path'], 'test.pt'))

        train_loader = loopyLoader(
                                 dataset=train_graphs,
                                 batch_size=self.config['batch_size'],
                                 num_workers=2,
                                 collate_fn=construct_molecule_batch,
                                 shuffle=True)
        valid_loader = loopyLoader(
                                 dataset=valid_graphs,
                                 batch_size=self.config['batch_size'],
                                 num_workers=2,
                                 collate_fn=construct_molecule_batch,
                                 shuffle=False)
        test_loader = loopyLoader(
                                 dataset=test_graphs,
                                 batch_size=self.config['batch_size'],
                                 num_workers=2,
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
                device=self.config['device'])
        
        return train_iterator, valid_iterator, test_iterator

