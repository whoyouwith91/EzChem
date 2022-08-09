from dataProcess import *
from k_gnn import DataLoader
from DataPrepareUtils import my_pre_transform #sPhysnet
from utils_functions import collate_fn # sPhysnet
import torch
import random

        
class get_data_loader():
   def __init__(self, config):
      self.config = config
      self.model = config['model']
      if config['explicit_split']:
         if config['dataset'] in ['qm9/u0']:
            all_index = torch.load('/scratch/projects/yzlab/group/temp_sx/qm9_processed/qm9_split.pt')
            self.train_index = all_index['train_index'].tolist()
            self.valid_index = all_index['valid_index'].tolist()
            self.test_index = all_index['test_index'].tolist()
         elif config['dataset'] in ['qm9/nmr/carbon']:
            all_index = torch.load('/scratch/dz1061/gcn/chemGraph/data/qm9/nmr/carbon/split/base/qm9-nmr-split.pt')
            self.train_index = all_index['train_index']
            self.valid_index = all_index['valid_index']
            self.test_index = all_index['test_index']
         self.train_size, self.val_size = config['train_size'], config['val_size']
      else:
         self.train_size, self.val_size, self.test_size = config['train_size'], config['val_size'], config['test_size']
         if config['sample'] or config['vary_train_only']:
            self.data_seed = config['data_seed']
      self.train_loader, self.val_loader, self.test_loader, self.num_features, self.num_bond_features = self.graph_loader()
      
    
   def graph_loader(self):        
      
      if self.config['model'] in ['1-interaction-GNN']:
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

         return train_loader,  val_loader, test_loader, num_features, num_bond_features, None

      elif self.config['model'] in ['physnet']:
         if self.config['dataset'] in ['nmr/carbon', 'nmr/hydrogen', 'protein/nmr', 'protein/nmr/alphaFold']:
            dataset = physnet_nmr(root=self.config['data_path'], pre_transform=my_pre_transform)
            #if self.config['dataset'] == 'qm9':
         elif self.config['dataset'] in ['frag14/nmr']:
            dataset = physnet_frag14(root=self.config['data_path'], pre_transform=my_pre_transform)
         else:
            dataset = physnet(root=self.config['data_path'], pre_transform=my_pre_transform)
         #dataset = DummyIMDataset(root=self.config['data_path'], dataset_name='processed.pt')
         num_i_2 = None
      
      else: # 1-GNN
         if self.config['dataset'] in ['sol_calc/smaller', 'sol_calc/all', 'logp_calc/smaller']:
            if self.config['propertyLevel'] == 'multiMol': # for multiple mol properties
               dataset = GraphDataset_multi(root=self.config['data_path'])
            if self.config['propertyLevel'] == 'molecule': # naive, only with solvation property
               if self.config['gnn_type'] == 'dmpnn':
                  dataset = GraphDataset_dmpnn_mol(root=self.config['data_path'])
               else:
                  dataset = GraphDataset_single(root=self.config['data_path'])
         elif self.config['dataset'] in ['qm9/nmr/carbon', 'qm9/nmr/carbon/smaller', 'qm9/nmr/hydrogen', 'nmr/hydrogen', 'nmr/carbon', 'protein/nmr', 'protein/nmr/alphaFold']:
            if self.config['gnn_type'] == 'dmpnn':
               dataset = GraphDataset_dmpnn_atom(root=self.config['data_path'])
            else:
               if 'Residue' in self.config['style']:
                  dataset = GraphDataset_atom_residue(root=self.config['data_path'])
               else:
                  dataset = GraphDataset_atom(root=self.config['data_path'])
         else: # for typical other datasets 
            if self.config['gnn_type'] == 'dmpnn':
               if self.config['propertyLevel'] == 'molecule': # naive, only with solvation property
                  dataset = GraphDataset_dmpnn_mol(root=self.config['data_path'])
               if self.config['propertyLevel'] in ['atom', 'atomMol']: #  either for atom property only or atom/mol property 
                  dataset = GraphDataset_dmpnn_atom(root=self.config['data_path'])
            else:
               dataset = GraphDataset(root=self.config['data_path'])
      
      num_features = dataset.num_features
      if self.config['model'] not in ['physnet']:
         num_bond_features = dataset[0]['edge_attr'].shape[1]
      else:
         num_bond_features = 0
          
      my_split_ratio = [self.train_size, self.val_size]  #my dataseet
      if self.config['explicit_split']:
         #mean, std = dataset[self.train_index].data.mol_y.mean(), dataset[self.train_index].data.mol_y.std()
         #dataset.data.mol_y = (dataset.data.mol_y - mean) / std
         test_dataset = dataset[self.test_index]
         val_dataset = dataset[self.valid_index]
         train_dataset = dataset[self.train_index]

      if self.config['sample']:
         random.seed(self.data_seed)
         if len(dataset) > self.train_size + self.val_size + self.test_size:
            dataset = dataset.index_select(random.sample(range(len(dataset)), self.train_size + self.val_size + self.test_size))
            train_dataset = dataset[:self.train_size]
            val_dataset = dataset[self.train_size:self.train_size + self.val_size]
            test_dataset = dataset[self.train_size + self.val_size:]
         else:
            assert len(dataset) == self.train_size + self.val_size + self.test_size

            if self.config['fix_test']: # fixed test and do CV
               rest_dataset = dataset[:my_split_ratio[0]+my_split_ratio[1]]
               train_dataset = rest_dataset.index_select(random.sample(range(my_split_ratio[0]+my_split_ratio[1]), my_split_ratio[0]))
               val_dataset = rest_dataset[list(set(rest_dataset.indices()) - set(train_dataset.indices()))]
               test_dataset = dataset[my_split_ratio[0]+my_split_ratio[1]:]
               
            elif self.config['vary_train_only']: # varying train size and random split for train/valid/test
               rest_dataset = dataset.index_select(random.sample(range(len(dataset)), my_split_ratio[0]+my_split_ratio[1]))
               test_dataset = dataset[list(set(dataset.indices()) - set(rest_dataset.indices()))]
               train_dataset = rest_dataset[:my_split_ratio[0]]
               val_dataset = rest_dataset[my_split_ratio[0]:my_split_ratio[0]+my_split_ratio[1]]
               train_dataset = train_dataset.index_select(random.sample(range(my_split_ratio[0]), self.config['sample_size']))
            else: # random split for train/valid/test
               rest_dataset = dataset.index_select(random.sample(range(len(dataset)), my_split_ratio[0]+my_split_ratio[1]))
               test_dataset = dataset[list(set(dataset.indices()) - set(rest_dataset.indices()))]
               train_dataset = rest_dataset[:my_split_ratio[0]]
               val_dataset = rest_dataset[my_split_ratio[0]:my_split_ratio[0]+my_split_ratio[1]]
      
      else: # not sampling
         if not self.config['vary_train_only']:
            test_dataset = dataset[my_split_ratio[0]+my_split_ratio[1]:]
            rest_dataset = dataset[:my_split_ratio[0]+my_split_ratio[1]]
            train_dataset, val_dataset = rest_dataset[:my_split_ratio[0]], rest_dataset[my_split_ratio[0]:]
         else:
            random.seed(self.data_seed)
            test_dataset = dataset[my_split_ratio[0]+my_split_ratio[1]:]
            rest_dataset = dataset[:my_split_ratio[0]+my_split_ratio[1]]
            train_dataset, val_dataset = rest_dataset[:my_split_ratio[0]], rest_dataset[my_split_ratio[0]:]
            train_dataset = train_dataset.index_select(random.sample(range(my_split_ratio[0]), self.config['sample_size']))

      if self.config['model'] in ['physnet']:
         if self.config['explicit_split']: # when using fixed data split 
            test_loader = torch.utils.data.DataLoader(dataset[self.test_index], batch_size=self.config['batch_size'], num_workers=0, collate_fn=collate_fn)
            val_loader = torch.utils.data.DataLoader(dataset[self.valid_index], batch_size=self.config['batch_size'], num_workers=0, collate_fn=collate_fn)
            train_loader = torch.utils.data.DataLoader(dataset[self.train_index], batch_size=self.config['batch_size'], num_workers=0, shuffle=True, collate_fn=collate_fn)
         else:
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config['batch_size'], num_workers=0, collate_fn=collate_fn)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config['batch_size'], num_workers=0, collate_fn=collate_fn)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['batch_size'], num_workers=0, shuffle=True, collate_fn=collate_fn)
      else:
         if self.config['gnn_type'] == 'dmpnn':
            test_loader = DataLoader_dmpnn(test_dataset, batch_size=self.config['batch_size'], num_workers=0)
            val_loader = DataLoader_dmpnn(val_dataset, batch_size=self.config['batch_size'], num_workers=0)
            train_loader = DataLoader_dmpnn(train_dataset, batch_size=self.config['batch_size'], num_workers=0, shuffle=True)
         else:
            test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], num_workers=0)
            train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], num_workers=0, shuffle=True)
      
      return train_loader, val_loader, test_loader, num_features, num_bond_features




