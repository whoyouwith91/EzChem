import argparse, sys
from DataPrepareUtils import my_pre_transform
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)
from tqdm import tqdm
from DummyIMDataset import DummyIMDataset
from utils_functions import collate_fn
from utils_functions import add_parser_arguments
from PhysDimeNet import PhysDimeNet
from utils_functions import floating_type
from Optimizers import EmaAmsGrad
import torch
import torch.nn.functional as F
import numpy as np
from prettytable import PrettyTable
from helper import *


net_kwargs = {
        'n_atom_embedding': 95,
        'n_feature': 160,
        'n_output': 1,
        'n_dime_before_residual': 1,
        'n_dime_after_residual': 2,
        'n_output_dense': 3,
        'n_phys_atomic_res': 1,
        'n_phys_interaction_res': 1,
        'n_phys_output_res': 1,
        'n_bi_linear': 8,
        'nh_lambda': 0.01,
        'normalize': True,
        'debug_mode': False,
        'activations': 'ssp ssp ssp',
        'shared_normalize_param': True,
        'restrain_non_bond_pred': True,
        'expansion_fn': '(P_BN,P-noOut_BN):gaussian_64_10.0',
        'modules': 'P-noOut P-noOut P',
        'bonding_type': 'BN BN BN',
        'uncertainty_modify': 'none',
        'coulomb_charge_correct': False,
        'target_names': ['NMR_chemical_shift'],
        'requires_embedding': False,
        'scheduler': 'NoamLR',
        'patience_epochs': 10,
        'epochs': 300,
        'warmup_epochs': 2,
        'const_lr': 0.001, 
        'init_lr': 0.0001,
        'max_lr': 0.001,
        'final_lr': 0.0001,
        'batch_size': 64,
        'model': 'physnet',
        'early_stopping': False
        }

device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
class QM9(InMemoryDataset):
    #raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
    #           'molnet_publish/qm9.zip')
    #raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    #processed_url = 'https://pytorch-geometric.com/datasets/qm9_v2.zip'

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(QM9, self).__init__(root, transform, pre_transform, pre_filter)
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

class NMR(InMemoryDataset):
    #raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
    #           'molnet_publish/qm9.zip')
    #raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    #processed_url = 'https://pytorch-geometric.com/datasets/qm9_v2.zip'

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(NMR, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['nmr.sdf', 'nmrshiftdb.pickle']

    @property
    def processed_file_names(self):
        return 'data_nmr.pt'

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
                    data = Data(R=pos, Z=z, y=y, mask=mask, N=N_, idx=int(name))
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

def train(model, optimizer, dataloader, config):
    '''
    Define loss and backpropagation
    '''
    model.train()
    all_loss = 0
    all_atoms = 0

    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        y0, _ = model(data)
        
        if config['mask']:
            loss = F.mse_loss(data.y[data.mask>0], y0.view(-1)[data.mask>0])
            all_atoms += data.mask.sum()
            all_loss += loss.item()*data.mask.sum()
        else:
            if config['dataset'] == 'sol_calc/ALL':
                loss = F.mse_loss(data.CalcSol, y0.view(-1))*data.E.size()[0]
                all_loss += loss.item()
            else:
                loss = F.mse_loss(data.y, y0.view(-1))
                all_atoms += data.N.sum()
                all_loss += loss.item()*data.N.sum()
        loss.backward()
        optimizer.step()
        if config['scheduler'] == 'NoamLR':
            scheduler.step()
    if config['dataset'] == 'sol_calc/ALL': 
        return np.sqrt((all_loss / len(dataloader.dataset)))
    else:
        return (all_loss / all_atoms.item()).sqrt()

def test(model, dataloader, config):
    '''
    taskType
    '''
    model.eval()
    error = 0
    total_N = 0
    with torch.no_grad():
       for data in dataloader:
           data = data.to(device)
           if config['mask']:
               error += F.l1_loss(model(data)[0].view(-1)[data.mask>0], data.y[data.mask>0])*data.mask.sum().item()
               total_N += data.mask.sum().item()
           else:
               if config['dataset'] == 'sol_calc/ALL':
                   error += F.l1_loss(model(data)[0].view(-1), data.CalcSol)*data.E.size()[0]
               else:
                   total_N += data.N.sum().item()
                   error += F.l1_loss(model(data)[0].view(-1), data.y)*data.N.sum().item()
       if config['dataset'] == 'sol_calc/ALL':
           return error.item() / len(dataloader.dataset) # MAE
       else:
           return error.item() / total_N # MAE

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--mask', action='store_true')
parser.add_argument('--finetuning', action='store_true')
parser.add_argument('--running_path', type=str)
parser.add_argument('--action', type=str)
parser.add_argument('--optimizer', type=str)

args = parser.parse_args()
this_dic = net_kwargs
this_dic['dataset'] = args.dataset
this_dic['action'] = args.action
this_dic['mask'] = args.mask
this_dic['running_path'] = args.running_path
if args.dataset == 'qm9':
    data = QM9(root='/scratch/dz1061/gcn/chemGraph/data/qm9', pre_transform=my_pre_transform)
    dataset = DummyIMDataset(root="/scratch/dz1061/gcn/chemGraph/data/qm9", dataset_name='data_nmr.pt')
elif args.dataset == 'nmr/carbon':
    data = NMR(root='/scratch/dz1061/gcn/chemGraph/data/nmr/carbon', pre_transform=my_pre_transform)
    dataset = DummyIMDataset(root="/scratch/dz1061/gcn/chemGraph/data/nmr/carbon", dataset_name='data_nmr.pt')
elif args.dataset == 'nmr/hydrogen':
    data = NMR(root='/scratch/dz1061/gcn/chemGraph/data/nmr/hydrogen', pre_transform=my_pre_transform)
    dataset = DummyIMDataset(root="/scratch/dz1061/gcn/chemGraph/data/nmr/hydrogen", dataset_name='data_nmr.pt')
elif args.dataset == 'sol_calc/ALL':
    #data = NMR(root='/scratch/dz1061/gcn/chemGraph/data/{}'.format(args.dataset), pre_transform=my_pre_transform)
    dataset = DummyIMDataset(root="/scratch/dz1061/gcn/chemGraph/data/{}/graphs/base/physnet".format(args.dataset), dataset_name='data_nmr.pt')
else:
    pass

if args.dataset == 'qm9':
    train_size, valid_size = 105972, 11775
elif args.dataset == 'nmr/carbon':
    train_size, valid_size = 20000, 1487
elif args.dataset == 'nmr/hydrogen':
    train_size, valid_size = 9194, 1000
elif args.dataset == 'sol_calc/ALL':
    train_size, valid_size = 452409, 56552
this_dic['train_size'] = train_size
with open(os.path.join(this_dic['running_path'], 'config.json'), 'w') as f:
    json.dump(this_dic, f, indent=2)

train_loader = torch.utils.data.DataLoader(dataset[:train_size], batch_size=this_dic['batch_size'], collate_fn=collate_fn, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(dataset[train_size:train_size+valid_size], batch_size=this_dic['batch_size'], collate_fn=collate_fn, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset[train_size+valid_size:], batch_size=this_dic['batch_size'], collate_fn=collate_fn, num_workers=0)

model = PhysDimeNet(**this_dic)
net = model.type(floating_type).to(device)
if args.finetuning:
    net.load_state_dict(torch.load('/scratch/dz1061/gcn/chemGraph/results/qm9/nmr/allAtoms/physnet/best_model/model_best.pt'))

if args.optimizer == 'EMA':
    shadow_net = PhysDimeNet(**this_dic)
    shadow_net = shadow_net.to(device)
    shadow_net = shadow_net.type(floating_type)
    shadow_net.load_state_dict(net.state_dict())
    optimizer = EmaAmsGrad(net.parameters(), shadow_net, lr=this_dic['const_lr'], ema=0.999)
else:
    optimizer = torch.optim.Adam(net.parameters(), lr=this_dic['const_lr'])
scheduler = build_lr_scheduler(optimizer, this_dic)

x = PrettyTable()
x.field_names = ['Epoch', 'LR', 'Train MSE', 'Valid MAE', 'Test MAE']
best_val_error = float("inf")
for epoch in range(300):
    if this_dic['scheduler'] == 'const': lr = this_dic['lr']
    elif this_dic['scheduler'] == 'decay': lr = scheduler.optimizer.param_groups[0]['lr']
    else: lr = scheduler.get_lr()

    loss = train(net, optimizer, train_loader, this_dic)
    if args.optimizer == 'EMA':
        shadow_net = optimizer.shadow_model
        val_error = test(shadow_net, val_loader, this_dic)
        test_error = test(shadow_net, test_loader, this_dic)
    else:
        val_error = test(net, val_loader, this_dic)
        test_error = test(net, test_loader, this_dic)
    if this_dic['scheduler'] == 'decay':
        scheduler.step(val_error)

    if args.optimizer == 'EMA':
        best_val_error = saveModel(this_dic, epoch, shadow_net, best_val_error, val_error)
    else:
        best_val_error = saveModel(this_dic, epoch, net, best_val_error, val_error)
    x.add_row([str(epoch), lr, loss.item(), val_error, test_error])
    saveToResultsFile(x, this_dic, name='data.txt')
    #sys.stdout.flush()
torch.save(net.state_dict(), os.path.join(this_dic['running_path'], 'trained_model', 'model_last.pt'))
