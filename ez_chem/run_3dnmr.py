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
import torch
import torch.nn.functional as F

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
        'action': 'NMR',
        'target_names': ['NMR_chemical_shift'],
        'requires_embedding': False
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

def train(model, optimizer, dataloader):
    '''
    Define loss and backpropagation
    '''
    model.train()
    all_loss = 0
    all_atoms = 0

    for data in tqdm.tqdm(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        y0, _ = model(data)

        loss = F.mse_loss(data.y, y0.float().view(-1))
        all_atoms += data.N.sum()
        all_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    return all_loss / all_atoms.item()

def test(model, dataloader):
    '''
    taskType
    '''
    model.eval()
    error = 0
    total_N = 0
    with torch.no_grad():
       for data in dataloader:
           data = data.to(device)
           total_N += data.N.sum().item()
           error += F.l1_loss(model(data)[0].view(-1), data.y)*data.N.sum().item()
       return error.item() / total_N # MAE

data = QM9(root='/scratch/dz1061/gcn/chemGraph/data/qm9', pre_transform=my_pre_transform)
qm9 = DummyIMDataset(root="/scratch/dz1061/gcn/chemGraph/data/qm9", dataset_name='data_nmr.pt')

train_size, valid_size = 105972, 11775
train_loader = torch.utils.data.DataLoader(qm9[:train_size], batch_size=64, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(qm9[train_size:train_size+valid_size], batch_size=64, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(qm9[train_size+valid_size:], batch_size=64, collate_fn=collate_fn)

model = PhysDimeNet(**net_kwargs)
net = model.type(floating_type).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for _ in range(100):
    loss = train(net, optimizer, train_loader, config)
    val_error = test(net, val_loader)
    test_error = test(net, test_loader)
    print(loss, val_error, test_error)
    sys.stdout.flush()

    #print(loss)
