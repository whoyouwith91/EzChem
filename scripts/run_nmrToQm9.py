import argparse, time, os 
import torch
from helper import *
from data import *
from trainer import *
from models import *
from utils_functions import floating_type
from Optimizers import EmaAmsGrad
from kwargs import *


def train_op(model, loader, config, scheduler):
    model.train()
    for data in loader:
        data = data.to('cuda')
        output = phsnet_model(data)
        optimizer.zero_grad()
        
        pred = model(output['vi'].float(), output['atom_mol_batch'])
        loss = get_loss_fn(config['loss'])(pred.reshape(-1), data.E.float())
        
        loss.backward()
        optimizer.step()
        if config['scheduler'] in ['NoamLR', 'step']:
            scheduler.step()
    return loss.item()
    #print(loss.item())

def test_op(model, loader, config):
    with torch.no_grad():
        model.eval()
        all_loss = 0
        for data in loader:
            data = data.to('cuda')
            output = phsnet_model(data)
            pred = model(output['vi'].float(), output['atom_mol_batch'])
            loss = get_metrics_fn(config['metrics'])(pred.reshape(-1), data.E.float())
            all_loss += loss.item()*100
        return all_loss/len(loader.dataset)

config = loadConfig('/scratch/dz1061/gcn/chemGraph/results/qm9/nmr/allAtoms/physnet/physnet/newModel/Exp_layer3_dim160_lr0.001_bs100_EMA_step_l1_relu_energyShift/seed_31/')
config_all = {**config['data_config'], **config['model_config'], **config['train_config'], **physnet_kwargs}

config_all['normalize'] = True
config_all['get_atom_embedding'] = True

config_all['n_feature'] = config_all['emb_dim']
config_all['requires_atom_prop'] = True
config_all['action'] = 'nmr'
energy_shift = torch.tensor([67.2858])
energy_scale = torch.tensor([85.8406])

config_all['energy_shift'] = energy_shift 
config_all['energy_scale'] = energy_scale

model = get_model(config_all)

config_all['data_path'] = '/scratch/dz1061/gcn/chemGraph/data/qm9/u0/graphs/base/physnet'
config_all['explicit_split'] = True
config_all['normalize'] = False
config_all['scheduler'] = 'NoamLR'
config_all['warmup_epochs'] = 2
config_all['init_lr'] = 0.0001
config_all['max_lr'] = 0.001
config_all['final_lr'] = 0.0001
config_all['patience_epochs'] = 2
config_all['decay_factor'] = 0.9
loader = get_data_loader(config_all)

state_dic = torch.load('/scratch/dz1061/gcn/chemGraph/results/qm9/nmr/allAtoms//physnet/physnet/newModel/Exp_layer3_dim160_lr0.001_bs100_EMA_step_l1_relu_energyShift/seed_31/best_model/model_best.pt')
model.load_state_dict(state_dic)

phsnet_model = model.type(floating_type).to('cuda')
phsnet_model.eval()


dnn_model = DNN(in_size=160, hidden_size=160, out_size=1, n_hidden=3, activation=F.leaky_relu, bn=True)
dnn_model.to('cuda')
optimizer = torch.optim.Adam(dnn_model.parameters(), lr=0.001)
scheduler = build_lr_scheduler(optimizer, config_all)

lr = 0.001
config_all['loss'] = 'l2'
config_all['running_path'] = '/scratch/dz1061/gcn/chemGraph/results/qm9/u0/physnet/physnet/TransferLearningFromqm9/nmr/allAtoms/DNN_NoamLR/seed_31'
if not os.path.exists(os.path.join(config_all['running_path'], 'trained_model/')):
    os.makedirs(os.path.join(config_all['running_path'], 'trained_model/'))
if not os.path.exists(os.path.join(config_all['running_path'], 'best_model/')):
    os.makedirs(os.path.join(config_all['running_path'], 'best_model/'))
results = createResultsFile(config_all) # create pretty table

for epoch in range(1000):
    if config_all['scheduler'] == 'const': lr = args.lr
    elif config_all['scheduler'] in ['NoamLR', 'step']: lr = scheduler.get_lr()[0]
    else:lr = scheduler.optimizer.param_groups[0]['lr'] # decaying on val error

    time_tic = time.time() # ending time 
    loss = train_op(dnn_model, loader.train_loader, config_all, scheduler)
    time_toc = time.time() # ending time 
    val_error = test_op(dnn_model, loader.val_loader, config_all)
    test_error = test_op(dnn_model, loader.test_loader, config_all)
    contents = [epoch, round(time_toc-time_tic, 2), round(lr,7), round(loss,3),  \
                round(val_error,3), round(test_error,3), round(param_norm(dnn_model),2), round(grad_norm(dnn_model),2)]
    results.add_row(contents)
    saveToResultsFile(results, config_all, name='data.txt')

