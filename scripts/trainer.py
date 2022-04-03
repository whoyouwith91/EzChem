import sys
from helper import *
from data import *
from models import *
from utils_functions import floating_type

def train_model(config):
    if config['uncertainty']:
        return train_uncertainty
    else:
        return train

def test_model(config):
    if config['uncertainty']:
        return test_uncertainty
    else:
        return test

def train(model, optimizer, dataloader, config, scheduler=None):
    '''
    Define loss and backpropagation
    '''
    model.train()
    all_loss = 0
    if config['propertyLevel'] == 'atom':
        all_atoms = 0
    for data in dataloader:
        data = data.to(config['device'])
        #print(data.x.shape)
        #sys.stdout.flush()
        optimizer.zero_grad()
        y = model(data) # y contains different outputs depending on the # of tasks
        
        if config['dataset'] in ['solNMR', 'solALogP', 'qm9/nmr/allAtoms', 'sol_calc/smaller', 'sol_calc/all']:
            if config['propertyLevel'] == 'molecule': # single task on regression
                assert config['taskType'] == 'single'
                loss = get_loss_fn(config['loss'])(y[1], data.y)
                if config['gnn_type'] == 'dmpnn':
                    all_loss += loss.item() * data.y.shape[0]
                else:
                    all_loss += loss.item() * data.num_graphs

            elif config['propertyLevel'] == 'atom': # 
                assert config['taskType'] == 'single'
                loss = get_loss_fn(config['loss'])(data.atom_y, y[0])
                all_loss += loss.item() * data.N.sum()
                all_atoms += data.N.sum()
            elif config['propertyLevel'] == 'multiMol':
                assert config['taskType'] == 'multi'
                loss = get_loss_fn(config['loss'])(y[0], data.mol_gas) + \
                       get_loss_fn(config['loss'])(y[1], data.mol_wat) + \
                       get_loss_fn(config['loss'])(y[2], data.mol_oct) + \
                       get_loss_fn(config['loss'])(y[3], data.mol_sol_wat) + \
                       get_loss_fn(config['loss'])(y[4], data.mol_sol_oct) + \
                       get_loss_fn(config['loss'])(y[5], data.mol_logp)
                all_loss += loss * data.num_graphs
            elif config['propertyLevel'] == 'atomMultiMol':
                assert config['taskType'] == 'multi'
                loss = get_loss_fn(config['loss'])(y[0], data.atom_y) + \
                       get_loss_fn(config['loss'])(y[1], data.mol_gas) + \
                       get_loss_fn(config['loss'])(y[2], data.mol_wat) + \
                       get_loss_fn(config['loss'])(y[3], data.mol_oct) + \
                       get_loss_fn(config['loss'])(y[4], data.mol_sol_wat) + \
                       get_loss_fn(config['loss'])(y[5], data.mol_sol_oct) + \
                       get_loss_fn(config['loss'])(y[6], data.mol_logp)
                all_loss += loss * data.num_graphs
            elif config['propertyLevel'] == 'atomMol':
                assert config['taskType'] == 'multi'
                loss = get_loss_fn(config['loss'])(y[0], data.atom_y) + get_loss_fn(config['loss'])(y[1], data.mol_sol_wat)
            else:
                 raise "LossError"

        elif config['dataset'] in ['nmr/carbon', 'nmr/hydrogen', 'qm9/nmr/carbon', 'qm9/nmr/hydrogen', 'frag14/nmr/carbon', 'frag14/nmr/hydrogen', 'protein/nmr', 'protein/nmr/alphaFold']:
            assert config['propertyLevel'] == 'atom'
            if config['model'] == 'physnet':
                loss = get_loss_fn(config['loss'])(data.atom_y[data.mask>0], y['atom_prop'].float().view(-1)[data.mask>0])
            else:
                loss = get_loss_fn(config['loss'])(data.atom_y[data.mask>0], y[0].view(-1)[data.mask>0])
            all_atoms += data.mask.sum() # with mask information
            all_loss += loss.item()*data.mask.sum()
        
        elif config['dataset'] == 'pka/chembl':
            if config['propertyLevel'] == 'atom': 
                loss = get_loss_fn(config['loss'])(data.mol_y, y[0].view(-1)[data.mask>0])
            elif config['propertyLevel'] == 'molecule':
                loss = get_loss_fn(config['loss'])(data.mol_y, y[1])
            else:
                pass
            all_loss += loss.item()*data.mask.sum()
        
        else: 
            if config['propertyLevel'] == 'molecule': # for single task, like exp solvation, solubility, ect
                assert config['taskType'] == 'single'
                loss = get_loss_fn(config['loss'])(y[1], data.mol_y)
                if config['gnn_type'] == 'dmpnn':
                    all_loss += loss.item() * data.mol_y.shape[0]
                else:
                    all_loss += loss.item() * data.num_graphs
            elif config['propertyLevel'] == 'atom': # Exp nmr/carbon, nmr/hydrogen
                assert config['taskType'] == 'single'
                loss = get_loss_fn(config['loss'])(data.atom_y, y[0], data.mask)
                all_atoms += data.mask.sum()
                all_loss += loss.item() * data.mask.sum()
            else:
                raise "LossError"
 
        loss.backward()
        if config['clip']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        
    if config['scheduler'] == 'NoamLR':
        scheduler.step()
    if config['optimizer'] in ['SWA']:
        optimizer.swap_swa_sgd()
    
    if config['metrics'] == 'l2':
        if config['test_level'] == 'molecule':
            return np.sqrt(all_loss / len(dataloader.dataset)) # RMSE for mol level properties.
        if config['test_level'] == 'atom':
            return np.sqrt(all_loss.item() / all_atoms.item()) # RMSE for mol level properties.
    
    if config['metrics'] == 'l1':
        if config['test_level'] == 'atom':
            return all_loss.item() / all_atoms.item() # MAE
        elif config['propertyLevel'] in ['multiMol', 'atomMol']:
            return all_loss.item() / len(dataloader.dataset) * 23 # ev --> kcal/mol
        else:
            assert config['test_level'] == 'molecule'
            return all_loss / len(dataloader.dataset) 
    
    if config['propertyLevel'] in ['atomMol', 'multiMol', 'atomMultiMol']: # mixed tasks of single/multi
        return loss.item()

def test(model, dataloader, config, onData=''):
    '''
    Test model's performance
    '''
    model.eval()
    error = 0
    if config['propertyLevel'] == 'multiMol':
        from collections import defaultdict
        error = defaultdict(float)
    if config['test_level'] == 'atom':
        total_N = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(config['device'])
            y = model(data)

            if config['dataset'] in ['solNMR', 'solALogP', 'qm9/nmr/allAtoms', 'sol_calc/smaller', 'sol_calc/all']:
                if config['test_level'] == 'molecule': # the metrics on single task, currently on solvation energy only
                    if config['propertyLevel'] == 'multiMol':
                        #pred = y[3] # only testing solvation energy
                        error['gas'] += get_metrics_fn(config['metrics'])(y[0], data.mol_gas) * data.num_graphs
                        error['wat'] += get_metrics_fn(config['metrics'])(y[1], data.mol_wat) * data.num_graphs
                        error['oct'] += get_metrics_fn(config['metrics'])(y[2], data.mol_oct) * data.num_graphs
                        error['sol_wat'] += get_metrics_fn(config['metrics'])(y[3], data.mol_sol_wat) * data.num_graphs
                        error['sol_oct'] += get_metrics_fn(config['metrics'])(y[4], data.mol_sol_oct) * data.num_graphs
                        error['logp'] += get_metrics_fn(config['metrics'])(y[5], data.mol_logp) * data.num_graphs
                        #error += get_metrics_fn(config['metrics'])(pred, data.mol_sol_wat) * data.num_graphs
                    elif config['propertyLevel'] in ['molecule', 'atomMol'] and config['gnn_type'] != 'dmpnn':
                        pred = y[1]
                        error += get_metrics_fn(config['metrics'])(pred, data.mol_y) * data.num_graphs
                    else:
                        pred = y[1]
                        if config['gnn_type'] == 'dmpnn':
                            error += get_metrics_fn(config['metrics'])(pred, data.mol_y) * data.mol_y.shape[0]
                        else:
                            error += get_metrics_fn(config['metrics'])(pred, data.mol_y) * data.num_graphs
                elif config['test_level'] == 'atom': #
                    total_N += data.N.sum().item()
                    error += get_metrics_fn(config['metrics'])(y[0], data.atom_y)*data.N.sum().item()
                else:
                    raise "MetricsError"

            elif config['dataset'] in ['nmr/carbon', 'nmr/hydrogen', 'qm9/nmr/carbon', 'qm9/nmr/hydrogen', 'frag14/nmr/carbon', 'frag14/nmr/hydrogen', 'protein/nmr', 'protein/nmr/alphaFold']:
                assert config['test_level'] == 'atom'
                if config['model'] == 'physnet':
                    error += get_metrics_fn(config['metrics'])(data.atom_y[data.mask>0], y['atom_prop'].float().view(-1)[data.mask>0])*data.mask.sum().item()
                else:
                    error += get_metrics_fn(config['metrics'])(data.atom_y[data.mask>0], y[0].view(-1)[data.mask>0])*data.mask.sum().item()
                total_N += data.mask.sum().item()
            
            elif config['dataset'] == 'pka/chembl':
                if config['propertyLevel'] == 'atom': 
                    error += get_metrics_fn(config['metrics'])(data.mol_y, y[0].view(-1)[data.mask>0]) * data.num_graphs
                elif config['propertyLevel'] == 'molecule':
                    error += get_metrics_fn(config['metrics'])(data.mol_y, y[1]) * data.num_graphs
                else:
                    pass 

            else: 
                if config['test_level'] == 'molecule':
                    if config['gnn_type'] == 'dmpnn':
                        error += get_metrics_fn(config['metrics'])(y[1], data.mol_y) * data.mol_y.shape[0]
                    else:
                        error += get_metrics_fn(config['metrics'])(y[1], data.mol_y) * data.num_graphs
                elif config['test_level'] == 'atom': # nmr/carbon hydrogen
                    total_N += data.mask.sum().item()
                    error += get_metrics_fn(config['metrics'])(y[0][data.mask>0].reshape(-1,1), data.atom_y[data.mask>0].reshape(-1,1))*data.mask.sum().item()
                else:
                    raise "MetricsError"

        if config['metrics'] == 'l2':
            if config['test_level'] == 'molecule':
                return np.sqrt(error.item() / len(dataloader.dataset)) # RMSE for mol level properties.
            if config['test_level'] == 'atom':
                return np.sqrt(error.item() / total_N) # RMSE for mol level properties.
        if config['metrics'] == 'l1':
            if config['test_level'] == 'atom':
                return error.item() / total_N # MAE
            elif config['propertyLevel'] in ['multiMol', 'atomMol']:
                if onData == 'test':
                    return error['sol_wat'].item() / len(dataloader.dataset) * 23 # ev --> kcal/mol
                else:
                    return sum(error.values()).item() / len(dataloader.dataset) * 23 # ev --> kcal/mol
            else:
                return error.item() / len(dataloader.dataset) 
        if config['metrics'] == 'class':
            return get_metrics_fn(config['metrics'])(np.hstack(labels), np.hstack(preds))

def train_uncertainty(model, optimizer, dataloader, config, scheduler=None): 
    model.train()
    all_loss = 0
    for data in dataloader:
        data = data.to(config['device'])
        optimizer.zero_grad()

        if config['uncertaintyMode'] == 'aleatoric': # data uncertainty
            _, mean, log_var = model(data)
            loss = get_loss_fn(config['loss'])(data.mol_y, mean, log_var)
            #logvars.extend(log_var.item())

        if config['uncertaintyMode'] == 'epistemic': # model uncertainty
            mean = model(data)
            loss = get_loss_fn(config['loss'])(data.y, mean)
        
        if config['uncertaintyMode'] == 'evidence':
            preds = model(data)
            means =  preds[:, [j for j in range(len(preds[0])) if j % 4 == 0]].view(-1,)
            lambdas =  preds[:, [j for j in range(len(preds[0])) if j % 4 == 1]].view(-1,)
            alphas =  preds[:, [j for j in range(len(preds[0])) if j % 4 == 2]].view(-1,)
            betas  =  preds[:, [j for j in range(len(preds[0])) if j % 4 == 3]].view(-1,)
            loss = get_loss_fn(config['loss'])(means, lambdas, alphas, betas, data.mol_y)
        #y.extend(data.y.item())
        #means.extend(mean.item())
        loss.backward()
        
        if config['model'].endswith('dropout'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if config['gnn_type'] == 'dmpnn':
            all_loss += loss.item() * data.mol_y.shape[0]
        else:
            all_loss += loss.item() * data.num_graphs

    if config['uncertaintyMode'] == 'aleatoric': # TODO
        return np.sqrt(all_loss / len(dataloader.dataset)), None, None
    if config['uncertaintyMode'] == 'epistemic': # TODO
        return loss_all/num 

def test_uncertainty(model, dataloader, config, onData=''):
    '''
   
    '''
    model.eval()
    error = 0
    #if config['uncertaintyMode'] == 'evidence':
    #    preds = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(config['device'])
            if config['uncertaintyMode'] == 'aleatoric':
                _, mean, log_var = model(data)
                #print(mean, log_var)
                #sys.stdout.flush()

                if config['gnn_type'] == 'dmpnn':
                    error += get_metrics_fn(config['metrics'])(data.mol_y, mean) * data.mol_y.shape[0]
                else:
                    error += get_metrics_fn(config['metrics'])(data.mol_y, mean) * data.num_graphs
                #var_.extend(log_var.item())
                #mean_.extend(mean.item())
                #MC_samples.append([mean_, var_])
                #mean_batch, var_batch, _ = model(data.to(device))
            if config['uncertaintyMode'] == 'epistemic':
                pass
                #MC_samples.append([mean_])
            if config['uncertaintyMode'] == 'evidence':
                preds = model(data)
                means =  preds[:, [j for j in range(len(preds[0])) if j % 4 == 0]].view(-1,)
                if config['gnn_type'] == 'dmpnn':
                    error += get_metrics_fn(config['metrics'])(data.mol_y, means) * data.mol_y.shape[0]
                else:
                    error += get_metrics_fn(config['metrics'])(data.mol_y, means) * data.num_graphs
        
        if config['uncertaintyMode'] == 'aleatoric':
            return np.sqrt(error.item() / len(dataloader.dataset))

        if config['uncertaintyMode'] == 'evidence':
            return np.sqrt(error.item() / len(dataloader.dataset))

