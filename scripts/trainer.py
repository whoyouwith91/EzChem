import sys
from helper import *
from data import *
from models import *
from utils_functions import floating_type

def train_model(config):
    if config['model'] in ['physnet']:
        return train_physnet
    else:
        return train

def test_model(config):
    if config['model'] in ['physnet']:
        return test_physnet
    else:
        return test

def cv_train(config, table):
    results = []

    for seed in [1, 13, 31]:
        #config['data_path'] = os.path.join(config['cv_path'], 'cv_'+str(i)+'/')
        #print(config)
        #print(config['data_path'])   
        set_seed(seed)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        config['device'] = device

        loader = get_data_loader(config)
        train_loader, val_loader, _, std, num_features, num_bond_features, num_i_2 = loader.train_loader, loader.val_loader, loader.test_loader, loader.std, loader.num_features, loader.num_bond_features, loader.num_i_2
        config['num_features'], config['num_bond_features'], config['num_i_2'], config['std'] = int(num_features), num_bond_features, num_i_2, std 

        model = get_model(config)
        args = objectview(config)
        model_ = model.to(device)
        #num_params = param_count(model_)
        
        optimizer = get_optimizer(config['optimizer'], model_)
        if this_dic['lr_style'] == 'decay':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=5, min_lr=0.00001)
        if this_dic['uncertainty']:
            optimizer = torchcontrib.optim.SWA(optimizer)

        for _ in range(1, config['epochs']):
            val_res, tst_res = []
            loss = train(model_, optimizer, train_loader, config)
            val_error = test(model_, val_loader, config)
            tst_error = test(model_, test_loader, config)
            val_res.append(val_error)
            tst_res.append(tst_error)
        
        results.append(tst_res[val_res.index(min(val_res))])
    
    table.add_row([config['emb_dim'], config['num_layer'], config['NumOutLayers'], config['lr'], config['batch_size'], np.mean(results), np.std(results)])
    print(table)
    sys.stdout.flush()
    print('\n')
    sys.stdout.flush()
    return np.average(results)

def train(model, optimizer, dataloader, config, scheduler=None):
    '''
    Define loss and backpropagation
    '''
    model.train()
    all_loss = 0
    if config['propertyLevel'] == 'atom':
        all_atoms = 0
    if config['dataset'] == 'solEFGs':
        preds, labels = [], []
    for data in dataloader:
        data = data.to(config['device'])
        optimizer.zero_grad()
        y = model(data) # y contains different outputs depending on the # of tasks
        
        if config['dataset'] in ['solNMR', 'solALogP', 'qm9/nmr/allAtoms']:
            if config['propertyLevel'] == 'molecule': # single task on regression
                assert config['taskType'] == 'single'
                loss = get_loss_fn(config['loss'])(y[1], data.mol_sol_wat)
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
                       get_loss_fn(config['loss'])(y[4], data.mol_sol_oct)
                       #get_loss_fn(config['loss'])(y[5], data.mol_logp)
            elif config['propertyLevel'] == 'atomMultiMol':
                assert config['taskType'] == 'multi'
                loss = get_loss_fn(config['loss'])(y[0], data.atom_y) + \
                       get_loss_fn(config['loss'])(y[1], data.mol_gas) + \
                       get_loss_fn(config['loss'])(y[2], data.mol_wat) + \
                       get_loss_fn(config['loss'])(y[3], data.mol_oct) + \
                       get_loss_fn(config['loss'])(y[4], data.mol_sol_wat) + \
                       get_loss_fn(config['loss'])(y[5], data.mol_sol_oct) + \
                       get_loss_fn(config['loss'])(y[6], data.mol_logp)
            elif config['propertyLevel'] == 'atomMol':
                assert config['taskType'] == 'multi'
                loss = get_loss_fn(config['loss'])(y[0], data.atom_y) + get_loss_fn(config['loss'])(y[1], data.mol_sol_wat)
            else:
                 raise "LossError"
        
        elif config['dataset'] == 'solEFGs': # solvation for regression and EFGs labels for classification 
            if config['propertyLevel'] == 'atomMol':
                assert config['taskType'] == 'multi'
                loss = get_loss_fn(config['atom_loss'])(y[0], data.atom_y) + get_loss_fn(config['mol_loss'])(y[1], data.mol_sol_wat)
            elif config['propertyLevel'] == 'atom':
                assert config['taskType'] == 'single'
                loss = get_loss_fn(config['loss'])(y[0], data.atom_y)
                idx = F.log_softmax(model(data)[0], 1).argmax(dim=1)
                preds.append(idx.detach().data.cpu().numpy())
                labels.append(data['atom_y'].detach().data.cpu().numpy())
            else:
                raise "LossError"
        
        else: 
            if config['propertyLevel'] == 'molecule': # for single task, like exp solvation, solubility, ect
                assert config['taskType'] == 'single'
                loss = get_loss_fn(config['loss'])(y[1], data.mol_y)
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
    
    if config['propertyLevel'] == 'atom':
        if config['dataset'] == 'solEFGs':
            return loss.item(), get_metrics_fn(config['metrics'])(np.hstack(labels), np.hstack(preds))
        else:
            return all_loss.item() / all_atoms.item() # MAE
    if config['propertyLevel'] == 'molecule':
        if config['metrics'] == 'l2':
            return np.sqrt(all_loss / len(dataloader.dataset)) # RMSE
        else:
            return all_loss / len(dataloader.dataset) # MAE
    if config['propertyLevel'] in ['atomMol', 'multiMol', 'atomMultiMol']:
        return loss.item()

def test(model, dataloader, config):
    '''
    Test model's performance
    '''
    model.eval()
    error = 0
    if config['test_level'] == 'atom':
        total_N = 0
    if config['dataset'] == 'solEFGs':
        preds, labels = [], []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(config['device'])
            y = model(data)

            if config['dataset'] in ['solNMR', 'solALogP', 'qm9/nmr/allAtoms']:
                if config['test_level'] == 'molecule': # the metrics on single task, currently on solvation energy only
                    if config['propertyLevel'] == 'multiMol':
                        pred = y[3]
                    elif config['propertyLevel'] in ['molecule', 'atomMol']:
                        pred = y[1]
                    error += get_metrics_fn(config['metrics'])(pred, data.mol_sol_wat) * data.num_graphs
                elif config['test_level'] == 'atom': #
                    total_N += data.N.sum().item()
                    error += get_metrics_fn(config['metrics'])(y[0], data.atom_y)*data.N.sum().item()
                else:
                    raise "MetricsError"
            
            elif config['dataset'] == 'solEFGs':
                if config['test_level'] == 'molecule': # the metrics on single task, currently on solvation energy only
                    error += get_metrics_fn(config['metrics'])(y[1], data.mol_sol_wat) * data.num_graphs
                elif config['test_level'] == 'atom': #
                    idx = F.log_softmax(y[0], 1).argmax(dim=1)
                    #_, idx = torch.max(model(data)[0], 1)
                    preds.append(idx.detach().data.cpu().numpy())
                    labels.append(data['atom_y'].detach().data.cpu().numpy())
                else:
                    raise "MetricsError"

            else: 
                if config['test_level'] == 'molecule':
                    error += get_metrics_fn(config['metrics'])(y[1], data.mol_y) * data.num_graphs
                elif config['test_level'] == 'atom': # nmr/carbon hydrogen
                    total_N += data.mask.sum().item()
                    error += get_metrics_fn(config['metrics'])(y[0][data.mask>0].reshape(-1,1), data.atom_y[data.mask>0].reshape(-1,1))*data.mask.sum().item()
                else:
                    raise "MetricsError"

        if config['metrics'] == 'l2':
            assert config['test_level'] == 'molecule'
            return np.sqrt(error.item() / len(dataloader.dataset)) # RMSE for mol level properties.
        if config['metrics'] == 'l1':
            if config['test_level'] == 'atom':
                return error.item() / total_N # MAE
            elif config['propertyLevel'] in ['multiMol', 'atomMol']:
                return error.item() / len(dataloader.dataset) * 23 # ev --> kcal/mol
            else:
                return error.item() / len(dataloader.dataset) 
        if config['metrics'] == 'class':
            return get_metrics_fn(config['metrics'])(np.hstack(labels), np.hstack(preds))

def train_dropout(model, optimizer, dataloader, config):
    model.train()
    all_loss = 0
    y = []
    means = []
    logvars = []
    num = 0
    for data in dataloader:
        num += 1
        data = data.to(config['device'])
        optimizer.zero_grad()

        if config['uncertainty'] == 'aleatoric': # data uncertainty
            mean, log_var, _ = model(data)
            loss = get_loss_fn(config['loss'])(data.y, mean, log_var)
            logvars.extend(log_var.cpu().data.numpy())

        if config['uncertainty'] == 'epistemic': # model uncertainty
            mean = model(data)
            loss = get_loss_fn(config['loss'])(data.y, mean)
        y.extend(data.y.cpu().data.numpy())
        means.extend(mean.cpu().data.numpy())
        loss.backward()
        
        if config['model'].endswith('dropout'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        all_loss += loss.item() * data.num_graphs

    if config['uncertainty'] == 'aleatoric': # todo
        return np.sqrt(all_loss / len(dataloader.dataset)), rmse, np.array(y), np.array(means), np.array(logvars)
    if config['uncertainty'] == 'epistemic': # todo
        return loss_all/num, rmse, None, None, None

def train_unsuper(model, optimizer, dataloader, config):
    model.train()
    loss_all = 0

    for i, data in enumerate(dataloader):
        recon_batch = model(data)
        loss = get_loss_fn(config['loss'])(recon_batch.contiguous().view(-1, recon_batch.size(-1)), Variable(data.SRC[:, 1:].contiguous().view(-1,), requires_grad=False), config)
        
        loss.backward()
        print(loss.item())
        sys.stdout.flush()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        loss_all += loss.cpu().data.numpy()

    return loss_all/len(dataloader)

def test_unsuper(model, dataloader, config):
    model.eval()

    epoch_loss = 0
    with torch.no_grad():
        for data in dataloader:
            recon_batch = model(data)
            loss = get_loss_fn(config['loss'])(recon_batch.contiguous().view(-1, recon_batch.size(-1)), Variable(data.SRC[:, 1:].contiguous().view(-1,), requires_grad=False),  config)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def train_VAE(model, optimizer, dataloader, config, epoch, kl_weight=0, preSaveKLweights=None):
    '''
    Define loss and backpropagation
    '''
    model.train()
    loss_all = 0
    
    if config['anneal']:
        if config['anneal_method'] == 'warmup':
           if epoch < config['anneal_epoch']:
              #kl_weight = kl_anneal_function('linear', step=step_cnt, k1=0.1, k2=0.2, max_value=0.1, x0=100000)
              kl_weight = 0.
           else: 
              kl_weight = 1.
        elif config['anneal_method'] == 'linear':
           kl_weight = kl_weight + 0.01
        elif config['anneal_method'] == 'logistic':
           kl_weight = float(1 / (1 + 20*np.exp(- 0.1 * (epoch - config['anneal_epoch']))))
        
    else:
        kl_weight = config['kl_weight']

    for i, data in enumerate(dataloader):
        if config['anneal_method'] == 'cyclical':
           kl_weight = preSaveKLweights.pop(0)
        recon_batch, mu, logvar = model(data)
        loss, CLE, KL = get_loss_fn(config['loss'])(recon_batch.contiguous().view(-1, recon_batch.size(-1)), Variable(data.SRC[:, 1:].contiguous().view(-1,), requires_grad=False), \
                                           mu.view(-1, config['varDimen']), logvar.view(-1, config['varDimen']), kl_weight, config, saveKL=True)
        if i % 50 == 0:
           print(loss.item(), CLE.item(), KL.item(), kl_weight)
        sys.stdout.flush() 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        loss_all += loss.cpu().data.numpy()
           
    return loss_all/len(dataloader), kl_weight, preSaveKLweights 

def test_dropout(model, K_test, dataloader, config):
    '''
    Calculate RMSE in dropout net
    '''
    model.eval()
    MC_samples = []
    #y, mean_, var_ = [], [], []
    loss_all = 0
    num = 0
    with torch.no_grad():
        for _ in range(K_test):
            y, mean_, var_ = [], [], []
            for data in dataloader:
               num += 1 
               if config['model'] not in ['loopybp', 'wlkernel', 'loopybp_dropout', 'wlkernel_dropout', 'loopybp_swag', 'wlkernel_swag']:
                  data = data.to(config['device'])
               if config['model'] in ['loopybp', 'wlkernel', 'loopybp_dropout', 'wlkernel_dropout', 'loopybp_swag', 'wlkernel_swag']:
                  batchTargets = data.targets()
                  data = data.batch_graph()
                  data.y = torch.FloatTensor(batchTargets).to(config['device'])
                  data.y = data.y.view(-1)
               if config['uncertainty'] == 'aleatoric':
                  mean, log_var, regularization = model(data)
                  loss = get_loss_fn(config['loss'])(data.y, mean, log_var) + regularization
                  var_.extend(log_var.cpu().data.numpy())
                  mean_.extend(mean.cpu().data.numpy())
                  #MC_samples.append([mean_, var_])
               #mean_batch, var_batch, _ = model(data.to(device))
               if config['uncertainty'] == 'epistemic':
                  mean = model(data)
                  loss = get_loss_fn(config['loss'])(data.y, mean)
                  mean_.extend(mean.cpu().data.numpy())
                  #MC_samples.append([mean_])
               y.extend(data.y.cpu().data.numpy())
               loss_all += loss.cpu().data.numpy()
            if config['uncertainty'] == 'aleatoric':
               MC_samples.append([mean_, var_])
            if config['uncertainty'] == 'epistemic':
               MC_samples.append([mean_])
        means = np.stack([tup[0] for tup in MC_samples]).reshape(K_test, len(mean_))
    
    rmse = np.mean((np.mean(means, 0) - np.array(y).squeeze())**2.)**0.5

    return loss_all/num, rmse

def test_VAE(model, dataloader, config, kl_weight):
    model.eval()
    
    epoch_loss = 0
    with torch.no_grad():
        for data in dataloader:
            recon_batch, mu, logvar = model(data)
            loss, _, _ = get_loss_fn(config['loss'])(recon_batch.contiguous().view(-1, recon_batch.size(-1)), Variable(data.SRC[:, 1:].contiguous().view(-1,), requires_grad=False), \
                                           mu.view(-1, config['varDimen']), logvar.view(-1, config['varDimen']),kl_weight, config)
            epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)

def test_dropout_uncertainty(model, K_test, dataloader, config):
    model.eval()
    MC_samples = []
    #mean_, var_, y = [], [], []
    loss_all = 0
    num = 0
    with torch.no_grad():
       for k in range(K_test):
            set_seed(k)
            mean_, var_, y = [], [], []
            for data in dataloader:
               num += 1
               if config['model'] not in ['loopybp', 'wlkernel', 'loopybp_dropout', 'wlkernel_dropout', 'loopybp_swag', 'wlkernel_swag']:
                  data = data.to(config['device'])
               if config['model'] in ['loopybp', 'wlkernel', 'loopybp_dropout', 'wlkernel_dropout', 'loopybp_swag', 'wlkernel_swag']:
                  batchTargets = data.targets()
                  data = data.batch_graph()
                  data.y = torch.FloatTensor(batchTargets).to(config['device'])
                  data.y = data.y.view(-1)

               if config['uncertainty'] == 'aleatoric':
                  mean_batch, var_batch, regularization = model(data)
                  loss = get_loss_fn(config['loss'])(data.y, mean_batch, var_batch) + regularization
                  var_.extend(var_batch.cpu().data.numpy())
               if config['uncertainty'] == 'epistemic':
                  mean_batch = model(data)
                  loss = get_loss_fn(config['loss'])(data.y, mean_batch)
               y.extend(data.y.cpu().data.numpy())
               mean_.extend(mean_batch.cpu().data.numpy())
               loss_all += loss.cpu().data.numpy()
            if config['uncertainty'] == 'aleatoric':
               MC_samples.append([mean_, var_])
            if config['uncertainty'] == 'epistemic':
               MC_samples.append([mean_])
       if config['uncertainty'] == 'aleatoric':
           means = np.stack([tup[0] for tup in MC_samples]).reshape(K_test, len(mean_))
           logvar = np.stack([tup[1] for tup in MC_samples]).reshape(K_test, len(mean_))
           rmse = np.mean((np.mean(means, 0) - np.array(y).squeeze())**2.)**0.5
           return np.mean(means, 0), np.array(y), loss_all/num, rmse, means, logvar

       if config['uncertainty'] == 'epistemic':
           means = np.stack([tup[0] for tup in MC_samples]).reshape(K_test, len(mean_))
           rmse = np.mean((np.mean(means, 0) - np.array(y).squeeze())**2.)**0.5
           return np.mean(means, 0), np.array(y), loss_all/num, rmse

def test_swag_uncertainty(model, K_test, dataloader, config):
    model.eval()
    MC_samples = []
    with torch.no_grad():
        for _ in range(K_test):
            sample_with_cov = True
            model.sample(scale=1.0, cov=sample_with_cov)
            model.eval()
        
            y_true, y_pred, ids = [], [], []
            #torch.manual_seed(i)
            for data in dataloader:
                if config['model'] not in ['loopybp', 'wlkernel', 'loopybp_dropout', 'wlkernel_dropout', 'loopybp_swag', 'wlkernel_swag']:
                    data = data.to(config['device'])
                if config['model'] in ['loopybp', 'wlkernel', 'loopybp_dropout', 'wlkernel_dropout', 'loopybp_swag', 'wlkernel_swag']:
                    batchTargets = data.targets()
                    data = data.batch_graph()
                    data.y = torch.FloatTensor(batchTargets).to(config['device'])
                    data.y = data.y.view(-1)
                pred_batch, _ = model(data)
                y_pred.extend(pred_batch.cpu().data.numpy())
                y_true.extend(data.y.cpu().data.numpy())
                #ids.extend(data.ids.cpu().data.numpy())
            MC_samples.append([y_pred, y_true])
    return MC_samples

def train_physnet(model, optimizer, dataloader, config, scheduler=None):
    '''
    Define loss and backpropagation
    '''
    model.train()
    all_loss = 0
    all_atoms = 0

    for data in dataloader:
        data = data.to(config['device'])
        optimizer.zero_grad()
        y0 = model(data)
        
        if config['mask']: # for qm9/nmr/carbon or Exp. nmr/carbon nmr/hydrogen
            loss = get_loss_fn(config['loss'])(data.y[data.mask>0], y0['atom_prop'].float().view(-1)[data.mask>0])
            all_atoms += data.mask.sum()
            all_loss += loss.item()*data.mask.sum()
        else:
            if config['dataset'] in ['sol_calc/ALL', 'sol_calc/ALL/COMPLETE']:
                loss = get_loss_fn(config['loss'])(data.CalcSol, y0['mol_prop'].view(-1)) # data.CalcSol is only for water solvation energy
                all_loss += loss.item()*data.E.size()[0]
            elif config['dataset'] in ['solNMR']:
                if config['propertyLevel'] == 'molecule':
                    loss = get_loss_fn(config['loss'])(data.mol_y, y0['mol_prop'].view(-1)) # data.CalcSol is only for water solvation energy
                    all_loss += loss.item()*data.mol_y.size()[0]
                if config['propertyLevel'] == 'atom':
                    loss = get_loss_fn(config['loss'])(data.atom_y, y0['atom_prop'].float().view(-1))
                    all_atoms += data.N.sum().item()
                    all_loss += loss.item()*data.N.sum()
                if config['propertyLevel'] == 'atomMol':
                    loss = get_loss_fn(config['loss'])(data.mol_y, y0['mol_prop'].view(-1)) + get_loss_fn(config['loss'])(data.atom_y, y0['atom_prop'].float().view(-1))
            elif config['dataset'] in ['qm9/u0']:
                loss = get_loss_fn(config['loss'])(data.E, y0['mol_prop'].float().view(-1))
                all_loss += loss.item()*data.E.size()[0]
            else: # for qm9/nmr/allAtoms
                loss = get_loss_fn(config['loss'])(data.y, y0['atom_prop'].float().view(-1))
                all_atoms += data.N.sum()
                all_loss += loss.item()*data.N.sum()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.)
        optimizer.step()
        if config['scheduler'] in ['NoamLR', 'step']:
            scheduler.step()
    if config['dataset'] in ['sol_calc/ALL', 'sol_calc/ALL/COMPLETE', 'qm9/u0']: 
        return np.sqrt((all_loss / len(dataloader.dataset)))
    elif config['dataset'] in ['solNMR']:
        if config['propertyLevel'] == 'molecule':
            return all_loss / len(dataloader.dataset) 
        if config['propertyLevel'] == 'atom':
             return all_loss / all_atoms
        if config['propertyLevel'] == 'atomMol':
            return loss.item()
    else:
        return (all_loss / all_atoms.item()).item()

def test_physnet(model, dataloader, config):
    '''
    taskType
    '''
    model.eval()
    error = 0
    total_N = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(config['device'])
            if config['mask']:
                error += get_metrics_fn(config['metrics'])(model(data)['atom_prop'].view(-1)[data.mask>0], data.y[data.mask>0])*data.mask.sum().item()
                total_N += data.mask.sum().item()
            else:
                if config['dataset'] in ['sol_calc/ALL', 'sol_calc/ALL/COMPLETE']:
                    error += get_metrics_fn(config['metrics'])(model(data)['mol_prop'].view(-1), data.CalcSol)*data.E.size()[0]
                elif config['dataset'] in ['solNMR']:
                    if config['test_level'] == 'molecule':
                        error += get_metrics_fn(config['metrics'])(model(data)['mol_prop'].view(-1), data.mol_y)*data.mol_y.size()[0]
                    if config['test_level'] == 'atom':
                        total_N += data.N.sum().item()
                        error += get_metrics_fn(config['metrics'])(model(data)['atom_prop'].view(-1), data.atom_y)*data.N.sum().item()
                    #if config['action'] == 'atomMol':
                    #    error += get_metrics_fn(config['metrics'])(model(data)['mol_prop'].view(-1), data.mol_y)*data.mol_y.size()[0]
                elif config['dataset'] in ['qm9/u0']:
                    error += get_metrics_fn(config['metrics'])(model(data)['mol_prop'].view(-1), data.E)*data.E.size()[0]
                else:
                    total_N += data.N.sum().item()
                    error += get_metrics_fn(config['metrics'])(model(data)['atom_prop'].view(-1), data.y)*data.N.sum().item()
        if config['dataset'] in ['sol_calc/ALL', 'sol_calc/ALL/COMPLETE', 'qm9/u0']:
            return error.item() / len(dataloader.dataset) # MAE
        elif config['dataset'] in ['solNMR']:
            if config['test_level'] == 'molecule':
                return error.item() / len(dataloader.dataset) # MAE
            if config['test_level'] == 'atom':
                return error / total_N # MAE
        else:
            return error.item() / total_N # MAE