import sys
from helper import *
from data import *
#from model import *
from models import *

def cv_train(config, table):
    results = []

    for seed in [1, 13, 31]:
        #config['data_path'] = os.path.join(config['cv_path'], 'cv_'+str(i)+'/')
        #print(config)
        #print(config['data_path'])   
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
           torch.cuda.manual_seed_all(0)
        config['device'] = device

        if config['pooling'] == 'set2set':
           config['pooling'] = list()
           config['pooling'].append('set2set')
           config['pooling'].append('1')

        loader = get_data_loader(config)
        train_loader, val_loader, _, std, num_features, num_bond_features, num_i_2 = loader.train_loader, loader.val_loader, loader.test_loader, loader.std, loader.num_features, loader.num_bond_features, loader.num_i_2
        config['num_features'], config['num_bond_features'], config['num_i_2'], config['std'] = int(num_features), num_bond_features, num_i_2, std 
        
        #if config['model'] in ['loopybp', 'wlkernel']:
        #   config['atom_fdim'] = int(num_features)
        #   config['bond_fdim'] = int(num_bond_features)
        #   config['atom_messages'] = False
        #   config['outDim'] = config['dimension']
        genModel = get_model(config)
        args = objectview(config)
        if config['model'] == '1-GNN':
             model = genModel(args.num_layer, args.emb_dim, args.NumOutLayers, args.num_tasks, graph_pooling=args.pooling, gnn_type=args.gnn_type)
        else:
             model = genModel(args.num_layer, args.emb_dim, args.NumOutLayers, args.num_tasks, num_i_2, graph_pooling=args.pooling, gnn_type=args.gnn_type)

        model_ = model.to(config['device'])
        #num_params = param_count(model_)
       
        if config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model_.parameters(), lr=config['lr'])
        if config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(model_.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)
        if config['optimizer'] == 'swa':
            optimizer = torchcontrib.optim.SWA(optimizer)
        if config['lr_style'] == 'decay':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=5, min_lr=0.00001)

        for _ in range(1, config['epochs']):
            val_res = []

            loss = train(model_, optimizer, train_loader, config)
            
            if config['dataset'] in ['mp', 'xlogp3']:
                #train_error = np.asscalar(loss.data.cpu().numpy()) # don't test the entire train set.
                train_error = loss
            else:
                train_error = test(model_, train_loader, config)
            val_error = test(model_, val_loader, config)
            val_res.append(val_error)
        
        results.append(min(val_res))
    
    table.add_row([config['emb_dim'], config['num_layer'], config['NumOutLayers'], config['lr'], config['batch_size'], np.mean(results), np.std(results)])
    print(table)
    sys.stdout.flush()
    print('\n')
    sys.stdout.flush()
    return np.average(results)

def train(model, optimizer, dataloader, config):
    '''
    Define loss and backpropagation
    '''
    model.train()
    #y = []
    #means = []
    all_loss = 0
    if config['propertyLevel'] == 'atom':
        all_atoms = 0

    for data in dataloader:
        if config['gnn_type'] not in ['loopybp', 'wlkernel', 'loopybp_dropout', 'wlkernel_dropout', 'loopybp_swag', 'wlkernel_swag']:
            data = data.to(config['device'])
        if config['gnn_type'] in ['loopybp', 'wlkernel', 'loopybp_dropout', 'wlkernel_dropout', 'loopybp_swag', 'wlkernel_swag']:
            batchTargets = data.targets()
            data = data.batch_graph()
            data.y = torch.FloatTensor(batchTargets).to(config['device'])
            data.y = data.y.view(-1)
            
        optimizer.zero_grad()
        y0, _ = model(data)
        if config['taskType'] == 'single' and config['propertyLevel'] == 'molecule':
            loss = get_loss_fn(config['loss'])(y0, data.y)
            all_loss += loss.item() * data.num_graphs
        if config['taskType'] == 'multi' and config['dataset'] == 'calcSolLogP':
            loss = get_loss_fn(config['loss'])(y0[:,0], data.y) + get_loss_fn(config['loss'])(y0[:,1], data.y1) + get_loss_fn(config['loss'])(y0[:,2], data.y2)
        if config['dataset'] == 'commonProperties':
            loss = get_loss_fn(config['loss'])(y0[:,0], data.y) + get_loss_fn(config['loss'])(y0[:,1], data.y1) + get_loss_fn(config['loss'])(y0[:,2], data.y2) + get_loss_fn(config['loss'])(y0[:,3], data.y3)
            all_loss += loss.item() * data.num_graphs
        if config['propertyLevel'] == 'atom':
            #print(y0[data.mask>0])
            loss = get_loss_fn(config['loss'])(data.y, y0, data.mask)
            all_atoms += data.mask.sum()
            all_loss += loss.item() * data.mask.sum()
            #print(loss)
            #sys.stdout.flush()

        loss.backward()
        if config['optimizer'] in ['sgd']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        #y.extend(data.y.cpu().data.numpy())
        #means.extend(y0.cpu().data.numpy())
    #if config['dataset'] in ['mp', 'mp_drugs']:
    #    rmse = np.mean((np.array(means).reshape(-1,) - np.array(y).squeeze())**2.)**0.5
    #    return rmse
    #else:
    if config['propertyLevel'] == 'atom':
        return np.sqrt(all_loss.item() / all_atoms.item())
    if config['propertyLevel'] == 'molecule':
        return np.sqrt(all_loss / len(dataloader.dataset))

def test(model, dataloader, config):
    '''
    taskType
    '''
    model.eval()
    error = 0
    if config['propertyLevel'] == 'atom':
        total_N = 0
    with torch.no_grad():
       for data in dataloader:
            if config['gnn_type'] not in ['loopybp', 'wlkernel', 'loopybp_dropout', 'wlkernel_dropout', 'loopybp_swag', 'wlkernel_swag']:
                data = data.to(config['device'])
            if config['gnn_type'] in ['loopybp', 'wlkernel', 'loopybp_dropout', 'wlkernel_dropout', 'loopybp_swag', 'wlkernel_swag']:
                y = data.targets()
                bs = data.num_graphs
                data = data.batch_graph()
                data.y = torch.FloatTensor(y).to(config['device'])
                data.y = data.y.view(-1)
                data.num_graphs = bs
            if config['dataset'] in ['qm9']:
                error_name = 'l1'
                error += get_metrics_fn(error_name)(model(data)[0], data.y).item() * data.num_graphs
            else:
                if config['taskType'] == 'single' and config['propertyLevel'] == 'molecule':
                    error += get_metrics_fn(config['metrics'])(model(data)[0], data.y) * data.num_graphs
                if config['taskType'] == 'multi' and config['dataset'] == 'calcSolLogP':
                    y0, _ = model(data)
                    error += (get_loss_fn(config['loss'])(y0[:,0], data.y) + get_loss_fn(config['loss'])(y0[:,1], data.y1) + get_loss_fn(config['loss'])(y0[:,2], data.y2))*data.num_graphs
                if config['taskType'] == 'multi' and config['dataset'] == 'commonProperties':
                    y0, _ = model(data)
                    error += (get_loss_fn(config['loss'])(y0[:,0], data.y) + get_loss_fn(config['loss'])(y0[:,1], data.y1) + get_loss_fn(config['loss'])(y0[:,2], data.y2) + get_loss_fn(config['loss'])(y0[:,3], data.y3))*data.num_graphs
                if config['propertyLevel'] == 'atom':
                    #print(data.mask)
                    #sys.stdout.flush()
                    #print(data.y)
                    #sys.stdout.flush()
                    total_N += data.mask.sum().item()
                    error += get_metrics_fn(config['metrics'])(model(data)[0][data.mask>0].reshape(-1,1), data.y[data.mask>0].reshape(-1,1))*data.mask.sum().item()
                    #error += MAE_loss(model(data)[0], data.y, data.mask)
       if config['dataset'] in ['qm9']:
           return error / len(dataloader.dataset) # MAE
       elif config['propertyLevel'] == 'atom' and config['dataset'] in ['nmr/hydrogen', 'nmr/carbon']:
           #print(error.item(), total_N.item())
           #sys.stdout.flush()
           return error.item() / total_N # MAE
       else:
           if config['taskType'] == 'single':
              return math.sqrt(error / len(dataloader.dataset)) # RMSE for ws, logp, mp, etc.
           if config['taskType'] == 'multi':
              return math.sqrt(error / len(dataloader.dataset))

def train_dropout(model, optimizer, dataloader, config):
    model.train()
    loss_all = 0
    y = []
    means = []
    logvars = []
    num = 0
    for data in dataloader:
        num += 1
        if config['model'] not in ['loopybp', 'wlkernel', 'loopybp_dropout', 'wlkernel_dropout', 'loopybp_swag', 'wlkernel_swag']:
            data = data.to(config['device'])
        if config['model'] in ['loopybp', 'wlkernel', 'loopybp_dropout', 'wlkernel_dropout', 'loopybp_swag', 'wlkernel_swag']:
            batchTargets = data.targets()
            data = data.batch_graph()
            data.y = torch.FloatTensor(batchTargets).to(config['device'])
            data.y = data.y.view(-1)
        optimizer.zero_grad()
        if config['uncertainty'] == 'aleatoric':
            mean, log_var, _ = model(data)
            loss = get_loss_fn(config['loss'])(data.y, mean, log_var)
            logvars.extend(log_var.cpu().data.numpy())
        if config['uncertainty'] == 'epistemic':
            mean = model(data)
            loss = get_loss_fn(config['loss'])(data.y, mean)
        y.extend(data.y.cpu().data.numpy())
        means.extend(mean.cpu().data.numpy())
        loss.backward()
        if config['model'] in ['loopybp_dropout', '1-2-GNN_dropout', 'wlkernel_dropout']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_all += loss.cpu().data.numpy()
    rmse = np.mean((np.array(means).reshape(-1,) - np.array(y).squeeze())**2.)**0.5
    if config['uncertainty'] == 'aleatoric':
        return loss_all/num, rmse, np.array(y), np.array(means), np.array(logvars)
    if config['uncertainty'] == 'epistemic':
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

