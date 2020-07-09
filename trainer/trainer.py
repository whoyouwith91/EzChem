from helper import *
from data import *
from model import *

def cv_train(config):
    results = []

    for i in range(5):
        config['data_path'] = os.path.join(config['cv_path'], 'cv_'+str(i)+'/')

        loader = get_data_loader(config)
        train_loader, val_loader, _, std, num_features, num_bond_features, num_i_2 = loader.train_loader, loader.val_loader, loader.test_loader, loader.std, loader.num_features, loader.num_bond_features, loader.num_i_2
        config['num_features'], config['num_bond_features'], config['num_i_2'], config['std'] = int(num_features), num_bond_features, num_i_2, std 
        
        if config['model'] in ['loopybp', 'wlkernel']:
           config['atom_fdim'] = int(num_features)
           config['bond_fdim'] = int(num_bond_features)
           config['atom_messages'] = False
           config['outDim'] = config['dimension']
 
        model = get_model(config)
        model_ = model(config).to(config['device'])
        #num_params = param_count(model_)
       
        if config['adam']:
            optimizer = torch.optim.Adam(model_.parameters(), lr=0.001)
        if config['sgd']:
            optimizer = torch.optim.SGD(model_.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        if config['swa']:
            optimizer = torchcontrib.optim.SWA(optimizer)
        if config['lr_style'] == 'decay':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=5, min_lr=0.00001)

        for _ in range(1, config['epochs']):
            val_res = []

            loss = train(model_, optimizer, train_loader, config)
            
            if config['dataset'] in ['mp', 'xlogp3']:
                #train_error = np.asscalar(loss.data.cpu().numpy()) # don't test the entire train set.
                train_error = loss
            else:
                train_error = test(model_, train_loader, config)
            val_error = test(model_, val_loader, config)
            #test_error = test(model_, test_loader, this_dic)

            #train_res.append(train_error)
            val_res.append(val_error)
            #test_res.append(test_error)

            #with open(os.path.join(cv_path, 'data.txt'), 'a') as f1:
            #    f1.write(str(epoch) + '\t' + str(round(time_toc-time_tic, 2)) + '\t' + str(round(train_error, 7)) + '\t' + str(round(val_error, 7)) + '\t' + str(round(test_error, 7)) + '\t' + str(param_norm(model_)) + '\t' + str(grad_norm(model_)) + '\n') 
            
        results.append(min(val_res))
        
    return np.average(results)

def train(model, optimizer, dataloader, config):
    '''
    Define loss and backpropagation
    '''
    model.train()
    loss_all = 0
    y = []
    means = []

    for data in dataloader:
        if config['model'] not in ['loopybp', 'wlkernel', 'loopybp_dropout', 'wlkernel_dropout', 'loopybp_swag', 'wlkernel_swag']:
            data = data.to(config['device'])
        if config['model'] in ['loopybp', 'wlkernel', 'loopybp_dropout', 'wlkernel_dropout', 'loopybp_swag', 'wlkernel_swag']:
            batchTargets = data.targets()
            data = data.batch_graph()
            data.y = torch.FloatTensor(batchTargets).to(config['device'])
            data.y = data.y.view(-1)
            
        optimizer.zero_grad()
        if config['taskType'] == 'single':
            y0, _ = model(data)
            loss = get_loss_fn(config['loss'])(y0, data.y)
        else:
            y0, y1 = model(data)
            loss = get_loss_fn(config['loss'])(y0, data.y) + get_loss_fn(config['loss'])(y1, data.y1)
        loss.backward()
        optimizer.step()
        
        y.extend(data.y.cpu().data.numpy())
        means.extend(y0.cpu().data.numpy())
    if config['dataset'] in ['mp', 'mp_drugs']:
        rmse = np.mean((np.array(means).reshape(-1,) - np.array(y).squeeze())**2.)**0.5
        return rmse
    else:
        return loss

def test(model, dataloader, config):
    '''
    taskType
    '''
    model.eval()
    error = 0
    if config['taskType'] == 'multi':
       error1 = 0
    with torch.no_grad():
       for data in dataloader:
            if config['model'] not in ['loopybp', 'wlkernel', 'loopybp_dropout', 'wlkernel_dropout', 'loopybp_swag', 'wlkernel_swag']:
                data = data.to(config['device'])
            if config['model'] in ['loopybp', 'wlkernel', 'loopybp_dropout', 'wlkernel_dropout', 'loopybp_swag', 'wlkernel_swag']:
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
                if config['taskType'] == 'single':
                    error += get_metrics_fn(config['metrics'])(model(data)[0], data.y) * data.num_graphs
                else:
                    y, y1 = model(data)
                    error += get_metrics_fn(config['metrics'])(y, data.y) * data.num_graphs
                    error1 += get_metrics_fn(config['metrics'])(y1, data.y1) * data.num_graphs
       if config['dataset'] in ['qm9']:
           return error / len(dataloader.dataset) # MAE
       else:
           if config['taskType'] == 'single':
              return math.sqrt(error / len(dataloader.dataset)) # RMSE for ws, logp, mp, etc.
           else:
              return (math.sqrt(error / len(dataloader.dataset)), math.sqrt(error1 / len(dataloader.dataset))) 

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
            mean, log_var, regularization = model(data)
            loss = get_loss_fn(config['loss'])(data.y, mean, log_var) + regularization
            logvars.extend(log_var.cpu().data.numpy())
        if config['uncertainty'] == 'epistemic':
            mean = model(data)
            loss = get_loss_fn(config['loss'])(data.y, mean)
        y.extend(data.y.cpu().data.numpy())
        means.extend(mean.cpu().data.numpy())
        loss.backward()
        if config['model'] in ['loopybp_dropout', '1-2-GNN_dropout', 'wlkernel_dropout']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        loss_all += loss.cpu().data.numpy()
    rmse = np.mean((np.array(means).reshape(-1,) - np.array(y).squeeze())**2.)**0.5
    if config['uncertainty'] == 'aleatoric':
        return loss_all/num, rmse, np.array(y), np.array(means), np.array(logvars)
    if config['uncertainty'] == 'epistemic':
        return loss_all/num, rmse, None, None, None

def train_VAE(model, optimizer, dataloader, config):
    '''
    Define loss and backpropagation
    '''
    model.train()
    loss_all = 0

    for i, data in enumerate(dataloader):
        recon_batch, mu, logvar = model(data)
        loss = get_loss_fn(config['loss'])(recon_batch, Variable(data.SRC.view(-1,), requires_grad=False), \
                                           mu.view(-1, config['varDimen']), logvar.view(-1, config['varDimen']), config)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        loss_all += loss.cpu().data.numpy()
         
    return loss_all/len(dataloader)

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

def test_VAE(model, dataloader, config):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for data in dataloader:
            recon_batch, mu, logvar = model(data)
            loss = get_loss_fn(config['loss'])(recon_batch, Variable(data.SRC.view(-1,), requires_grad=False), \
                                           mu.view(-1, config['varDimen']), logvar.view(-1, config['varDimen']), config)
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

