ALLOWABLE_ATOM_SYMBOLS = ['H', 'C', 'N', 'O', 'S', 'F', 'I', 'P', 'Cl', 'Br']

def mol_with_atom_index( mol ):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol
    
# filter out molecules that are within elements groups
with open('/beegfs/dz1061/gcn/chemGraph/data/smiles/atom10_smiles.txt', 'w') as f:
    for i in text:
        mol = Chem.MolFromSmiles(i)
        elements = list(Counter(atom.GetSymbol() for atom in mol.GetAtoms()).keys())
        if set(elements) < set(ALLOWABLE_ATOM_SYMBOLS):
            f.write(i)
            f.write('\n')
            
            from collections import Counter

# create a vocabunary
counter = Counter()

with open('/beegfs/dz1061/gcn/chemGraph/data/smiles/atom10_smiles.txt') as rf:
    for line in rf:
        items = tokenizer.tokenize(line.strip())
        counter.update(items)
        #items = self._tokenize(line.strip())
vocab = torchtext.vocab.Vocab(counter)

torch.save(vocab, '/beegfs/dz1061/gcn/chemGraph/data/smiles/atom10_vocab.pt')


def get_hbond_donor_indice(m):
    """
    indice = m.GetSubstructMatches(HDonorSmarts)
    if len(indice)==0: return np.array([])
    indice = np.array([i for i in indice])[:,0]
    return indice
    """
    # smarts = ["[!$([#6,H0,-,-2,-3])]", "[!H0;#7,#8,#9]"]
    smarts = ["[!#6;!H0]"]
    indice = []
    for s in smarts:
        s = Chem.MolFromSmarts(s)
        indice += [i[0] for i in m.GetSubstructMatches(s)]
    indice = np.array(indice)
    return indice


def get_hbond_acceptor_indice(m):
    # smarts = ["[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]",
    #          "[#6,#7;R0]=[#8]"]
    smarts = [
        '[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' +
                                     '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' +
                                     '$([nH0,o,s;+0])]']
    indice = []
    for s in smarts:
        #print(s)
        s = Chem.MolFromSmarts(s)
        indice += [i[0] for i in m.GetSubstructMatches(s)]
    indice = np.array(indice)
    return indice

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
        
        if config['dataset'] in ['qm9/nmr/carbon', 'nmr/carbon', 'nmr/hydrogen', 'protein/nmr']: # for qm9/nmr/carbon or Exp. nmr/carbon nmr/hydrogen
            loss = get_loss_fn(config['loss'])(data.y[data.mask>0], y0['atom_prop'].float().view(-1)[data.mask>0])
            all_atoms += data.mask.sum() # with mask information
            all_loss += loss.item()*data.mask.sum()
        
        elif config['dataset'] in ['sol_calc/ALL', 'sol_calc/ALL/COMPLETE', 'sol_calc/smaller', 'deepchem/freesol', 'deepchem/delaney', 'deepchem/logp', 'mp/bradley', 'pka/dataWarrior/acidic', 'pka/dataWarrior/basic']:
            loss = get_loss_fn(config['loss'])(data.mol_sol_wat, y0['mol_prop'].view(-1)) # data.mol_sol_wat is not only for water solvation energy. TODO
            all_loss += loss.item()*data.mol_sol_wat.size()[0]
        
        elif config['dataset'] in ['solNMR']: #TODO
            if config['propertyLevel'] == 'molecule':
                loss = get_loss_fn(config['loss'])(data.mol_y, y0['mol_prop'].view(-1)) # data.CalcSol is only for water solvation energy
                all_loss += loss.item()*data.mol_y.size()[0]
            elif config['propertyLevel'] == 'atom':
                loss = get_loss_fn(config['loss'])(data.atom_y, y0['atom_prop'].float().view(-1))
                all_atoms += data.N.sum().item()
                all_loss += loss.item()*data.N.sum()
            elif config['propertyLevel'] == 'atomMol':
                loss = get_loss_fn(config['loss'])(data.mol_y, y0['mol_prop'].view(-1)) + get_loss_fn(config['loss'])(data.atom_y, y0['atom_prop'].float().view(-1))
            else:
                pass 

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
    
    if config['dataset'] in ['sol_calc/ALL', 'sol_calc/ALL/COMPLETE', 'qm9/u0', 'sol_calc/smaller', 'deepchem/freesol', 'deepchem/delaney', 'deepchem/logp', 'mp/bradley', 'pka/dataWarrior/acidic', 'pka/dataWarrior/basic']: 
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
            if config['dataset'] in ['qm9/nmr/carbon', 'nmr/carbon', 'nmr/hydrogen']: # for qm9/nmr/carbon or Exp. nmr/carbon nmr/hydrogen
                error += get_metrics_fn(config['metrics'])(model(data)['atom_prop'].view(-1)[data.mask>0], data.y[data.mask>0])*data.mask.sum().item()
                total_N += data.mask.sum().item()

            elif config['dataset'] in ['sol_calc/ALL', 'sol_calc/ALL/COMPLETE', 'sol_calc/smaller', 'deepchem/freesol', 'deepchem/delaney', 'deepchem/logp', 'mp/bradley', 'pka/dataWarrior/acidic', 'pka/dataWarrior/basic']:
                error += get_metrics_fn(config['metrics'])(model(data)['mol_prop'].view(-1), data.mol_sol_wat)*data.mol_sol_wat.size()[0]
            
            elif config['dataset'] in ['solNMR']:
                if config['test_level'] == 'molecule':
                    error += get_metrics_fn(config['metrics'])(model(data)['mol_prop'].view(-1), data.mol_y)*data.mol_y.size()[0]
                elif config['test_level'] == 'atom':
                    total_N += data.N.sum().item()
                    error += get_metrics_fn(config['metrics'])(model(data)['atom_prop'].view(-1), data.atom_y)*data.N.sum().item()
                    #if config['action'] == 'atomMol':
                    #    error += get_metrics_fn(config['metrics'])(model(data)['mol_prop'].view(-1), data.mol_y)*data.mol_y.size()[0]
                else:
                    pass 
            
            elif config['dataset'] in ['qm9/u0']:
                error += get_metrics_fn(config['metrics'])(model(data)['mol_prop'].view(-1), data.E)*data.E.size()[0]
            
            else:
                total_N += data.N.sum().item()
                error += get_metrics_fn(config['metrics'])(model(data)['atom_prop'].view(-1), data.y)*data.N.sum().item()
        
        if config['dataset'] in ['sol_calc/ALL', 'sol_calc/ALL/COMPLETE', 'qm9/u0', 'sol_calc/smaller', 'deepchem/freesol', 'deepchem/delaney', 'deepchem/logp', 'mp/bradley', 'pka/dataWarrior/acidic', 'pka/dataWarrior/basic']:
            return error.item() / len(dataloader.dataset) # MAE
        elif config['dataset'] in ['solNMR']:
            if config['test_level'] == 'molecule':
                return error.item() / len(dataloader.dataset) # MAE
            if config['test_level'] == 'atom':
                return error / total_N # MAE
        else:
            return error.item() / total_N # MAE

def getGraphs():
    if this_dic['dataset'] in ['qm9/nmr/carbon', 'qm9/nmr/hydrogen']:
                if this_dic['ACSF']:
                    acsf = ACSF(
                                species=['C', 'F', 'H', 'N', 'O'],
                                rcut=args.cutoff,
                                g2_params=[[1, 1], [1, 2], [1, 3]],
                                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)
                for id_, mol, tar in zip(all_data['molecule_id'], all_data['rdmol'], all_data['values']):
                    molgraphs = {}
                    mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])

                    if not this_dic['ACSF']:
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    if this_dic['ACSF']:
                        path_to_xyz = '/ext3/qm9/xyz/QMXYZ'
                        #if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(file_id))):
                        #    MolToXYZFile(mol, os.path.join(alldatapath, args.dataset, 'split', args.style, this_dic['xyz'], '{}.xyz'.format(file_id)))
                        atoms = ase_read(os.path.join(path_to_xyz, '{}.xyz'.format(id_))) # path to the singularity file overlay-50G-10M.ext3
                        molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms())))
                    
                    if not this_dic['physnet']:
                        atomic_number = []
                        for atom in mol.GetAtoms():
                            atomic_number.append(atom.GetAtomicNum())
                        z = torch.tensor(atomic_number, dtype=torch.long)
                        molgraphs['Z'] = z
                        molgraphs['N'] = torch.FloatTensor([mol.GetNumAtoms()])

                        molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                        molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                        mask = np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)
                        vals = np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)
                        for k, v in tar[0].items():
                            mask[int(k), 0] = 1.0
                            vals[int(k), 0] = v
                        molgraphs['atom_y'] = torch.FloatTensor(vals).flatten()
                        molgraphs['mask'] = torch.FloatTensor(mask).flatten()
                        
                        examples.append(molgraphs)
                    else:
                        mol_sdf = mol
                        examples[id_] = [mol_sdf, tar]

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))

    elif this_dic['dataset'] == 'qm9/u0':
                if this_dic['ACSF']:
                    acsf = ACSF(
                                species=['C', 'F', 'H', 'N', 'O'],
                                rcut=10.0,
                                g2_params=[[1, 1], [1, 2], [1, 3]],
                                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)
                for name, mol in suppl.items():
                    #name = mol.GetProp('_Name')
                    if name not in nmr.keys():
                        continue
                    molgraphs = {}
                    #mol = value[0][0]
                    mol_graph = MolGraph(mol, args.usePeriodics, this_dic['model'])

                    if not this_dic['ACSF']:
                        molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                    if this_dic['BPS']:
                        bp_features = torch.FloatTensor(bp_sym(mol)[0, :mol.GetNumAtoms(), 1:])
                        molgraphs['x'] = torch.cat([molgraphs['x'], bp_features], 1)
                    if this_dic['ACSF']:
                        if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(name))):
                            MolToXYZFile(mol, os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(name)))
                        atoms = ase_read(os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(name)))
                        molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms())))
                    
                    atomic_number = []
                    for atom in mol.GetAtoms():
                        atomic_number.append(atom.GetAtomicNum())
                    z = torch.tensor(atomic_number, dtype=torch.long)
                    molgraphs['Z'] = z

                    molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                    molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                    
                    molgraphs['atom_y'] = torch.FloatTensor(nmr[name])
                    molgraphs['mol_y'] = torch.FloatTensor([u0[name]])
                    #molgraphs['atom_y'] = torch.FloatTensor([i[0] for i in rdMolDescriptors._CalcCrippenContribs(mol)])
                    molgraphs['N'] = torch.FloatTensor([mol.GetNumAtoms()])
                    
                    if this_dic['atom_classification']:
                        try:
                            efg = mol2frag(mol, returnidx=True, vocabulary=list(efgs_vocabulary), toEnd=True, extra_included=True, TreatHs='include', isomericSmiles=False)
                            molgraphs['atom_efgs'] = torch.tensor(getAtomToEFGs(efg, efgs_vocabulary)).view(-1).long()
                        except:
                            molgraphs['atom_efgs'] = None
                    examples.append(molgraphs)

                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, args.format, this_dic['model'], 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))

class GNN_1_2(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1_2, self).__init__()
        self.num_i_2 = config['num_i_2']
        self.config = config
        self.dataset = config['dataset']
        self.num_layer = config['num_layer']
        self.fully_connected_layer_sizes = config['fully_connected_layer_sizes']
        self.JK = config['JK']
        self.emb_dim = config['emb_dim']
        self.num_tasks = config['num_tasks']
        self.graph_pooling = config['pooling']
        self.propertyLevel = config['propertyLevel']
        self.gnn_type = config['gnn_type']
        self.gradCam = config['gradCam']
        self.tsne = config['tsne']
        self.uncertainty = config['uncertainty']
        self.uncertaintyMode = config['uncertaintyMode']
        self.weight_regularizer = config['weight_regularizer']
        self.dropout_regularizer = config['dropout_regularizer']
        self.features = config['mol_features']

        self.gnn = GNN(config)
        self.convISO1 = GraphConv(self.emb_dim + self.num_i_2, self.emb_dim)
        self.convISO2 = GraphConv(self.emb_dim, self.emb_dim)
        self.outLayers = nn.ModuleList()
        #self.out_batch_norms = torch.nn.ModuleList()

        if self.uncertainty:
            self.uncertaintyLayers = nn.ModuleList()
        if self.graph_pooling != 'edge': # except for edge pooling, coded here
            self.pool = PoolingFN(config)

        #For graph-level property predictions
        if self.graph_pooling[:-1][0] == "set2set": # set2set will double dimension
            self.mult = 3
        else:
            self.mult = 2
        
        if self.JK == "concat": # change readout layers input and output dimension
            embed_size = self.mult * (self.num_layer + 1) * self.emb_dim
        elif self.graph_pooling == 'conv': # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim / 2  
        elif self.features:
            if self.dataset == 'sol_calc/ALL': # 208 total mol descriptors # total is 200
               embed_size = self.mult * self.emb_dim + 208
        else: # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim 
        
        for idx, (L_in, L_out) in enumerate(zip([embed_size] + self.fully_connected_layer_sizes, self.fully_connected_layer_sizes + [self.num_tasks])):
            if idx != len(self.fully_connected_layer_sizes):
                fc = nn.Sequential(Linear(L_in, L_out), activation_func(config), nn.Dropout(config['drop_ratio']))
                #L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)    
                if self.uncertainty: # for uncertainty 
                    self.uncertaintyLayers.append(NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer)) 
            else:
                fc = nn.Sequential(Linear(L_in, L_out), nn.Dropout(config['drop_ratio']))
                last_fc = fc
            self.outLayers.append(fc)

        if self.uncertaintyMode == 'epistemic': 
            self.drop_mu = NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer) # working on last layer
        if self.uncertaintyMode == 'aleatoric': # 
            self.outLayers.append(last_fc) 

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))
        #self.convISO1.load_state_dict(torch.load(conv1File))
        #self.convISO2.load_state_dict(torch.load(conv2File))

    def forward(self, data):
        x, iso_type_2, edge_index, edge_index_2, assignment_index_2, edge_attr, batch, batch_2 = \
                data.x, data.iso_type_2, data.edge_index, data.edge_index_2, data.assignment_index_2, \
                    data.edge_attr.long(), data.batch, data.batch_2

        node_representation = self.gnn(x, edge_index, edge_attr)
        x_1 = self.pool(node_representation, batch)

        x = avg_pool(node_representation, data.assignment_index_2)
        #data.x = torch.cat([data.x, data_iso], dim=1)
        x = torch.cat([x, iso_type_2], dim=1)
        x = F.relu(self.convISO1(x, edge_index_2))
        x = F.relu(self.convISO2(x, edge_index_2))
        x_2 = scatter_mean(x, batch_2, dim=0)   # to add stability to models
        
        MolEmbed = torch.cat([x_1, x_2], dim=1)
        if not self.training and not self.gradCam and self.tsne: # for TSNE analysis
            return node_representation, MolEmbed

        # read-out layers
        if not self.uncertainty:
            for layer in self.outLayers:
                MolEmbed = layer(MolEmbed)
            if self.num_tasks > 1:
                return MolEmbed, None
            if self.propertyLevel == 'atom':
                return MolEmbed.view(-1,1), None
            if self.graph_pooling == 'edge':
                return MolEmbed.view(-1), None
            else:
                #return self.pool(MolEmbed, batch).view(-1), None
                return MolEmbed.view(-1), None
        # for uncertainty analysis
        else:
            for layer, drop in zip(self.outLayers[1:-1], self.uncertaintyLayers):
                x, _ = drop(MolEmbed, layer)
            if self.config['uncertainty'] == 'epistemic':
                mean, regularization[-1] = self.drop_mu(x, self.outLayers[-1])
                return mean.squeeze()
            if self.config['uncertainty'] == 'aleatoric':
                mean = self.outLayers[-2](x)
                log_var = self.outLayers[-1](x)
                return mean, log_var


class GNN_1_EFGS(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1_EFGS, self).__init__()
        self.num_layer = config['num_layer']
        self.NumOutLayers = config['NumOutLayers']
        self.JK = config['JK']
        self.emb_dim = config['emb_dim']
        self.num_tasks = config['num_tasks']
        self.graph_pooling = config['pooling']
        self.efgs_vocab = config['efgs_lenth']

        self.gnn = GNN(config)
        # For EFGS
        self.convISO3 = GraphConv(self.emb_dim + self.efgs_vocab, self.emb_dim)
        self.convISO4 = GraphConv(self.emb_dim, self.emb_dim)


        self.outLayers = nn.ModuleList()

        #Different kind of graph pooling
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * self.emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(self.emb_dim, 1))
        elif self.graph_pooling == "set2set":
            set2set_iter = 2
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * self.emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(self.emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if self.graph_pooling[:-1][0] == "set2set":
            self.mult = 3
        else:
            self.mult = 2
        if self.JK == "concat":
            L_in, L_out = self.mult * (self.num_layer + 1) * self.emb_dim, self.emb_dim
        else:
            L_in, L_out = self.mult * self.emb_dim, self.emb_dim

        fc = nn.Sequential(Linear(L_in, L_out), nn.ReLU())
        self.outLayers.append(fc)
        for _ in range(self.NumOutLayers):
            L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)
            fc = nn.Sequential(Linear(L_in, L_out), activation_func(config))
            self.outLayers.append(fc)
        last_fc = nn.Linear(L_out, self.num_tasks)
        self.outLayers.append(last_fc)

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, iso_type_2, iso_type_3, edge_index, edge_index_2, edge_index_3, assignment_index_2, \
                assignment_index_3, edge_attr, batch, batch_2, batch_3 = data.x, data.iso_type_2, data.iso_type_3, \
                    data.edge_index, data.edge_index_2, data.edge_index_3, data.assignment_index_2, data.assignment_index_3, \
                        data.edge_attr.long(), data.batch, data.batch_2, data.batch_3
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)
        x_1 = self.pool(node_representation, batch)

        x = avg_pool(node_representation, assignment_index_3)
        x = torch.cat([x, iso_type_3], dim=1)
        x = F.relu(self.convISO3(x, edge_index_3))
        x = F.relu(self.convISO4(x, edge_index_3))
        x_3 = scatter_mean(x, batch_3, dim=0) # 

        MolEmbed = torch.cat([x_1, x_3], dim=1)
        for layer in self.outLayers:
             MolEmbed = layer(MolEmbed)

        if self.num_tasks > 1:
            return MolEmbed, None
        else:
            return MolEmbed.view(-1), None

class GNN_1_WithWater(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1_WithWater, self).__init__()
        self.dataset = config['dataset']
        self.num_layer = config['num_layer']
        self.fully_connected_layer_sizes = config['fully_connected_layer_sizes']
        self.JK = config['JK']
        self.emb_dim = config['emb_dim']
        self.num_tasks = config['num_tasks']
        self.graph_pooling = config['pooling']
        self.propertyLevel = config['propertyLevel']
        self.gnn_type = config['gnn_type']
        self.gradCam = config['gradCam']
        self.tsne = config['tsne']
        self.uncertainty = config['uncertainty']
        self.uncertaintyMode = config['uncertaintyMode']
        self.weight_regularizer = config['weight_regularizer']
        self.dropout_regularizer = config['dropout_regularizer']
        self.features = config['mol_features']

        self.gnn = GNN(config)
        self.outLayers = nn.ModuleList()
        if self.uncertainty:
            self.uncertaintyLayers = nn.ModuleList()
        #Different kind of graph pooling
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * self.emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(self.emb_dim, 1))
        elif self.graph_pooling == "set2set":
            set2set_iter = 2
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * self.emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(self.emb_dim, set2set_iter)
        elif self.graph_pooling == 'conv':
            self.pool = []
            self.pool.append(global_add_pool)
            self.pool.append(global_mean_pool)
            self.pool.append(global_max_pool)
            self.pool.append(GlobalAttention(gate_nn = torch.nn.Linear(self.emb_dim, 1)))
            self.pool.append(Set2Set(self.emb_dim, 2))
            self.convPool = nn.Conv1d(len(self.pool), 1, 2, stride=2)

        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level property predictions
        if self.graph_pooling[:-1][0] == "set2set": # set2set will double dimension
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat": # change readout layers input and output dimension
            embed_size = self.mult * (self.num_layer + 1) * self.emb_dim
        elif self.graph_pooling == 'conv': # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim / 2  
        elif self.features:
            if self.dataset == 'sol_calc/ALL': # 208 total mol descriptors # total is 200
               embed_size = self.mult * self.emb_dim + 208
        else: # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim 

        for idx, (L_in, L_out) in enumerate(zip([embed_size] + self.fully_connected_layer_sizes, self.fully_connected_layer_sizes + [self.num_tasks])):
            if idx != len(self.fully_connected_layer_sizes):
                fc = nn.Sequential(Linear(L_in, L_out), activation_func(config), nn.Dropout(config['drop_ratio']))
                #L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)    
                if self.uncertainty: # for uncertainty 
                    self.uncertaintyLayers.append(NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer)) 
            else:
                fc = nn.Sequential(Linear(L_in, L_out), nn.Dropout(config['drop_ratio']))
            self.outLayers.append(fc)

        if self.uncertaintyMode == 'epistemic': 
            self.drop_mu = NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer) # working on last layer
        if self.uncertaintyMode == 'aleatoric': # 
            self.outLayers.append(last_fc) 

    def from_pretrained(self, model_file_solute, model_file_solvent):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn_solute.load_state_dict(torch.load(model_file_solute))
        self.gnn_hydrated_solute.load_state_dict(torch.load(model_file_solvent))


    def forward(self, data):
        solute_x, solute_edge_index, solute_edge_attr, batch = data.x, data.edge_index, data.edge_attr.long(), data.batch
        hyd_solute_x, hyd_solute_edge_index, hyd_solute_edge_attr, hyd_solute_batch, mask = data.hyd_solute_x, data.hyd_solute_edge_index, data.hyd_solute_edge_attr.long(), \
            data.hyd_solute_batch, data.hyd_solute_mask

        solute_node_representation = self.gnn_solute(solute_x, solute_edge_index, solute_edge_attr) # 22 * 64
        hydrated_solute_node_representation = self.gnn_hydrated_solute(hyd_solute_x, hyd_solute_edge_index, hyd_solute_edge_attr) # 22 * 64

        solute_representation = self.pool(solute_node_representation, batch)
        if self.dataset in ['logpWithWater']:
            solute_representation = self.pool(solute_node_representation[mask>0], batch[mask>0])
        hydrated_solute_representation = self.pool(hydrated_solute_node_representation[mask>0], hyd_solute_batch[mask>0])

        final_representation = torch.cat([solute_representation, hydrated_solute_representation], dim=1)

        for layer in self.outLayers:
             final_representation = layer(final_representation)
        if self.num_tasks > 1:
            return final_representation, None
        else:
            return final_representation.view(-1), None


class GNN_1_WithWater_simpler(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1_WithWater_simpler, self).__init__()
        self.config = config
        self.dataset = config['dataset']
        self.num_layer = config['num_layer']
        self.fully_connected_layer_sizes = config['fully_connected_layer_sizes']
        self.JK = config['JK']
        self.emb_dim = config['emb_dim']
        self.num_tasks = config['num_tasks']
        self.graph_pooling = config['pooling']
        self.propertyLevel = config['propertyLevel']
        self.gnn_type = config['gnn_type']
        self.gradCam = config['gradCam']
        self.tsne = config['tsne']
        self.uncertainty = config['uncertainty']
        self.uncertaintyMode = config['uncertaintyMode']
        self.weight_regularizer = config['weight_regularizer']
        self.dropout_regularizer = config['dropout_regularizer']
        self.features = config['mol_features']
        self.twoHop = config['twoHop']

        self.gnn_solute = GNN(config)
        self.gnn_hydrated_solute = GNN(config)
        self.outLayers = nn.ModuleList()
        if self.uncertainty:
            self.uncertaintyLayers = nn.ModuleList()
        if self.graph_pooling not in ['edge', 'topk', 'sag']: # except for edge pooling, coded here
            self.pool = PoolingFN(config) # after node embedding updating and pooling 
        
        #For graph-level property predictions
        if self.graph_pooling[:-1][0] == "set2set": # set2set will double dimension
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat": # change readout layers input and output dimension
            embed_size = self.mult * (self.num_layer + 1) * self.emb_dim
        elif self.graph_pooling == 'conv': # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim / 2  
        elif self.features:
            if self.dataset == 'sol_calc/ALL': # 208 total mol descriptors # total is 200
               embed_size = self.mult * self.emb_dim + 208
        else: # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim 

        for idx, (L_in, L_out) in enumerate(zip([embed_size] + self.fully_connected_layer_sizes, self.fully_connected_layer_sizes + [self.num_tasks])):
            if idx != len(self.fully_connected_layer_sizes):
                fc = nn.Sequential(Linear(L_in, L_out), activation_func(config), nn.Dropout(config['drop_ratio']))
                #L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)    
                if self.uncertainty: # for uncertainty 
                    self.uncertaintyLayers.append(NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer)) 
            else:
                fc = nn.Sequential(Linear(L_in, L_out), nn.Dropout(config['drop_ratio']))
                last_fc = fc
            self.outLayers.append(fc)
        #if self.propertyLevel == 'atomMol':
        #    self.outLayers1 = copy.deepcopy(self.outLayers) # another trainable linear layers

        if self.uncertaintyMode == 'epistemic': 
            self.drop_mu = NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer) # working on last layer
        if self.uncertaintyMode == 'aleatoric': # 
            self.outLayers.append(last_fc)  

    def from_pretrained(self, model_file_solute, model_file_solvent):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn_solute.load_state_dict(torch.load(model_file_solute))
        self.gnn_hydrated_solute.load_state_dict(torch.load(model_file_solvent))


    def forward(self, data):
        solute_x, solute_edge_index, solute_edge_attr, batch = data.x, data.edge_index, data.edge_attr.long(), data.batch
        hyd_solute_x, hyd_solute_edge_index, hyd_solute_edge_attr, hyd_solute_batch, mask = data.hyd_solute_x, data.hyd_solute_edge_index, data.hyd_solute_edge_attr.long(), \
            data.hyd_solute_batch, data.hyd_solute_mask

        solute_node_representation = self.gnn_solute(solute_x, solute_edge_index, solute_edge_attr) # 22 * 64
        hydrated_solute_node_representation = self.gnn_hydrated_solute(hyd_solute_x, hyd_solute_edge_index, hyd_solute_edge_attr) # 22 * 64

        #solute_representation = self.pool(solute_node_representation, batch)
        if self.dataset in ['logpWithWater']:
            solute_representation = self.pool(solute_node_representation[mask<1], batch[mask<1])
        #hydrated_solute_representation = self.pool(hydrated_solute_node_representation[mask<1], hyd_solute_batch[mask<1])
        
        if self.dataset in ['logpWithWater']:
            final_representation = torch.cat([solute_node_representation, hydrated_solute__node_representation], dim=1)
        else:
            batch = hyd_solute_batch
            final_representation = hydrated_solute_node_representation
        
        node_representation = final_representation
        # graph pooling 
        if self.propertyLevel in ['molecule', 'atomMol']: 
            if self.graph_pooling == 'conv':
                MolEmbed_list = []
                for p in self.pool:
                    MolEmbed_list.append(p(node_representation, batch))
                MolEmbed_stack = torch.stack(MolEmbed_list, 1).squeeze()
                MolEmbed = self.pool(MolEmbed_stack).squeeze()
            elif self.graph_pooling == 'edge':
                MolEmbed = global_mean_pool(node_representation, final_batch).squeeze()
            elif self.graph_pooling == 'atomic': # 
                MolEmbed = node_representation #(-1, emb_dim)
            else: # normal pooling functions besides conv and edge
                MolEmbed = self.pool(node_representation, batch)  # atomic read-out (-1, 1)
        if self.propertyLevel == 'atom':
            MolEmbed = node_representation #(-1, emb_dim)
        #elif self.propertyLevel in ['atom', 'atomMol']:
        #    MolEmbed = node_representation  # atomic read-out
        if self.features: # concatenating molecular features
            #print(data.features.shape, MolEmbed.shape)
            MolEmbed = torch.cat((MolEmbed, data.features.view(MolEmbed.shape[0], -1)), -1)
        if not self.training and not self.gradCam and self.tsne: # for TSNE analysis
            return node_representation, MolEmbed

        # read-out layers
        if not self.uncertainty:
            for layer in self.outLayers: # 
                MolEmbed = layer(MolEmbed) 
            #if self.dataset == 'solNMR' and self.propertyLevel == 'atomMol':
            #    for layer in self.outLayers1:
            #        node_representation = layer(node_representation)
            if self.num_tasks > 1:
                if self.dataset == 'solNMR':
                    assert MolEmbed.size(-1) == self.num_tasks
                    return MolEmbed[:,0].view(-1), self.pool(MolEmbed[:,1], batch).view(-1)
                if self.dataset == 'solEFGs':
                    assert MolEmbed.size(-1) == self.num_tasks
                    return MolEmbed[:,:-1], self.pool(MolEmbed[:,-1], batch).view(-1)
                else:
                    return MolEmbed, None
            elif self.propertyLevel == 'atom' and self.dataset != 'solNMR':
                return MolEmbed.view(-1,1), None
            elif self.propertyLevel == 'molecule' and self.graph_pooling == 'atomic':
                return MolEmbed.view(-1), self.pool(MolEmbed, batch).view(-1)
            else:
                return MolEmbed.view(-1), MolEmbed.view(-1)
            
        # for uncertainty analysis
        else:
            for layer, drop in zip(self.outLayers[1:-1], self.uncertaintyLayers):
                x, _ = drop(MolEmbed, layer)
            if self.config['uncertainty'] == 'epistemic':
                mean, regularization[-1] = self.drop_mu(x, self.outLayers[-1])
                return mean.squeeze()
            if self.config['uncertainty'] == 'aleatoric':
                mean = self.outLayers[-2](x)
                log_var = self.outLayers[-1](x)
                return mean, log_var

class GNN_1_interaction_old(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1_interaction_old, self).__init__()
        self.dataset = config['dataset']
        self.num_layer = config['num_layer']
        self.fully_connected_layer_sizes = config['fully_connected_layer_sizes']
        self.JK = config['JK']
        self.emb_dim = config['emb_dim']
        self.num_tasks = config['num_tasks']
        self.graph_pooling = config['pooling']
        self.propertyLevel = config['propertyLevel']
        self.gnn_type = config['gnn_type']
        self.gradCam = config['gradCam']
        self.tsne = config['tsne']
        self.uncertainty = config['uncertainty']
        self.uncertaintyMode = config['uncertaintyMode']
        self.weight_regularizer = config['weight_regularizer']
        self.dropout_regularizer = config['dropout_regularizer']
        self.features = config['mol_features']

        self.gnn_solute = GNN(config)
        if self.solvent == 'water': # to do adding to args
            self.gnn_solvent = nn.Sequential(nn.Linear(self.config['num_atom_features'], self.emb_dim),
                                            torch.nn.ReLU(), \
                                            nn.Linear(self.emb_dim, self.emb_dim))
        elif self.solvent == 'octanol':
            self.gnn_solvent = GNN(config)
        else:
            raise ValueError('Solvent need to be specified.')

        self.imap = nn.Linear(2*self.emb_dim, 1) # create a placeholder for interaction map
        self.outLayers = nn.ModuleList()
        #Different kind of graph pooling
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * self.emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(self.emb_dim, 1))
        elif self.graph_pooling == "set2set":
            set2set_iter = 2
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * self.emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(self.emb_dim, set2set_iter)
        elif self.graph_pooling == 'conv':
            self.pool = []
            self.pool.append(global_add_pool)
            self.pool.append(global_mean_pool)
            self.pool.append(global_max_pool)
            self.pool.append(GlobalAttention(gate_nn = torch.nn.Linear(self.emb_dim, 1)))
            self.pool.append(Set2Set(self.emb_dim, 2))
            self.convPool = nn.Conv1d(len(self.pool), 1, 2, stride=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if self.graph_pooling[:-1][0] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat": # change readout layers input and output dimension
            embed_size = self.mult * (self.num_layer + 1) * self.emb_dim
        elif self.graph_pooling == 'conv': # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim / 2  
        elif self.features:
            if self.dataset == 'sol_calc/ALL': # 208 total mol descriptors # total is 200
               embed_size = self.mult * self.emb_dim + 208
        else: # change readout layers input and output dimension
            embed_size = self.mult * self.emb_dim 

        for idx, (L_in, L_out) in enumerate(zip([embed_size] + self.fully_connected_layer_sizes, self.fully_connected_layer_sizes + [self.num_tasks])):
            if idx != len(self.fully_connected_layer_sizes):
                fc = nn.Sequential(Linear(L_in, L_out), activation_func(config), nn.Dropout(config['drop_ratio']))
                #L_in, L_out = self.outLayers[-1][0].out_features, int(self.outLayers[-1][0].out_features / 2)    
                if self.uncertainty: # for uncertainty 
                    self.uncertaintyLayers.append(NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer)) 
            else:
                fc = nn.Sequential(Linear(L_in, L_out), nn.Dropout(config['drop_ratio']))
                last_fc = fc
            self.outLayers.append(fc)

        if self.uncertaintyMode == 'epistemic': 
            self.drop_mu = NNDropout(weight_regularizer=self.weight_regularizer, dropout_regularizer=self.dropout_regularizer) # working on last layer
        if self.uncertaintyMode == 'aleatoric': # 
            self.outLayers.append(last_fc) 

    def from_pretrained(self, model_file_solute, model_file_solvent):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn_solute.load_state_dict(torch.load(model_file_solute))
        self.gnn_solvent.load_state_dict(torch.load(model_file_solvent))


    def forward(self, data):
        solute_batch, solute_x, solute_edge_index, solute_edge_attr, solute_length_matrix, solvent_batch, solvent_x, solvent_length_matrix = \
                data.batch, data.x, data.edge_index, data.edge_attr.long(), data.solute_length_matrix, \
                data.solvent_batch, data.solvent_x, data.solvent_length_matrix
        if self.solvent == 'octanol':
            solvent_edge_index, solvent_edge_attr = data.solvent_edge_index, data.solvent_edge_attr.long()

        solute_representation = self.gnn_solute(solute_x, solute_edge_index, solute_edge_attr) # 22 * 64
        if self.solvent == 'water':
            solvent_representation = self.gnn_solvent(solvent_x)
        if self.solvent == 'octanol':
            solvent_representation = self.gnn_solvent(solvent_x, solvent_edge_index, solvent_edge_attr) # 27 * 64
        #MolEmbed = self.pool(node_representation, batch)

        # Interaction part 
        len_map = torch.mm(solute_length_matrix.t(), solvent_length_matrix)  # interaction map to control which solvent mols  22*27
        #corresponds to which solute mol
        if 'dot' not in self.interaction: # to be adding to args
            X1 = solute_representation.unsqueeze(0) # 1*22*64
            Y1 = solvent_representation.unsqueeze(1) # 27*1*64
            X2 = X1.repeat(solvent_representation.shape[0], 1, 1) # 27*22*64
            Y2 = Y1.repeat(1, solute_representation.shape[0], 1) # 27*22*64
            Z = torch.cat([X2, Y2], -1) # 27*22*128

            if self.interaction == 'general':
                interaction_map = self.imap(Z).squeeze(2) # 27*22
            if self.interaction == 'tanh-general':
                interaction_map = torch.tanh(self.imap(Z)).squeeze(2)

            interaction_map = torch.mul(len_map.float(), interaction_map.t()) # 22*27
            ret_interaction_map = torch.clone(interaction_map)

        elif 'dot' in self.interaction:
            interaction_map = torch.mm(solute_representation, solvent_representation.t()) # interaction coefficient 22 * 27
            if 'scaled' in self.interaction:
                interaction_map = interaction_map / (np.sqrt(self.emb_dim))

            ret_interaction_map = torch.clone(interaction_map)
            ret_interaction_map = torch.mul(len_map.float(), ret_interaction_map) # 22 * 27
            interaction_map = torch.tanh(interaction_map) # 22*27
            interaction_map = torch.mul(len_map.float(), interaction_map) # 22 * 27

        solvent_prime = torch.mm(interaction_map.t(), solute_representation) # 27 * 64
        solute_prime = torch.mm(interaction_map, solvent_representation) # 22 * 64

        # Prediction
        solute_representation = torch.cat((solute_representation, solute_prime), dim=1) # 22 * 128
        solvent_representation = torch.cat((solvent_representation, solvent_prime), dim=1) # 27 * 128
        #print(solute_representation.shape)
        solute_representation = self.pool(solute_representation, solute_batch) # bs * 128
        solvent_representation = self.pool(solvent_representation, solvent_batch) # bs * 128
        #print(solute_representation.shape)
        final_representation = torch.cat((solute_representation, solvent_representation), 1) # bs * 256

        for layer in self.outLayers:
             final_representation = layer(final_representation)
        if self.num_tasks > 1:
            return final_representation, ret_interaction_map
        else:
            return final_representation.view(-1), ret_interaction_map

class DNN(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_hidden, activation, bn):
        super().__init__()
        self.bn = bn
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        self.hiddens = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(n_hidden)])
        self.bn_hiddens = nn.ModuleList([nn.BatchNorm1d(hidden_size) for i in range(n_hidden)])
        self.bn = nn.BatchNorm1d(hidden_size)
        # output layer
        self.linear2 = nn.Linear(hidden_size, out_size)
        self.activation = activation
        
    def forward(self, X, atom_mol_batch):
        out = self.linear1(X)
        for linear, bn in zip(self.hiddens, self.bn_hiddens):
            if self.bn:
                out = bn(out)
            out = self.activation(out)
            out = linear(out)
        if self.bn:
            out = self.bn(out)
        out = self.activation(out)
        out = self.linear2(out)
        out = scatter(reduce='add', src=out, index=atom_mol_batch, dim=0)
        return out


    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return 'processed.pt'
    
    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d['x'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                mol_y=d['mol_y'],
                Z=d['Z'],
                N=d['N'],
                #smiles=d['smiles'],
                ids=d['id']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#------------------Naive-------------------------------------

#------------------NMR--------------------------------------
class GraphDataset_atom(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(GraphDataset_atom, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return 'processed.pt'
    
    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d['x'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                atom_y=d['atom_y'],
                mask=d['mask'],
                Z=d['Z'],
                N=d['N'],
                id=d['ID']      
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#------------------NMR--------------------------------------------

#-----------------------PhysNet-------------------------------
class physnet(InMemoryDataset):
    #raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
    #           'molnet_publish/qm9.zip')
    #raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    #processed_url = 'https://pytorch-geometric.com/datasets/qm9_v2.zip'

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(physnet, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return 'processed.pt'

    def process(self):

        #with open(self.raw_paths[1], 'rb') as f: # modify this function TODO
        #    data = pickle.load(f)
        #all_ = pd.concat([data['train_df'], data['test_df']])
        #print(all_['molecule_id'].tolist())
        
        #suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
        #                           sanitize=False)

        data_list = []
        raw_data_list = torch.load(self.raw_paths[0])
        for _, value in raw_data_list.items():
            mol = value[0]
            #name = mol.GetProp('_Name')
            #print(name)
            N = mol.GetNumAtoms()
            #print(N)
            N_ = torch.tensor(N).view(-1)

            #pos = mol.GetItemText(i).split('\n')[4:4 + N]
            pos = []
            for i in range(N):
                position = mol.GetConformer().GetAtomPosition(i) 
                pos.append([position.x, position.y, position.z])
            pos = torch.tensor(pos, dtype=torch.float)

            atomic_number = []
            for atom in mol.GetAtoms():
                atomic_number.append(atom.GetAtomicNum())
            z = torch.tensor(atomic_number, dtype=torch.long)
            
            mask = np.zeros((N, 1), dtype=np.float32)
            vals = np.zeros((N, 1), dtype=np.float32)
            for k,v in value[1][0].items():
                mask[int(k), 0] = 1.0
                vals[int(k), 0] = v
            #mol_y = torch.FloatTensor([value[1]]).flatten() #for calculated solvation energy
            #data = Data(R=pos, Z=z, mol_sol_wat=mol_y, N=N_) # for calculated solvation energy
            atom_y = torch.FloatTensor(vals).flatten()
            mask = torch.FloatTensor(mask).flatten()
            data = Data(R=pos, Z=z, atom_y=atom_y, N=N_, mask=mask) # for calculated solvation energy

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            data_edge = self.pre_transform(data, edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                         cutoff=10.0, boundary_factor=100., use_center=True, mol=None, cal_3body_term=False,
                                         bond_atom_sep=False, record_long_range=True)
            data_list.append(data_edge)

        torch.save(self.collate(data_list), self.processed_paths[0])
#---------------------------------------------PhysNet------------------------------------------------

#---------------------------------------------PhysNet_NMR------------------------------------------------
class physnet_nmr(InMemoryDataset):
    #raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
    #           'molnet_publish/qm9.zip')
    #raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    #processed_url = 'https://pytorch-geometric.com/datasets/qm9_v2.zip'

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(physnet_nmr, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['nmr.sdf', 'nmrshiftdb.pickle']

    @property
    def processed_file_names(self):
        return 'processed.pt'

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
                    data = Data(R=pos, Z=z, atom_y=y, mask=torch.FloatTensor(mask).view(-1), N=N_, idx=int(name))
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
#---------------------------------------------PhysNet_NMR------------------------------------------------

#---------------------------------------------QM9-NMR------------------------------------------------
class QM9_nmr(InMemoryDataset):
    #raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
    #           'molnet_publish/qm9.zip')
    #raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    #processed_url = 'https://pytorch-geometric.com/datasets/qm9_v2.zip'

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(QM9_nmr, self).__init__(root, transform, pre_transform, pre_filter)
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
#-------------------------------------------QM9-NMR--------------------------------------------------

#-------------------------------------Interaction----------------------------------------------------
def collate_WithWater(data_list):
        keys = data_list[0].keys
        assert 'batch' not in keys

        batch = Batch()
        for key in keys:
            batch[key] = []
        batch.batch = []
        batch.num_nodes = []
        batch.solute_length_matrix = []

        if 'hyd_solute_x' in keys:
            batch.hyd_solute_batch = []
            batch.hyd_solute_num_nodes = []

        if 'hyd_solute_edge_index' in keys:
            keys.remove('hyd_solute_edge_index')
        if 'edge_index' in keys:
            keys.remove('edge_index')

        props = [
            'edge_index_2', 'assignment_index_2', 'edge_index_3',
            'assignment_index_3', 'assignment_index_2to3'
        ]
        keys = [x for x in keys if x not in props]

        cumsum_1 = N_1 = cumsum_2 = N_2 = cumsum_3 = N_3 = 0

        for i, data in enumerate(data_list):
            for key in keys:
                batch[key].append(data[key])

            N_1 = data.x.shape[0]
            #print(N_1)
            batch.edge_index.append(data.edge_index + cumsum_1)
            batch.batch.append(torch.full((N_1, ), i, dtype=torch.long))
            batch.num_nodes.append(N_1)

            if 'hyd_solute_x' in data:
                N_2 = data.hyd_solute_x.shape[0]
                batch.hyd_solute_num_nodes.append(N_2)
                batch.hyd_solute_batch.append(torch.full((N_2, ), i, dtype=torch.long))
                if 'hyd_solute_edge_index' in data:
                    batch.hyd_solute_edge_index.append(data.hyd_solute_edge_index + cumsum_2)

            cumsum_1 += N_1
            cumsum_2 += N_2

        keys = [x for x in batch.keys if x not in ['batch', 'hyd_solute_batch', 'solute_length_matrix', 'hyd_solute_length_matrix']]

        for key in keys:
            if torch.is_tensor(batch[key][0]):
                batch[key] = torch.cat(
                    batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        batch.batch = torch.cat(batch.batch, dim=-1)
        if 'hyd_solute_x' in data:
            batch.hyd_solute_batch = torch.cat(batch.hyd_solute_batch, dim=-1)
        return batch.contiguous()

class knnGraph_WithWater(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph_WithWater, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['solute_temp.pt', 'hydrated_solute_temp.pt']

    @property
    def processed_file_names(self):
        return '1-GNN-withWater.pt'

    def download(self):
        pass

    def process(self):
        raw_solute_data = torch.load(self.raw_paths[0])
        raw_hydrated_solute_data = torch.load(self.raw_paths[1])

        data_list = [
            Data(
                x=solute['x'],
                edge_index=solute['edge_index'],
                edge_attr=solute['edge_attr'],
                mol_y=solute['y'],
                hyd_solute_x=hyd_solute['x'],
                hyd_solute_edge_index=hyd_solute['edge_index'],
                hyd_solute_edge_attr=hyd_solute['edge_attr'],
                hyd_solute_mask=hyd_solute['mask']

                ) for solute, hyd_solute in zip(raw_solute_data, raw_hydrated_solute_data)
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class DataLoader_WithWater(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        super(DataLoader_WithWater, self).__init__(dataset, collate_fn=collate_WithWater, **kwargs)
#---------------------------------------- Interaction ----------------------------------------------

class MyPreTransform_twoHop(object):
    def __call__(self, data):
        G = to_networkx(data)
        adj = adjacency_matrix(G).toarray()
        adj2 = adj.dot(adj)
        np.fill_diagonal(adj2, 0)
        data.edge_index_twoHop = torch.tensor(np.array(np.where(adj2 == 1))).long()

        D = euclidean_distances(data.pos)
        rbf = gaussian_rbf(torch.tensor(D).view(-1,1), centers, widths, cutoff, return_dict=True)['rbf'].view(D.shape[0], -1, 64)
        data.edge_attr_twoHop = rbf[data.edge_index_twoHop[0,], data.edge_index_twoHop[1,], :]
        return data

class MyPreTransform_EFGS(object):
    def __call__(self, data):
        x = data.x
        data.x = data.x[:, :10]
        data = TwoMalkin()(data)
        data.x = x
        data = smiles2gdata(data)

        return data

class MyPreTransform_centrality(object):
    def __call__(self, data):
        G = to_networkx(data)
        centrality = torch.FloatTensor(list(betweenness_centrality(G, k=int(data.N.numpy()[0])).values())).view(-1, 1)
        data.x = torch.cat((data.x, centrality), dim=1)
        return data

class MyPreTransform(object):
    def __call__(self, data):
        x = data.x
        data.x = data.x[:, :10] # 10 because first 10 bits are element types
        data = TwoMalkin()(data)
        #data 1= ConnectedThreeMalkin()(data)
        data.x = x
        
        return data

def smiles2gdata(data):
    mol = Chem.MolFromSmiles(data.smiles)
    #mol = Chem.AddHs(mol)

    a,b ,c, d = mol2frag(mol, returnidx=True, vocabulary=list(vocab), toEnd=True, extra_included=True, TreatHs='include', isomericSmiles=False)
    ass_idx = {x:i for i,t in enumerate(c+d) for x in t}
    ei2 = []

    for bond in mol.GetBonds():
        if ass_idx[bond.GetBeginAtomIdx()] == ass_idx[bond.GetEndAtomIdx()]: continue
        groupA, groupB = ass_idx[bond.GetBeginAtomIdx()], ass_idx[bond.GetEndAtomIdx()]
        ei2.extend([[groupA, groupB],[groupB, groupA]])
    if ei2:
        data.edge_index_3 = torch.LongTensor(ei2).transpose(0,1)
    else:
        data.edge_index_3 = torch.LongTensor()

    vocab_index = torch.LongTensor([list(vocab).index(x) for x in a + b])
    data.assignment_index_3 = torch.LongTensor([[key,value] for key, value in ass_idx.items()]).transpose(0,1)
    data.iso_type_3 = F.one_hot(vocab_index, num_classes=len(vocab)).to(torch.float)

    del data.smiles

    return data

def _cutoff_fn(D, cutoff):
    """
    Private function called by rbf_expansion(D, K=64, cutoff=10)
    """
    x = D / cutoff
    x3 = x ** 3
    x4 = x3 * x
    x5 = x4 * x

    result = 1 - 6 * x5 + 15 * x4 - 10 * x3
    return result

def gaussian_rbf(D, centers, widths, cutoff, return_dict=False):
    """
    The rbf expansion of a distance
    Input D: matrix that contains the distance between to atoms
          K: Number of generated distance features
    Output: A matrix containing rbf expanded distances
    """

    rbf = _cutoff_fn(D, cutoff) * torch.exp(-widths * (torch.exp(-D) - centers) ** 2)
    if return_dict:
        return {"rbf": rbf}
    else:
        return rbf

class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes > 1  # Remove graphs with less than 1 nodes.

#vocab = pickle.load(open('/scratch/dz1061/gcn/datasets/EFGS/vocab/ours/ours_vocab.pt', 'rb'))

'''
Periodic properties: (in order as)
Element period #, element group #, atomic weight, VDW atomic radius, 
covalent atomic radius, ionization energy, electronegativity, 
election affinity
'''
ATOM_FEATURES_PERIODIC = {

    'H': [-1.460954261,-2.88512737,-0.8978528170000001,-1.953788528,-1.809863616,0.535333521,-1.020768417,-0.791439307],
    'C': [-0.568148879,-0.07850686700000001,-0.601798994,0.112718569,-0.357866064,-0.396995828,-0.425320174,-0.41149686700000004],
    'B': [-0.568148879,-0.294400752,-0.6341874829999999,1.021981691,-0.099733165,-1.578159015,-1.292973329,-1.147495385],
    'N': [-0.568148879,0.137387018,-0.548064941,-0.50723356,-0.519199125,0.908584278,0.40830736700000003,-1.3542617030000001],
    'O': [-0.568148879,0.353280902,-0.494444716,-0.6312239860000001,-0.680532186,0.543308965,1.088819645,-0.264446453],
    'S': [0.324656502,0.353280902,-0.06227609,0.5260199879999999,0.577865693,-0.755890787,-0.374281753,0.19611245800000002],
    'F': [-0.568148879,0.569174787,-0.413743398,-1.333836399,-0.970931697,2.060637097,2.007511221,1.138130846],
    'Cl': [0.324656502,0.569174787,0.02886699,0.319369279,0.481065856,0.28410705,0.61246105,1.345643613],
    'Br': [1.217461884,0.569174787,1.225136739,0.650010414,1.061864877,-0.176076042,0.272204911,1.157538515],
    'P': [0.324656502,0.137387018,-0.091500001,0.5260199879999999,0.6423989170000001,-0.70524672,-1.0377812240000002,-0.797410897],
    'I': [2.110267265,0.569174787,2.48986471,1.2699625429999999,1.674930511,-0.719602519,-0.23817929699999998,0.929125181]
}
