from torch import nn
from torch_geometric.nn.conv import GCNConv, SAGEConv, GraphConv, ResGatedGraphConv, GATConv, GINConv, GINEConv, EdgeConv, NNConv, PNAConv
from supergat_conv import SuperGATConv

def get_gnn(config):
    name = config['gnn_type']

    if name == 'gcn':
        return gcn_conv
    if name == 'sage':
        return sage_conv(config)
    if name == 'graphconv':
        return graph_conv(config)
    if name == 'resgatedgraphconv':
        return res_gated_graph_conv(config)
    if name == 'gatconv':
        return gat_conv(config)
    if name == 'ginconv':
        return gin_conv(config)
    if name == 'gineconv':
        return gine_conv(config)
    if name == 'edgeconv':
        return edge_conv(config)
    if name == 'supergat':
        return sgat_conv(config)
    if name == 'pnaconv':
        return pna_conv(config)
    if name == 'nnconv':
        return nn_conv(config)
    

class gcn_conv():
    # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
    def __init__(self, config):
        super(gcn_conv, self).__init__()
        self.emb_dim = config['emb_dim'] 
        self.aggr = config['aggregate'] # default is add
        
    def model(self):
        # input: x(-1, emb_dim), edge index
        # return: x(-1, emb_dim)
        return GCNConv(in_channels=self.emb_dim, out_channels=self.emb_dim, aggr=self.aggr)

class sage_conv():
    # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv
    def __init__(self, config):
        super(sage_conv, self).__init__()
        self.emb_dim = config['emb_dim']
        self.aggr = config['aggregate'] # default is mean
        
    def model(self):
        # input: x(-1, emb_dim), edge index
        # return: x(-1, emb_dim)
        return SAGEConv(in_channels=self.emb_dim, out_channels=self.emb_dim, aggr=self.aggr)

class graph_conv():
    #https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv
    def __init__(self, config):
        super(graph_conv, self).__init__()
        self.emb_dim = config['emb_dim']
        self.aggr = config['aggregate'] # default is add
        
    def model(self):
        # input: x(-1, emb_dim), edge index 
        # return: x(-1, emb_dim)
        return GraphConv(in_channels=self.emb_dim, out_channels=self.emb_dim, aggr=self.aggr)

class res_gated_graph_conv():
    #https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ResGatedGraphConv
    def __init__(self, config):
        super(res_gated_graph_conv, self).__init__()
        self.emb_dim = config['emb_dim']
        self.aggr = config['aggregate'] # default is add
        
    def model(self):
        # input: x(-1, emb_dim), edge index
        # # return: x(-1, emb_dim)
        return ResGatedGraphConv(in_channels=self.emb_dim, out_channels=self.emb_dim, aggr=self.aggr)

class gat_conv():
    #https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv
    def __init__(self, config):
        super(gat_conv, self).__init__()
        self.emb_dim = config['emb_dim']
        self.aggr = config['aggregate'] # default is add
        
    def model(self):
        # input: x(-1, emb_dim), edge index 
        # return: x(-1, emb_dim)
        return GATConv(in_channels=self.emb_dim, out_channels=self.emb_dim, aggr=self.aggr) # n_heads

class gin_conv():
    #https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINConv
    def __init__(self, config):
        super(gin_conv, self).__init__()
        self.emb_dim = config['emb_dim']
        self.aggr = config['aggregate'] # default is add
        self.nn = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim), nn.ReLU())
        
    def model(self):
        # input: x(-1, emb_dim), edge index 
        # return: x(-1, emb_dim)
        return GINConv(nn=self.nn, train_eps=True)

class gine_conv():
    # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINEConv
    def __init__(self, config):
        super(gine_conv, self).__init__()
        self.emb_dim = config['emb_dim']
        self.aggr = config['aggregate'] # default is add 
        self.nn = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim), nn.ReLU())
        
    def model(self):
        # input: x(-1, emb_dim), edge(-1, emb_dim), edge index
        # edge need be converted from initial num bond feature to emb_dim before being fed 
        # return x(-1, emb_dim)
        return GINEConv(nn=self.nn, train_eps=True)

class edge_conv():
    def __init__(self, config):
        #https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.EdgeConv
        super(edge_conv, self).__init__()
        self.emb_dim = config['emb_dim']
        self.aggr = config['aggregate'] # default is max
        self.nn = nn.Sequential(nn.Linear(2*self.emb_dim, self.emb_dim), nn.ReLU())
        
    def model(self):
        # input: x(-1, emb_dim), edge index
        # return: x(-1, emb_dim)
        return EdgeConv(nn=self.nn)

class sgat_conv():
    def __init__(self, config):
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SuperGATConv
        super(sgat_conv, self).__init__()
        self.emb_dim = config['emb_dim']
        self.aggr = config['aggregate'] # default is add
        
    def model(self):
        # input: x(-1, emb_dim), edge inex
        # return: x(-1, emb_dim)
        return SuperGATConv(self.emb_dim, int(self.emb_dim/8), heads=8, attention_type='MX',
                                  edge_sample_ratio=0.8, is_undirected=True)

class pna_conv():
    # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.PNAConv
    def __init__(self, config):
        super(pna_conv, self).__init__()
        self.aggregators = ['mean', 'min', 'max', 'std']
        self.scalers = ['identity', 'amplification', 'attenuation']
        self.emb_dim = config['emb_dim']
        self.aggr = config['aggregate']

    def model(self):
        # input: x(-1, emb_dim), bond(-1, emb_dim), edge index, deg
        # return: x(-1, emb_dim)
        return PNAConv(self.emb_dim, self.emb_dim, aggregators=self.aggregators, scalers=self.scalers, deg=deg,
                           edge_dim=self.emb_dim, towers=4, pre_layers=1, post_layers=1,
                           divide_input=False)

class nn_conv():
    def __init__(self, config):
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.NNConv
        super(nn_conv, self).__init__()
        self.emb_dim = config['emb_dim']
        self.aggr = config['aggregate'] # default is add
        #self.atom_features = config['num_atom_features']
        #self.bond_features = config['num_bond_features']
        self.nn = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim*self.emb_dim), nn.ReLU())
        
    def model(self):
        # input: intial atom/bond features, and edge index
        return NNConv(in_channels=self.emb_dim, out_channels=self.emb_dim, nn=self.nn)