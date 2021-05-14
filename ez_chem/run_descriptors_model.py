import sys, os
import torch
import xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from prettytable import PrettyTable

class descriptorLoader():
    def __init__(self, config):
        self.all_ = torch.load(config['data_path'])
        self.train_size = config['train_size']
        self.valid_size = config['valid_size']
        self.norm = config['normFeatures']
    
    def transform(self):
        all_rev = [s['descriptors'] for s in self.all_]
        y_rev = [s['y'] for s in self.all_]

        self.df_all = pd.DataFrame(all_rev)
        self.df_all['y'] = y_rev
        self.df_all = self.df_all.astype(float)

        if self.df_all.isna().values.any():
            self.df_all = self.df_all.dropna().reset_index(drop=True)

        use_col = [col for col in self.df_all.columns if col != 'y']
        self.X_all = self.df_all[use_col]
        self.X_all.columns = descriptors
        self.y_all = self.df_all['y']
    
    def dataSplit(self):
        X_train, X_valid, X_test = self.X_all[:self.train_size], self.X_all[self.train_size:self.train_size + self.valid_size], self.X_all[self.train_size + self.valid_size:]
        Y_train, Y_valid, Y_test = self.y_all[:self.train_size], self.y_all[self.train_size:self.train_size + self.valid_size], self.y_all[self.train_size + self.valid_size:]
        if self.norm:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_valid = sc.transform(X_valid)
            X_test = sc.transform(X_test)
        return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)


#The model
class Model(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""
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
        
    def forward(self, X):
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
        #print(out.shape)
        return out

def train(model, loader, device):
    all_loss = 0
    model.train()
    for X, y in loader:
        optimizer.zero_grad()
        pred = model(X.to(device))

        loss = F.mse_loss(pred.reshape(-1), y.to(device))
        all_loss += loss.item()*X.shape[0]

        loss.backward()
        optimizer.step()
    return (all_loss / len(loader.dataset)).sqrt()
        
        
def test(model, loader, device):
    with torch.no_grad():
        model.eval()
        all_loss = 0
        for X, y in loader:
            pred = model(X.to(device))
            loss = F.mse_loss(pred.reshape(-1), y.to(device))
            all_loss += loss.item()*X.shape[0]
        return np.sqrt(all_loss/len(loader.dataset))

config = {
    'data_path': '/scratch/dz1061/gcn/chemGraph/data/sol_calc/ALL/descriptors/base/ML/raw/temp.pt',
    'train_size': 80000,
    'valid_size': 10000,
    'normFeatures': True
}

from rdkit.Chem import Descriptors
descriptors = list(np.array(Descriptors._descList)[:,0])

loader = descriptorLoader(config)
loader.transform()
used_descriptors = torch.load()
X_all_use = loader.X_all[used_descriptors]

loader.X_all = X_all_use
train_rev, valid_rev, test_rev = loader.dataSplit()

train_dataset = torch.utils.data.TensorDataset(torch.as_tensor(np.array(train_rev[0])).float(), torch.as_tensor(np.array(train_rev[1])).float())
valid_dataset = torch.utils.data.TensorDataset(torch.as_tensor(np.array(valid_rev[0])).float(), torch.as_tensor(np.array(valid_rev[1])).float())
test_dataset = torch.utils.data.TensorDataset(torch.as_tensor(np.array(test_rev[0])).float(), torch.as_tensor(np.array(test_rev[1])).float())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

device = torch.device('cuda')

model = Model(train_rev[0].shape[1], hidden_size=512, out_size=1, n_hidden=3, activation=F.leaky_relu, bn=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
header = ['Epoch', 'LR', 'Train RMSE', 'Valid RMSE', 'Test RMSE']
x = PrettyTable(header)

for epoch in range(300):
    lr = 0.001
    train_rmse = train(model, train_loader, device)
    valid_rmse = (test(model, valid_loader, device)
    test_rmse = test(model, test_loader, device)
    x.add_row([str(epoch), lr, train_rmse, valid_rmse, test_rmse])
    print(x)
    sys.stdout.flush()


