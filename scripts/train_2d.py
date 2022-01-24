from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
import argparse, os, pickle
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd 
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors

class RDKit_2D:
    def __init__(self, smi):
        self.mols = [Chem.MolFromSmiles(i) for i in smi]
        self.smiles = smi
        
    def compute_2Drdkit(self):
        rdkit_2d_desc = []
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        header = calc.GetDescriptorNames()
        for i in range(len(self.mols)):
            ds = calc.CalcDescriptors(self.mols[i])
            rdkit_2d_desc.append(ds)
        df = pd.DataFrame(rdkit_2d_desc,columns=header)
        df.insert(loc=0, column='SMILES', value=self.smiles)
        #df.to_csv(name[:-4]+'_RDKit_2D.csv', index=False)
        return df

def ExplicitBitVect_to_NumpyArray(bitvector):
    bitstring = bitvector.ToBitString()
    intmap = map(int, bitstring)
    return np.array(list(intmap))

def get_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='2D')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--allDataPath', type=str, default='/scratch/dz1061/gcn/chemGraph/data')
    parser.add_argument('--running_path', type=str, help='path to save model', default='/scratch/dz1061/gcn/chemGraph/results') 
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--style', type=str)  # fp or 2d
    return parser.parse_args()

def getMol2D(x):
    rdkit_2d = RDKit_2D(x['SMILES'])
    df = rdkit_2d.compute_2Drdkit()
    #df['target'] = x['target']
    x_use = df
    
    return x_use

def getMorgan(df):
    df_X = []
    for smi in df['SMILES']:
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
        df_X.append(ExplicitBitVect_to_NumpyArray(fp))
    return df_X



def main():
    args = get_parser()
    data_path = os.path.join(args.allDataPath, args.dataset, 'split')
    running_path = os.path.join(args.running_path, args.dataset, 'XGBoost', args.style, args.experiment) 
    if not os.path.exists(running_path):
        os.makedirs(running_path)

    SPACE = {'max_depth': hp.choice('max_depth', [4, 6, 8, 10, 12]),
         'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400,500,600]),
         'learning_rate': hp.choice('learning_rate', [0.001, 0.005, 0.01, 0.05, 0.1]),
         'colsample_bytree': hp.choice('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
         'subsample': hp.choice('subsample', [0.6, 0.7, 0.8, 0.9, 1.0])
         }

    def objective(space):
        #from sklearn.model_selection import cross_val_score
        model = XGBRegressor(max_depth=space['max_depth'], 
                            n_estimators=space['n_estimators'],
                            learning_rate=space['learning_rate'],
                            colsample_bytree=space['colsample_bytree'],
                            subsample=space['subsample'])
        #mse = cross_val_score(model , data['train_X'], data['train_Y'], \
        #                scoring="neg_mean_squared_error").mean()
        model.fit(data['train_X'], data['train_Y'])
        pred_y = model.predict(data['valid_X'])
        rmse = np.sqrt(mean_squared_error(data['valid_Y'], pred_y))

        return {'loss': rmse, 'status': STATUS_OK}
        
    data = {}
    #data[]
    train_ = pd.read_csv(os.path.join(data_path, 'train.csv'))
    valid_ = pd.read_csv(os.path.join(data_path, 'valid.csv'))
    test_ = pd.read_csv(os.path.join(data_path, 'test.csv'))

    if args.style == 'fp': 
        for df, df_new in zip([train_, valid_, test_], ['train_', 'valid_', 'test_']):
            assert 'SMILES' in df.columns
            data[df_new + 'X'] = np.array(getMorgan(df))
            data[df_new + 'Y'] = df['target'].to_numpy()
        all_x = pd.DataFrame(data['train_X'].tolist() + data['valid_X'].tolist() + data['test_X'].tolist())
        all_y = pd.DataFrame(train_['target'].tolist() + valid_['target'].tolist() + test_['target'].tolist())
        all_df = pd.concat([all_x, all_y], axis=1)

    if args.style == '2d':
        for df, df_new in zip([train_, valid_, test_], ['train_', 'valid_', 'test_']):
            assert 'SMILES' in df.columns
            assert 'target' in df.columns
            data[df_new + 'X'] = np.array(getMol2D(df).iloc[:, 1:-1])
            data[df_new + 'Y'] = df['target'].to_numpy()
        all_x = pd.DataFrame(data['train_X'].tolist() + data['valid_X'].tolist() + data['test_X'].tolist())
        all_y = pd.DataFrame(train_['target'].tolist() + valid_['target'].tolist() + test_['target'].tolist())
        all_df = pd.concat([all_x, all_y], axis=1)
    
    trials = Trials()
    best_hyperparams = fmin(fn=objective, space=SPACE, algo=tpe.suggest, \
                            max_evals=50, trials=trials, verbose=False)
    
    best_ = space_eval(SPACE, best_hyperparams)
    #print(best_)
    metrics = defaultdict(list)
    for seed in [0, 1, 13, 31, 123]:
        train, test = train_test_split(all_df, test_size=0.1, random_state=seed)
        best_model = XGBRegressor(**best_)
        best_model.fit(np.array(train.iloc[:, :-1]), train.iloc[:, -1].to_numpy())
        
        train_pred, test_pred = best_model.predict(np.array(train.iloc[:, :-1])), best_model.predict(np.array(test.iloc[:, :-1]))
        train_rmse = np.sqrt(mean_squared_error(train.iloc[:, -1].to_numpy(), train_pred))
        test_rmse = np.sqrt(mean_squared_error(test.iloc[:, -1].to_numpy(), test_pred))
        metrics['train'].append(train_rmse)
        metrics['test'].append(test_rmse)

    results = {}
    results['best_params'] = best_
    results['performance'] = metrics
    results['train_size'] = train.shape[0]
    results['test_size'] = test.shape[0]

    with open(os.path.join(running_path, 'results.pickle'), 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    #cycle_index(10,2)
    main()