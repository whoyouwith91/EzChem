import sys, os
import time, random, pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

data = pickle.load(open('/scratch/dz1061/gcn/chemGraph/data/deepchem/delaney/split/molNet/2D/delaney.pickle', 'rb'))

SPACE = {'max_depth': hp.choice('max_depth', [4, 6, 8, 10, 12]),
         'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400,500,600]),
         'learning_rate': hp.choice('learning_rate', [0.001, 0.005, 0.01, 0.05, 0.1]),
         'colsample_bytree': hp.choice('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
         'subsample': hp.choice('subsample', [0.6, 0.7, 0.8, 0.9, 1.0])
         }

def objective(space):
    model = XGBRegressor(max_depth=space['max_depth'], 
                         n_estimators=space['n_estimators'],
                         learning_rate=space['learning_rate'],
                         colsample_bytree=space['colsample_bytree'],
                         subsample=space['subsample'])
    model.fit(data['train_X'], data['train_Y'])
    pred_y = model.predict(data['valid_X'])
    rmse = np.sqrt(mean_squared_error(data['valid_Y'], pred_y))

    return {'loss': rmse, 'status': STATUS_OK}

trials = Trials()
best_hyperparams = fmin(fn=objective, space=SPACE, algo=tpe.suggest, \
                        max_evals=300, trials=trials)

    