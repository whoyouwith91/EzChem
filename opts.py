import os 

BASIC = ['dataset', 'allDataPath', 'running_path', 'model', 'normalize', 'batch_size', 'dimension', \
         'dropout', 'act_fn' , 'weights', 'seed', 'optimizer', 'loss', 'metrics', 'lr', 'lr_style', \
         'epochs', 'early_stopping', 'train_type', ]
GNN = ['depths', 'NumOutLayers', 'pooling']
GNNVariants = ['efgs', 'water_interaction', 'InterByConcat', 'InterBySub', 'mol']
VAE = ['vocab', 'numEncoLayers', 'numDecoLayers', 'numEncoders', 'numDecoders', 'varDimen']
UQ = ['uncertainty', 'uncertainty_method', 'swag_start']
TRANSFER = ['transfer_from', 'pre_trained_path', 'pre_trained_model']