# Author: Markus Viljanen
import sys
print(sys.argv)
model = sys.argv[1]
setting = sys.argv[2]
#tree_depth = int(sys.argv[3])
#l2 = float(sys.argv[4])
#gamma = float(sys.argv[4])
#subsample = float(sys.argv[6])
#num_round = int(sys.argv[7])
fold = int(sys.argv[3])

# Imports
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
import random 

hyperparameters = {(row['setting'],row['model'],row['fold']): {'num_round': row['nrounds'], 
                   'max_depth':row['tree_depth'], 'lambda':row['l2'], 'gamma':row['gamma'], 'subsample': row['subsample']} 
                   for i,row in pd.read_csv('XGB_optimal.csv').iterrows()} 

# Toxicity data
df = pd.read_csv('df_tox.csv')
df['Species'] = df['Species'].astype('category')
df['SMILES'] = df['SMILES'].astype('category')

# Fingerprint
fingerprints = pd.read_csv('fingerprints.csv') #'fingerprints2.csv'
fingerprints.set_index('SMILES', inplace=True)
fingerprints = np.log(1 + fingerprints)
fingerprints = fingerprints/fingerprints.max()
fingerprints.reset_index(inplace=True)

# Merge
feature_columns = ["fp_%d" % i for i in range(1024)]
df = df.merge(fingerprints, on='SMILES', how='left', sort=False)

fn = "/mnt/scratch_dir/viljanem/predictions/XGB_predictions.csv"

y_pred = None

params = hyperparameters[(setting, model, fold)]
test_fold = df["{setting}_test{fold}".format(setting=setting, fold=fold)]
train = df.index.values[~test_fold]
test = df.index.values[test_fold]

if model == "XGB":
    rmses = []
    train_buffer = "/mnt/scratch_dir/viljanem/xlearn_data/train_{setting}_{fold}.xgb".format(setting=setting, fold=fold)
    test_buffer = "/mnt/scratch_dir/viljanem/xlearn_data/test_{setting}_{fold}.xgb".format(setting=setting, fold=fold)
    test_fn = "/mnt/scratch_dir/viljanem/xlearn_data/test_{setting}_{fold}.svm".format(setting=setting, fold=fold)

    dtrain = xgb.DMatrix(train_buffer)
    dtest = xgb.DMatrix(test_buffer)

    param = {'objective': 'reg:squarederror'} #'max_depth':tree_depth, 'lambda':l2, 'gamma':gamma, 'subsample': subsample} 
    param.update(params)
    bst = xgb.train(param, dtrain, params['num_round'])
    y_pred = bst.predict(dtest)
    y_test = np.loadtxt(test_fn, delimiter=" ", usecols=0)

if not y_pred is None:
    save = pd.DataFrame({"model":model, "setting":setting, "fold": fold, "index":test, "y_pred":y_pred, "y": y_test})
    save.to_csv(fn, header=False, index=False, mode='a')


