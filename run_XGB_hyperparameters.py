# Author: Markus Viljanen
import sys
print(sys.argv)
model = sys.argv[1]
setting = sys.argv[2]
tree_depth = int(sys.argv[3])
l2 = float(sys.argv[4])
gamma = float(sys.argv[5])
subsample = float(sys.argv[6])
fold = int(sys.argv[7])
# tree depth 2, 3, 4, 5 (default 6)
# lambda 0 0.1 1 10 100 (default 1)
# gamma 0 0.1 1 10 (default 0)
# subsample 0.25 0.5 0.75 1.0 (default 1.0)

# Imports
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
import random 

num_round = 3000

fn = "/mnt/scratch_dir/viljanem/hyperparameters/XGB_hyperparameters.csv"

ncv  = 5

rmse = None

if model == "XGB":
    rmses = {}
    for i in range(ncv):
        print("="*20 + "Fold {fold}".format(fold=i+1) + "="*20, end="\n\n\n")
        train_buffer = "/mnt/scratch_dir/viljanem/xlearn_fold1/train_{setting}_{fold}_{inner}.xgb".format(setting=setting, fold=fold, inner=i+1)
        test_buffer = "/mnt/scratch_dir/viljanem/xlearn_fold1/test_{setting}_{fold}_{inner}.xgb".format(setting=setting, fold=fold, inner=i+1)
        test_fn = "/mnt/scratch_dir/viljanem/xlearn_fold1/test_{setting}_{fold}_{inner}.xgb".format(setting=setting, fold=fold, inner=i+1)

        dtrain = xgb.DMatrix(train_buffer)
        dtest = xgb.DMatrix(test_buffer)

        param = {'objective': 'reg:squarederror', 'eval_metric':'rmse', 'max_depth':tree_depth, 
                 'lambda':l2, 'gamma':gamma, 'subsample': subsample} #, 'tree_method': 'hist'
        evallist = [(dtest, 'eval')]#, (dtrain, 'train')
        #bst = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=100)
        #bst = xgb.train(param, dtrain, bst.best_iteration)
        #y_test = np.loadtxt(test_fn, delimiter=" ", usecols=0)
        #y_pred = bst.predict(dtest)
        #rmse = mean_squared_error(y_test, y_pred, squared=False) 
        #rmses.append(rmse)
        evals_result = {}
        bst = xgb.train(param, dtrain, num_round, evals=evallist, evals_result=evals_result, early_stopping_rounds=50)
        rmses["fold_{fold}".format(fold=i+1)] = pd.Series(evals_result['eval']['rmse'], dtype=float)
    rmses = pd.DataFrame(rmses)
    rmses.index = np.arange(len(rmses)) + 1
    rmses = rmses.mean(axis=1)
    nrounds = rmses.idxmin()
    rmse = rmses.min()
    #print("nrounds optimal: {nrounds}".format(nrounds=nrounds))

    save = pd.DataFrame({"model":[model], "setting":[setting], "fold": [fold], "nrounds": [nrounds], "tree_depth":[tree_depth], "l2":[l2], "gamma":[gamma], "subsample":[subsample], "rmse":[rmse]})
    save.to_csv(fn, header=False, index=False, mode='a')

