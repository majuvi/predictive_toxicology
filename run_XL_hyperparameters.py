# Author: Markus Viljanen
import sys
print(sys.argv)
model = sys.argv[1]
setting = sys.argv[2]
alpha = float(sys.argv[3])
k = int(sys.argv[4])
epoch = int(sys.argv[5])
fold = int(sys.argv[6])
#epoch = 500

# Imports
import xlearn as xl
import numpy as np
import pandas as pd
import random
import os
import os.path

from scipy.stats import zscore
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

if os.path.isfile('XL_optimal.csv'):
    hyperparameters = {(row['setting'],row['model'],row['fold']): {'alpha': row['alpha'], 'k': row['k']} 
                       for i,row in pd.read_csv('XL_optimal.csv').iterrows()} 

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

# Training set
df = df[~df["{setting}_test{fold}".format(setting=setting, fold=fold)]]

#create_hyperparameter_sets(df, "setting1", libsvm=True)
#create_hyperparameter_sets(df, "setting2", libsvm=True)
#create_hyperparameter_sets(df, "setting1", libsvm=False)
#create_hyperparameter_sets(df, "setting2", libsvm=False)
#setting = None

fn = "/mnt/scratch_dir/viljanem/hyperparameters/XL_hyperparameters.csv"

validate = False if model[-5:] == "final" else True
print(model)
print(validate)

rmse = None
ncv = 5

if setting == "setting1":
    if (model == "XL3a") or (model == "XL3a_final"):
        rmses = []
        for i in range(ncv):
            print("="*20 + "Fold {fold}".format(fold=i+1) + "="*20, end="\n\n\n")
            fn_train = "/mnt/scratch_dir/viljanem/xlearn_fold1/train_nofeat_setting1_{fold}_{inner}.svm".format(fold=fold, inner=i+1)
            fn_test = "/mnt/scratch_dir/viljanem/xlearn_fold1/test_nofeat_setting1_{fold}_{inner}.svm".format(fold=fold, inner=i+1)

            fm_model = xl.create_fm()
            #fm_model.setNoBin()
            fm_model.setTrain(fn_train)
            params = {'task':'reg', 'lr':0.03, 'lambda':alpha, 'epoch':epoch, 'k':k, 'metric': 'rmse'}
            if validate:
                fm_model.setValidate(fn_test)
            else:
                params_opt = hyperparameters[(setting, model[:-6], fold)]
                alpha = params_opt['alpha']
                k = params_opt['k']
                params.update({'lambda': alpha, 'k': k})
            hash = random.getrandbits(128)
            fm_model.fit(params, "/mnt/scratch_dir/viljanem/model%d.out"%hash)
            fm_model.setTest(fn_test)  # Test data
            fm_model.predict("/mnt/scratch_dir/viljanem/model%d.out"%hash, "/mnt/scratch_dir/viljanem/output%d.csv"%hash)
            y_test = np.loadtxt(fn_test, delimiter=" ", usecols=0)
            y_pred = np.loadtxt("/mnt/scratch_dir/viljanem/output%d.csv" % hash, delimiter=" ", usecols=0)
            rmse = mean_squared_error(y_test, y_pred, squared=False) 
            rmses.append(rmse)
            os.remove("/mnt/scratch_dir/viljanem/model%d.out"%hash)
            os.remove("/mnt/scratch_dir/viljanem/output%d.csv"%hash)
        rmse = np.mean(rmses)
    if (model == "XL3b") or (model == "XL3b_final"):
        rmses = []
        for i in range(ncv):
            print("="*20 + "Fold {fold}".format(fold=i+1) + "="*20, end="\n\n\n")
            fn_train = "/mnt/scratch_dir/viljanem/xlearn_fold1/train_nofeat_setting1_{fold}_{inner}.ffm".format(fold=fold, inner=i+1)
            fn_test = "/mnt/scratch_dir/viljanem/xlearn_fold1/test_nofeat_setting1_{fold}_{inner}.ffm".format(fold=fold, inner=i+1)

            fm_model = xl.create_ffm()
            #fm_model.setNoBin()
            fm_model.setTrain(fn_train)
            params = {'task':'reg', 'lr':0.03, 'lambda':alpha, 'epoch':epoch, 'k':k, 'metric': 'rmse'}
            if validate:
                fm_model.setValidate(fn_test)
            else:
                params_opt = hyperparameters[(setting, model[:-6], fold)]
                alpha = params_opt['alpha']
                k = params_opt['k']
                params.update({'lambda': alpha, 'k': k})
            hash = random.getrandbits(128)
            fm_model.fit(params, "/mnt/scratch_dir/viljanem/model%d.out"%hash)
            fm_model.setTest(fn_test)  # Test data
            fm_model.predict("/mnt/scratch_dir/viljanem/model%d.out"%hash, "/mnt/scratch_dir/viljanem/output%d.csv"%hash)
            y_test = np.loadtxt(fn_test, delimiter=" ", usecols=0)
            y_pred = np.loadtxt("/mnt/scratch_dir/viljanem/output%d.csv" % hash, delimiter=" ", usecols=0)
            rmse = mean_squared_error(y_test, y_pred, squared=False) 
            rmses.append(rmse)
            os.remove("/mnt/scratch_dir/viljanem/model%d.out"%hash)
            os.remove("/mnt/scratch_dir/viljanem/output%d.csv"%hash)
        rmse = np.mean(rmses)
    if (model == "XL5a") or (model == "XL5a_final"):
        rmses = []
        for i in range(ncv):
            print("="*20 + "Fold {fold}".format(fold=i+1) + "="*20, end="\n\n\n")
            fn_train = "/mnt/scratch_dir/viljanem/xlearn_fold1/train_setting1_{fold}_{inner}.svm".format(fold=fold, inner=i+1)
            fn_test = "/mnt/scratch_dir/viljanem/xlearn_fold1/test_setting1_{fold}_{inner}.svm".format(fold=fold, inner=i+1)

            fm_model = xl.create_fm()
            #fm_model.setNoBin()
            fm_model.setTrain(fn_train)
            params = {'task':'reg', 'lr':0.03, 'lambda':alpha, 'epoch':epoch, 'k':k, 'metric': 'rmse'}
            if validate:
                fm_model.setValidate(fn_test)
            else:
                params_opt = hyperparameters[(setting, model[:-6], fold)]
                alpha = params_opt['alpha']
                k = params_opt['k']
                params.update({'lambda': alpha, 'k': k})
            hash = random.getrandbits(128)
            fm_model.fit(params, "/mnt/scratch_dir/viljanem/model%d.out"%hash)
            fm_model.setTest(fn_test)  # Test data
            fm_model.predict("/mnt/scratch_dir/viljanem/model%d.out"%hash, "/mnt/scratch_dir/viljanem/output%d.csv"%hash)
            y_test = np.loadtxt(fn_test, delimiter=" ", usecols=0)
            y_pred = np.loadtxt("/mnt/scratch_dir/viljanem/output%d.csv" % hash, delimiter=" ", usecols=0)
            rmse = mean_squared_error(y_test, y_pred, squared=False) 
            rmses.append(rmse)
            os.remove("/mnt/scratch_dir/viljanem/model%d.out"%hash)
            os.remove("/mnt/scratch_dir/viljanem/output%d.csv"%hash)
        rmse = np.mean(rmses)
    if (model == "XL5b") or (model == "XL5b_final"):
        rmses = []
        for i in range(ncv):
            print("="*20 + "Fold {fold}".format(fold=i+1) + "="*20, end="\n\n\n")
            fn_train = "/mnt/scratch_dir/viljanem/xlearn_fold1/train_setting1_{fold}_{inner}.ffm".format(fold=fold, inner=i+1)
            fn_test = "/mnt/scratch_dir/viljanem/xlearn_fold1/test_setting1_{fold}_{inner}.ffm".format(fold=fold, inner=i+1)

            fm_model = xl.create_ffm()
            #fm_model.setNoBin()
            fm_model.setTrain(fn_train)
            params = {'task':'reg', 'lr':0.03, 'lambda':alpha, 'epoch':epoch, 'k':k, 'metric': 'rmse'}
            if validate:
                fm_model.setValidate(fn_test)
            else:
                params_opt = hyperparameters[(setting, model[:-6], fold)]
                alpha = params_opt['alpha']
                k = params_opt['k']
                params.update({'lambda': alpha, 'k': k})
            hash = random.getrandbits(128)
            fm_model.fit(params, "/mnt/scratch_dir/viljanem/model%d.out"%hash)
            fm_model.setTest(fn_test)  # Test data
            fm_model.predict("/mnt/scratch_dir/viljanem/model%d.out"%hash, "/mnt/scratch_dir/viljanem/output%d.csv"%hash)
            y_test = np.loadtxt(fn_test, delimiter=" ", usecols=0)
            y_pred = np.loadtxt("/mnt/scratch_dir/viljanem/output%d.csv" % hash, delimiter=" ", usecols=0)
            rmse = mean_squared_error(y_test, y_pred, squared=False) 
            rmses.append(rmse)
            os.remove("/mnt/scratch_dir/viljanem/model%d.out"%hash)
            os.remove("/mnt/scratch_dir/viljanem/output%d.csv"%hash)
        rmse = np.mean(rmses)
if setting == "setting2":
    if (model == "XL5a") or (model == "XL5a_final"):
        rmses = []
        for i in range(ncv):
            print("="*20 + "Fold {fold}".format(fold=i+1) + "="*20, end="\n\n\n")
            fn_train = "/mnt/scratch_dir/viljanem/xlearn_fold1/train_setting2_{fold}_{inner}.svm".format(fold=fold, inner=i+1)
            fn_test = "/mnt/scratch_dir/viljanem/xlearn_fold1/test_setting2_{fold}_{inner}.svm".format(fold=fold, inner=i+1)

            fm_model = xl.create_fm()
            #fm_model.setNoBin()
            fm_model.setTrain(fn_train)
            params = {'task':'reg', 'lr':0.03, 'lambda':alpha, 'epoch':epoch, 'k':k, 'metric': 'rmse'}
            if validate:
                fm_model.setValidate(fn_test)
            else:
                params_opt = hyperparameters[(setting, model[:-6], fold)]
                alpha = params_opt['alpha']
                k = params_opt['k']
                params.update({'lambda': alpha, 'k': k})
            hash = random.getrandbits(128)
            fm_model.fit(params, "/mnt/scratch_dir/viljanem/model%d.out"%hash)
            fm_model.setTest(fn_test)  # Test data
            fm_model.predict("/mnt/scratch_dir/viljanem/model%d.out"%hash, "/mnt/scratch_dir/viljanem/output%d.csv"%hash)
            y_test = np.loadtxt(fn_test, delimiter=" ", usecols=0)
            y_pred = np.loadtxt("/mnt/scratch_dir/viljanem/output%d.csv" % hash, delimiter=" ", usecols=0)
            rmse = mean_squared_error(y_test, y_pred, squared=False) 
            rmses.append(rmse)
            os.remove("/mnt/scratch_dir/viljanem/model%d.out"%hash)
            os.remove("/mnt/scratch_dir/viljanem/output%d.csv"%hash)
        rmse = np.mean(rmses)
    if (model == "XL5b") or (model == "XL5b_final"):
        rmses = []
        for i in range(ncv):
            print("="*20 + "Fold {fold}".format(fold=i+1) + "="*20, end="\n\n\n")
            fn_train = "/mnt/scratch_dir/viljanem/xlearn_fold1/train_setting2_{fold}_{inner}.ffm".format(fold=fold, inner=i+1)
            fn_test = "/mnt/scratch_dir/viljanem/xlearn_fold1/test_setting2_{fold}_{inner}.ffm".format(fold=fold, inner=i+1)

            fm_model = xl.create_ffm()
            #fm_model.setNoBin()
            fm_model.setTrain(fn_train)
            params = {'task':'reg', 'lr':0.03, 'lambda':alpha, 'epoch':epoch, 'k':k, 'metric': 'rmse'}
            if validate:
                fm_model.setValidate(fn_test)
            else:
                params_opt = hyperparameters[(setting, model[:-6], fold)]
                alpha = params_opt['alpha']
                k = params_opt['k']
                params.update({'lambda': alpha, 'k': k})
            hash = random.getrandbits(128)
            fm_model.fit(params, "/mnt/scratch_dir/viljanem/model%d.out"%hash)
            fm_model.setTest(fn_test)  # Test data
            fm_model.predict("/mnt/scratch_dir/viljanem/model%d.out"%hash, "/mnt/scratch_dir/viljanem/output%d.csv"%hash)
            y_test = np.loadtxt(fn_test, delimiter=" ", usecols=0)
            y_pred = np.loadtxt("/mnt/scratch_dir/viljanem/output%d.csv" % hash, delimiter=" ", usecols=0)
            rmse = mean_squared_error(y_test, y_pred, squared=False) 
            rmses.append(rmse)
            os.remove("/mnt/scratch_dir/viljanem/model%d.out"%hash)
            os.remove("/mnt/scratch_dir/viljanem/output%d.csv"%hash)
        rmse = np.mean(rmses)
if not rmse is None:
    save = pd.DataFrame({"model":[model], "setting":[setting], "fold": [fold], "alpha":[alpha], "epoch":[epoch], "k":[k], "rmse":[rmse]})
    save.to_csv(fn, header=False, index=False, mode='a')
