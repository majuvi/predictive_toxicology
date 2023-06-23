# Author: Markus Viljanen
import sys
print(sys.argv)
model = sys.argv[1]
setting = sys.argv[2]
fold = int(sys.argv[3])

# Imports
import xlearn as xl
import numpy as np
import pandas as pd

from scipy.stats import zscore
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

import random 
import csv
import os


hyperparameters = {(row['setting'],row['model'],row['fold']): {'lambda': row['alpha'], 'k': row['k'], 'epoch': row['epoch']} 
                   for i,row in pd.read_csv('XL_optimal_epoch.csv').iterrows()} 

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


fn = "/mnt/scratch_dir/viljanem/predictions/XL_predictions.csv"

y_pred = None

params_opt = hyperparameters[(setting, model, fold)]
test_fold = df["{setting}_test{fold}".format(setting=setting, fold=fold)]
train = df.index.values[~test_fold]
test = df.index.values[test_fold]
print(params_opt)

if setting == "setting1":
    if model == "XL3a_final":
        fn_train = "/mnt/scratch_dir/viljanem/xlearn_data/train_nofeat_setting1_{fold}.svm".format(fold=fold)
        fn_test = "/mnt/scratch_dir/viljanem/xlearn_data/test_nofeat_setting1_{fold}.svm".format(fold=fold)

        fm_model = xl.create_fm()
        #fm_model.setNoBin()
        fm_model.setTrain(fn_train)
        params = {'task':'reg', 'lr':0.03, 'metric': 'rmse'} # 'lambda':0.001, 'epoch':250, 'k':128, 
        params.update(params_opt)
        hash = random.getrandbits(128)
        fm_model.fit(params, "/mnt/scratch_dir/viljanem/model%d.out"%hash)
        fm_model.setTest(fn_test)  # Test data
        fm_model.predict("/mnt/scratch_dir/viljanem/model%d.out"%hash, "/mnt/scratch_dir/viljanem/output%d.csv"%hash)
        y_pred = np.loadtxt("/mnt/scratch_dir/viljanem/output%d.csv" % hash, delimiter=" ", usecols=0)
        y_test = np.loadtxt(fn_test, delimiter=" ", usecols=0)
        os.remove("/mnt/scratch_dir/viljanem/model%d.out"%hash)
        os.remove("/mnt/scratch_dir/viljanem/output%d.csv"%hash)
    if model == "XL3b_final":
        fn_train = "/mnt/scratch_dir/viljanem/xlearn_data/train_nofeat_setting1_{fold}.ffm".format(fold=fold)
        fn_test = "/mnt/scratch_dir/viljanem/xlearn_data/test_nofeat_setting1_{fold}.ffm".format(fold=fold)

        fm_model = xl.create_ffm()
        #fm_model.setNoBin()
        fm_model.setTrain(fn_train)
        params = {'task':'reg', 'lr':0.03, 'metric': 'rmse'} # 'lambda':0.001, 'epoch':30, 'k':128, 
        params.update(params_opt)
        hash = random.getrandbits(128)
        fm_model.fit(params, "/mnt/scratch_dir/viljanem/model%d.out"%hash)
        fm_model.setTest(fn_test)  # Test data
        fm_model.predict("/mnt/scratch_dir/viljanem/model%d.out"%hash, "/mnt/scratch_dir/viljanem/output%d.csv"%hash)
        y_pred = np.loadtxt("/mnt/scratch_dir/viljanem/output%d.csv" % hash, delimiter=" ", usecols=0)
        y_test = np.loadtxt(fn_test, delimiter=" ", usecols=0)
        os.remove("/mnt/scratch_dir/viljanem/model%d.out"%hash)
        os.remove("/mnt/scratch_dir/viljanem/output%d.csv"%hash)
    if model == "XL5a_final":
        fn_train = "/mnt/scratch_dir/viljanem/xlearn_data/train_setting1_{fold}.svm".format(fold=fold)
        fn_test = "/mnt/scratch_dir/viljanem/xlearn_data/test_setting1_{fold}.svm".format(fold=fold)

        fm_model = xl.create_fm()
        #fm_model.setNoBin()
        fm_model.setTrain(fn_train)
        params = {'task':'reg', 'lr':0.03, 'metric': 'rmse'} #' lambda':0.001, 'epoch':1000, 'k':128, 
        params.update(params_opt)
        hash = random.getrandbits(128)
        fm_model.fit(params, "/mnt/scratch_dir/viljanem/model%d.out"%hash)
        fm_model.setTest(fn_test)  # Test data
        fm_model.predict("/mnt/scratch_dir/viljanem/model%d.out"%hash, "/mnt/scratch_dir/viljanem/output%d.csv"%hash)
        y_pred = np.loadtxt("/mnt/scratch_dir/viljanem/output%d.csv" % hash, delimiter=" ", usecols=0)
        y_test = np.loadtxt(fn_test, delimiter=" ", usecols=0)
        os.remove("/mnt/scratch_dir/viljanem/model%d.out"%hash)
        os.remove("/mnt/scratch_dir/viljanem/output%d.csv"%hash)
    if model == "XL5b_final":
        fn_train = "/mnt/scratch_dir/viljanem/xlearn_data/train_setting1_{fold}.ffm".format(fold=fold)
        fn_test = "/mnt/scratch_dir/viljanem/xlearn_data/test_setting1_{fold}.ffm".format(fold=fold)

        fm_model = xl.create_ffm()
        #fm_model.setNoBin()
        fm_model.setTrain(fn_train)
        params = {'task':'reg', 'lr':0.03, 'metric': 'rmse'} # 'lambda':0.001, 'epoch':70, 'k':128, 
        params.update(params_opt)
        hash = random.getrandbits(128)
        fm_model.fit(params, "/mnt/scratch_dir/viljanem/model%d.out"%hash)
        fm_model.setTest(fn_test)  # Test data
        fm_model.predict("/mnt/scratch_dir/viljanem/model%d.out"%hash, "/mnt/scratch_dir/viljanem/output%d.csv"%hash)
        y_pred = np.loadtxt("/mnt/scratch_dir/viljanem/output%d.csv" % hash, delimiter=" ", usecols=0)
        y_test = np.loadtxt(fn_test, delimiter=" ", usecols=0)
        os.remove("/mnt/scratch_dir/viljanem/model%d.out"%hash)
        os.remove("/mnt/scratch_dir/viljanem/output%d.csv"%hash)
if setting == "setting2":
    if model == "XL5a_final":
        fn_train = "/mnt/scratch_dir/viljanem/xlearn_data/train_setting2_{fold}.svm".format(fold=fold)
        fn_test = "/mnt/scratch_dir/viljanem/xlearn_data/test_setting2_{fold}.svm".format(fold=fold)

        fm_model = xl.create_fm()
        #fm_model.setNoBin()
        fm_model.setTrain(fn_train)
        params = {'task':'reg', 'lr':0.03, 'metric': 'rmse'} # 'lambda':0.001, 'epoch':10, 'k':128, 
        params.update(params_opt)
        hash = random.getrandbits(128)
        fm_model.fit(params, "/mnt/scratch_dir/viljanem/model%d.out"%hash)
        fm_model.setTest(fn_test)  # Test data
        fm_model.predict("/mnt/scratch_dir/viljanem/model%d.out"%hash, "/mnt/scratch_dir/viljanem/output%d.csv"%hash)
        y_pred = np.loadtxt("/mnt/scratch_dir/viljanem/output%d.csv" % hash, delimiter=" ", usecols=0)
        y_test = np.loadtxt(fn_test, delimiter=" ", usecols=0)
        os.remove("/mnt/scratch_dir/viljanem/model%d.out"%hash)
        os.remove("/mnt/scratch_dir/viljanem/output%d.csv"%hash)
    if model == "XL5b_final":
        fn_train = "/mnt/scratch_dir/viljanem/xlearn_data/train_setting2_{fold}.ffm".format(fold=fold)
        fn_test = "/mnt/scratch_dir/viljanem/xlearn_data/test_setting2_{fold}.ffm".format(fold=fold)

        fm_model = xl.create_ffm()
        #fm_model.setNoBin()
        fm_model.setTrain(fn_train)
        params = {'task':'reg', 'lr':0.03, 'metric': 'rmse'} #'lambda':0.001, 'epoch':10, 'k':128, 
        params.update(params_opt)
        hash = random.getrandbits(128)
        fm_model.fit(params, "/mnt/scratch_dir/viljanem/model%d.out"%hash)
        fm_model.setTest(fn_test)  # Test data
        fm_model.predict("/mnt/scratch_dir/viljanem/model%d.out"%hash, "/mnt/scratch_dir/viljanem/output%d.csv"%hash)
        y_pred = np.loadtxt("/mnt/scratch_dir/viljanem/output%d.csv" % hash, delimiter=" ", usecols=0)
        y_test = np.loadtxt(fn_test, delimiter=" ", usecols=0)
        os.remove("/mnt/scratch_dir/viljanem/model%d.out"%hash)
        os.remove("/mnt/scratch_dir/viljanem/output%d.csv"%hash)


if not y_pred is None:
    save = pd.DataFrame({"model":model, "setting":setting, "fold": fold, "index":test, "y_pred":y_pred, "y": y_test})
    save.to_csv(fn, header=False, index=False, mode='a')

