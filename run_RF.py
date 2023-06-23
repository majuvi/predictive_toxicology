# Author: Markus Viljanen
import sys
print(sys.argv)
model = sys.argv[1]
setting = sys.argv[2]
#n_estimators = int(sys.argv[3])
#max_features = float(sys.argv[4])
#max_samples = float(sys.argv[5])
fold = int(sys.argv[3])

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

from scipy.stats import zscore
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, MinMaxScaler
from sklearn.model_selection import GroupKFold

from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor



hyperparameters = {(row['setting'],row['model'],row['fold']): {'n_estimators': row['n_estimators'], 'max_features': row['max_features'], 'max_samples': row['max_samples']} 
                   for i,row in pd.read_csv('RF_optimal.csv').iterrows()} 
params = hyperparameters[(setting, model, fold)]
n_estimators = int(params['n_estimators'])
max_features = params['max_features']
max_samples = params['max_samples']


bootstrap = True
if max_samples > 1.0:
    max_samples = None
    bootstrap = False
    

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


# Label and feature matrices
y = df['Value_Log10'].values

enc = OneHotEncoder()
Xi = csr_matrix(enc.fit_transform(df[['Species']])) # Species dummy
Xj = csr_matrix(enc.fit_transform(df[['SMILES']])) # Drug dummy
Xd = csr_matrix(enc.fit_transform(df[['Duration.Cat']])) # Duration dummy
Xf = csr_matrix(df[feature_columns]) # Drug fingerprint (numeric)

# No features: dummy indicators for species, drugs, duration
#Xij  = hstack([Xi, Xj, Xd], format='csr')

# Features: + class, phylum, drug fingerprint
Xif = hstack([Xi, Xd, Xf], format='csr')
Xijf = hstack([Xi, Xj, Xd, Xf], format='csr')



fn = "/mnt/scratch_dir/viljanem/predictions/RF_predictions.csv"

y_pred = None

test_fold = df["{setting}_test{fold}".format(setting=setting, fold=fold)]
train = df.index.values[~test_fold]
test = df.index.values[test_fold]

if setting == "setting1":

    clf = RandomForestRegressor(n_jobs=1, random_state = 50, n_estimators = n_estimators, max_leaf_nodes = None, 
                                criterion = 'squared_error', max_depth = None, 
                                max_features = max_features, max_samples=max_samples, bootstrap = bootstrap, 
                                min_samples_split = 2, min_samples_leaf = 1, 
                                min_impurity_decrease = 0, min_weight_fraction_leaf = 0, )
    clf.fit(Xijf[train], y[train])
    y_pred = clf.predict(Xijf[test]) #cross_val_predict(clf, Xijf, y, cv=cv)
if setting == "setting2":
    clf = RandomForestRegressor(n_jobs=1, random_state = 50, n_estimators = n_estimators, max_leaf_nodes = None, 
                                criterion = 'squared_error', max_depth = None, 
                                max_features = max_features, max_samples=max_samples, bootstrap = bootstrap, 
                                min_samples_split = 11, min_samples_leaf = 4, 
                                min_impurity_decrease = 0, min_weight_fraction_leaf = 0, )
    clf.fit(Xif[train], y[train])
    y_pred = clf.predict(Xif[test]) #cross_val_predict(clf, Xif, y, cv=cv)

if not y_pred is None:
    save = pd.DataFrame({"model":model, "setting":setting, "fold": fold, "index":test, "y_pred":y_pred, "y": y[test]})
    save.to_csv(fn, header=False, index=False, mode='a')
