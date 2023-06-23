# Author: Markus Viljanen
import sys
print(sys.argv)
model = sys.argv[1]
setting = sys.argv[2]
n_estimators = int(sys.argv[3])
#max_features = float(sys.argv[4])
#min_samples_leaf = int(sys.argv[5])
max_features = float(sys.argv[4])
max_samples = float(sys.argv[5])
fold = int(sys.argv[6])

bootstrap = True
if max_samples > 1.0:
    max_samples = None
    bootstrap = False

#min_samples_split  1 2 4 8 16
#min_samples_leaf 1 2 4 8 16

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


fn = "/mnt/scratch_dir/viljanem/hyperparameters/RF_hyperparameters.csv"

ncv  = 5

drugs = df['SMILES'].astype('category').cat.codes.values
experiments = (df['SMILES'].astype('str') + ' X ' + 
               df['Species'].astype('str')).astype('category').cat.codes.values

from sklearn.model_selection import GroupKFold
class LeaveGroupsOut:
    
    def __init__(self, n_splits, groups):
        self.cv = GroupKFold(n_splits=n_splits)
        self.groups = groups
        
    def split(self, X, y=None, groups=None):
        for train, test in self.cv.split(X,y, groups=self.groups):
            yield train, test

rmse = None

if setting == "setting1":
    cv = LeaveGroupsOut(n_splits=ncv, groups=experiments)
    
    clf = RandomForestRegressor(n_jobs=1, random_state = 50, n_estimators = n_estimators, max_leaf_nodes = None, 
                                criterion = 'squared_error', max_depth = None, 
                                max_features = max_features, max_samples=max_samples, bootstrap = bootstrap, 
                                min_samples_split = 2, min_samples_leaf = 1, 
                                min_impurity_decrease = 0, min_weight_fraction_leaf = 0, )

    #clf = RandomForestRegressor(n_estimators = n_estimators, 
    #                            max_features=max_features, max_samples=max_samples, bootstrap = bootstrap,
    #                            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
    #                            criterion = 'squared_error', max_depth = None, max_leaf_nodes = None)
    rmse = -cross_val_score(clf, Xijf, y, scoring="neg_root_mean_squared_error", cv=cv).mean()
if setting == "setting2":
    cv = LeaveGroupsOut(n_splits=ncv, groups=drugs)
    clf = RandomForestRegressor(n_jobs=1, random_state = 50, n_estimators = n_estimators, max_leaf_nodes = None, 
                                criterion = 'squared_error', max_depth = None, 
                                max_features = max_features, max_samples=max_samples, bootstrap = bootstrap, 
                                min_samples_split = 11, min_samples_leaf = 4, 
                                min_impurity_decrease = 0, min_weight_fraction_leaf = 0, )
    #clf = RandomForestRegressor(n_estimators = n_estimators, 
    #                            max_features=max_features, max_samples=max_samples, bootstrap = bootstrap,
    #                            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
    #                            criterion = 'squared_error', max_depth = None, max_leaf_nodes = None)
    rmse = -cross_val_score(clf, Xif, y, scoring="neg_root_mean_squared_error", cv=cv).mean()

if not rmse is None:
    save = pd.DataFrame({"model":[model], "setting":[setting], "fold": [fold], "n_estimators":[n_estimators], "max_features":[max_features], "max_samples":[max_samples], "rmse":[rmse]})
    save.to_csv(fn, header=False, index=False, mode='a')
