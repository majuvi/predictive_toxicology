# Author: Markus Viljanen
import sys
print(sys.argv)
model = sys.argv[1]
setting = sys.argv[2]
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

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error

hyperparameters = {(row['setting'],row['model'],row['fold']): {'alpha': row['alpha']} 
                   for i,row in pd.read_csv('LM_optimal.csv').iterrows()} 

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
Xij  = hstack([Xi, Xj, Xd], format='csr')

# Features: + class, phylum, drug fingerprint
Xif = hstack([Xi, Xd, Xf], format='csr')
Xijf = hstack([Xi, Xj, Xd, Xf], format='csr')

# Given sparse matrices of dummy indicators Xi and features Xf, calculate all their interaction terms
def get_interactions(Xi, Xf):
    n  = Xi.shape[0]
    ns = Xi.shape[1]
    nf = Xf.shape[1]
    nd = 0
    indptr = [nd]
    indices = []
    data = []
    for i in range(n):
        j = Xi.getrow(i).indices[0]
        features = Xf.getrow(i)
        nd += len(features.data)
        indptr.append(nd)
        indices.extend(features.indices + j * nf)
        data.extend(features.data)
    Xif_ = csr_matrix((data, indices, indptr), shape=(n, ns*nf))
    return(Xif_)



fn = "/mnt/scratch_dir/viljanem/predictions/LM_predictions.csv"

y_pred = None

params = hyperparameters[(setting, model, fold)]
test_fold = df["{setting}_test{fold}".format(setting=setting, fold=fold)]
train = df.index.values[~test_fold]
test = df.index.values[test_fold]

if setting == "setting1":
    if model == "LM0":
        #### Null model
        clf = DummyRegressor()
        clf.fit(Xi[train], y[train])
        y_pred = clf.predict(Xi[test])
    if model == "LM1":
        #### Baseline: mean of species
        clf = Ridge(**params)#Ridge(alpha=10)
        clf.fit(Xi[train], y[train])
        y_pred = clf.predict(Xi[test])
    if model == "LM2":
        #### Baseline: mean of compound
        clf = Ridge(**params)#Ridge(alpha=0.1)
        clf.fit(Xj[train], y[train])
        y_pred = clf.predict(Xj[test])
    if model == "LM3":
        #### Baseline: mean of species and compound
        clf = Ridge(**params)#Ridge(alpha=1)
        clf.fit(Xij[train], y[train])
        y_pred = clf.predict(Xij[test])
    if model == "LM4":
        #### Baseline: linear model of dummy species, dummy compound, and compound features
        clf = Ridge(**params)#Ridge(alpha=10)
        clf.fit(Xijf[train], y[train])
        y_pred = clf.predict(Xijf[test])
    if model == "LM5":
        #### Baseline: linear model of dummy species, dummy compound, compound features, species x compound interaction
        Xif_ = get_interactions(Xi, Xf)
        Xijf_if = hstack([Xijf, Xif_], format='csr')
        clf = Ridge(**params)#Ridge(alpha=10)
        clf.fit(Xijf_if[train], y[train])
        y_pred = clf.predict(Xijf_if[test])
if setting == "setting2":
    if model == "LM0":
        #### Null model
        clf = DummyRegressor()
        clf.fit(Xi[train], y[train])
        y_pred = clf.predict(Xi[test])
    if model == "LM1":
        #### Baseline: mean of species
        clf = Ridge(**params)#Ridge(alpha=10)
        clf.fit(Xi[train], y[train])
        y_pred = clf.predict(Xi[test])
    if model == "LM4":
        #### Baseline: linear model of dummy species, dummy compound, and compound features
        clf = Ridge(**params)#Ridge(alpha=100)
        clf.fit(Xif[train], y[train])
        y_pred = clf.predict(Xif[test])
    if model == "LM5":
        #### Baseline: linear model of dummy species, dummy compound, compound features, species x compound interaction
        Xif_ = get_interactions(Xi, Xf)
        Xif_if = hstack([Xif, Xif_], format='csr')
        clf = Ridge(**params)#Ridge(alpha=100)
        clf.fit(Xif_if[train], y[train])
        y_pred = clf.predict(Xif_if[test])

if not y_pred is None:
    save = pd.DataFrame({"model":model, "setting":setting, "fold": fold, "index":test, "y_pred":y_pred, "y": y[test]})
    save.to_csv(fn, header=False, index=False, mode='a')
