# Author: Markus Viljanen
import sys
print(sys.argv)
model = sys.argv[1]
setting = sys.argv[2]
M = int(sys.argv[3])
penalty = float(sys.argv[4])
gamma = float(sys.argv[5])
fold = int(sys.argv[6])

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

from sklearn.svm import LinearSVR, SVR
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline



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


fn = "/mnt/scratch_dir/viljanem/hyperparameters/SVR_hyperparameters.csv"

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
    if M > 0:
        clf = make_pipeline(Nystroem(kernel="rbf", gamma=gamma, random_state=42, n_components=M), 
                        #Nystroem(gamma=0.1, random_state=42, n_components=5000),
                        LinearSVR(loss="squared_epsilon_insensitive", max_iter=10000, C=penalty) #, Ridge()
                       ) 
    else:
        clf = SVR(gamma=gamma, C=penalty)
    rmse = -cross_val_score(clf, Xijf, y, scoring="neg_root_mean_squared_error", cv=cv).mean()
if setting == "setting2":
    cv = LeaveGroupsOut(n_splits=ncv, groups=drugs)
    if M > 0:
        clf = make_pipeline(Nystroem(kernel="rbf", gamma=gamma, random_state=42, n_components=M), 
                        #Nystroem(gamma=0.1, random_state=42, n_components=5000),
                        LinearSVR(loss="squared_epsilon_insensitive", max_iter=10000, C=penalty) #, Ridge()
                       ) 
    else:
        clf = SVR(gamma=gamma, C=penalty)
    rmse = -cross_val_score(clf, Xif, y, scoring="neg_root_mean_squared_error", cv=cv).mean()

if not rmse is None:
    save = pd.DataFrame({"model":[model], "setting":[setting], "fold": [fold], "M":[M], "penalty":[penalty], "gamma":[gamma], "rmse":[rmse]})
    save.to_csv(fn, header=False, index=False, mode='a')

