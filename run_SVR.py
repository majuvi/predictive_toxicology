# Author: Markus Viljanen
import sys
print(sys.argv)
model = sys.argv[1]
setting = sys.argv[2]
#M = int(sys.argv[3])
#penalty = float(sys.argv[4])
#gamma = float(sys.argv[5])
#print(M, penalty, gamma)
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

from sklearn.svm import LinearSVR, SVR
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline


hyperparameters = {(row['setting'],row['model'],row['fold']): {'M': row['M'], 'penalty': row['penalty'], 'gamma': row['gamma']} 
                   for i,row in pd.read_csv('SVR_optimal.csv').iterrows()} 
params = hyperparameters[(setting, model, fold)]
M = params['M']
penalty = params['penalty']
gamma = params['gamma']

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



fn = "/mnt/scratch_dir/viljanem/predictions/SVR_predictions.csv"

y_pred = None

test_fold = df["{setting}_test{fold}".format(setting=setting, fold=fold)]
train = df.index.values[~test_fold]
test = df.index.values[test_fold]

if setting == "setting1":
    if M > 0:
        clf = make_pipeline(Nystroem(kernel="rbf", gamma=gamma, random_state=42, n_components=M), 
                        #Nystroem(gamma=0.1, random_state=42, n_components=5000),
                        LinearSVR(loss="squared_epsilon_insensitive", max_iter=10000, C=penalty) #, Ridge()
                       ) 
    else:
        clf = SVR(gamma=gamma, C=penalty)
    #y_pred = cross_val_predict(clf, Xijf, y, cv=cv)
    clf.fit(Xijf[train], y[train])
    y_pred = clf.predict(Xijf[test])
if setting == "setting2":
    if M > 0:
        clf = make_pipeline(Nystroem(kernel="rbf", gamma=gamma, random_state=42, n_components=M), 
                        #Nystroem(gamma=0.1, random_state=42, n_components=5000),
                        LinearSVR(loss="squared_epsilon_insensitive", max_iter=10000, C=penalty) #, Ridge()
                       ) 
    else:
        clf = SVR(gamma=gamma, C=penalty)
    #y_pred = cross_val_predict(clf, Xif, y, cv=cv)
    clf.fit(Xif[train], y[train])
    y_pred = clf.predict(Xif[test])

if not y_pred is None:
    save = pd.DataFrame({"model":model, "setting":setting, "fold": fold, "index":test, "y_pred":y_pred, "y": y[test]})
    save.to_csv(fn, header=False, index=False, mode='a')
