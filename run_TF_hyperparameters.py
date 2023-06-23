# Author: Markus Viljanen
import sys
print(sys.argv)
model = sys.argv[1]
setting = sys.argv[2]
M = int(sys.argv[3])
penalty = float(sys.argv[4])
fold = int(sys.argv[5])

# Imports
import pandas as pd
import numpy as np

from scipy.stats import zscore
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


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
y = np.log10(df['Value_Value']).values

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

fn = "/mnt/scratch_dir/viljanem/hyperparameters/TF_hyperparameters.csv"

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

def train_tf_final(train_st, test_st, y_train, y_test):
    BUFFER_SIZE = 16384
    BATCH_SIZE = 128
    N_TRAIN = train_st.shape[0]
    N_TEST = test_st.shape[0]
    N_FEATURES = train_st.shape[1]
    STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

    train_dataset = tf.data.Dataset.from_tensor_slices((train_st, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_st, y_test))
    test_dataset = test_dataset.batch(BATCH_SIZE)

    inputs = keras.Input(shape=(N_FEATURES,), sparse=True, name="indicators")
    x = layers.Dense(M, activation="relu", kernel_regularizer=regularizers.l2(penalty), name="dense_1")(inputs) 
    #x = layers.Dropout(0.5)(x)
    #x = layers.Dense(10, activation="relu", kernel_regularizer=regularizers.l2(0.001), name="dense_2")(x)
    outputs = layers.Dense(1, name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    #lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    #  0.01,
    #  decay_steps=STEPS_PER_EPOCH*1000,
    #  decay_rate=1,
    #  staircase=False)
    model.compile(
        optimizer=keras.optimizers.Adam(),  # Optimizer, lr_schedule
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    history = model.fit(
        train_dataset,
        epochs=100,
        validation_data=test_dataset,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)]
    )
    rmse = history.history['val_loss']
    return(rmse)


if setting == "setting1":
    cv = LeaveGroupsOut(n_splits=ncv, groups=experiments)
    if model == "TF3":
        rmses = {}
        for i, (train, test) in enumerate(cv.split(Xij)):
            temp = Xij[train].tocoo()
            train_st = tf.sparse.SparseTensor(
                np.concatenate((temp.row.reshape(-1,1), temp.col.reshape(-1,1)), axis=1),
                values = temp.data, dense_shape=temp.shape)
            temp = Xij[test].tocoo()
            test_st = tf.sparse.SparseTensor(
                np.concatenate((temp.row.reshape(-1,1), temp.col.reshape(-1,1)), axis=1),
                values = temp.data, dense_shape=temp.shape)
            rmse = train_tf_final(train_st, test_st, y[train], y[test])
            rmses["fold_{fold}".format(fold=i+1)] = pd.Series(rmse)
        rmses = pd.DataFrame(rmses)
        rmses.index = np.arange(len(rmses)) + 1
        rmses = rmses.mean(axis=1)
        nrounds = rmses.idxmin()
        rmse = rmses.min()
        print("nrounds optimal: {nrounds}".format(nrounds=nrounds))
    if model == "TF5":
        rmses = {}
        for i, (train, test) in enumerate(cv.split(Xij)):
            temp = Xijf[train].tocoo()
            train_st = tf.sparse.SparseTensor(
                np.concatenate((temp.row.reshape(-1,1), temp.col.reshape(-1,1)), axis=1),
                values = temp.data, dense_shape=temp.shape)
            temp = Xijf[test].tocoo()
            test_st = tf.sparse.SparseTensor(
                np.concatenate((temp.row.reshape(-1,1), temp.col.reshape(-1,1)), axis=1),
                values = temp.data, dense_shape=temp.shape)
            rmse = train_tf_final(train_st, test_st, y[train], y[test])
            rmses["fold_{fold}".format(fold=i+1)] = pd.Series(rmse)
        rmses = pd.DataFrame(rmses)
        rmses.index = np.arange(len(rmses)) + 1
        rmses = rmses.mean(axis=1)
        nrounds = rmses.idxmin()
        rmse = rmses.min()
        print("nrounds optimal: {nrounds}".format(nrounds=nrounds))
if setting == "setting2":
    cv = LeaveGroupsOut(n_splits=ncv, groups=drugs)
    if model == "TF5":
        rmses = {}
        for i, (train, test) in enumerate(cv.split(Xij)):
            temp = Xif[train].tocoo()
            train_st = tf.sparse.SparseTensor(
                np.concatenate((temp.row.reshape(-1,1), temp.col.reshape(-1,1)), axis=1),
                values = temp.data, dense_shape=temp.shape)
            temp = Xif[test].tocoo()
            test_st = tf.sparse.SparseTensor(
                np.concatenate((temp.row.reshape(-1,1), temp.col.reshape(-1,1)), axis=1),
                values = temp.data, dense_shape=temp.shape)
            rmse = train_tf_final(train_st, test_st, y[train], y[test])
            rmses["fold_{fold}".format(fold=i+1)] = pd.Series(rmse)
        rmses = pd.DataFrame(rmses)
        rmses.index = np.arange(len(rmses)) + 1
        rmses = rmses.mean(axis=1)
        nrounds = rmses.idxmin()
        rmse = rmses.min()
        print("nrounds optimal: {nrounds}".format(nrounds=nrounds))


if not rmse is None:
    save = pd.DataFrame({"model":[model], "setting":[setting], "fold": [fold], "M":[M], "penalty":[penalty], "nrounds":[nrounds], "rmse":[rmse]})
    save.to_csv(fn, header=False, index=False, mode='a')
    

