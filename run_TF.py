# Author: Markus Viljanen
import sys
print(sys.argv)
model = sys.argv[1]
setting = sys.argv[2]
#M = int(sys.argv[3])
#penalty = float(sys.argv[4])
#epochs = int(sys.argv[5])
fold = int(sys.argv[3])

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

hyperparameters = {(row['setting'],row['model'],row['fold']): {'M': row['M'], 'penalty': row['penalty'], 'nrounds': row['nrounds']} 
                   for i,row in pd.read_csv('TF_optimal.csv').iterrows()} 
params = hyperparameters[(setting, model, fold)]
M = params['M']
penalty = params['penalty']
epochs = params['nrounds']

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


def train_tf(train_st, test_st, y_train, y_test):
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
        epochs=epochs
    )
    y_pred = model.predict(test_dataset).reshape(-1)
    return(y_pred)


fn = "/mnt/scratch_dir/viljanem/predictions/TF_predictions.csv"

y_pred = None

test_fold = df["{setting}_test{fold}".format(setting=setting, fold=fold)]
train = df.index.values[~test_fold]
test = df.index.values[test_fold]

if setting == "setting1":
    if model == "TF3":
        temp = Xij[train].tocoo()
        train_st = tf.sparse.SparseTensor(
            np.concatenate((temp.row.reshape(-1,1), temp.col.reshape(-1,1)), axis=1),
            values = temp.data, dense_shape=temp.shape)
        temp = Xij[test].tocoo()
        test_st = tf.sparse.SparseTensor(
            np.concatenate((temp.row.reshape(-1,1), temp.col.reshape(-1,1)), axis=1),
            values = temp.data, dense_shape=temp.shape)
        y_pred = train_tf(train_st, test_st, y[train], y[test])
    if model == "TF5":
        temp = Xijf[train].tocoo()
        train_st = tf.sparse.SparseTensor(
            np.concatenate((temp.row.reshape(-1,1), temp.col.reshape(-1,1)), axis=1),
            values = temp.data, dense_shape=temp.shape)
        temp = Xijf[test].tocoo()
        test_st = tf.sparse.SparseTensor(
            np.concatenate((temp.row.reshape(-1,1), temp.col.reshape(-1,1)), axis=1),
            values = temp.data, dense_shape=temp.shape)
        y_pred = train_tf(train_st, test_st, y[train], y[test])
if setting == "setting2":
    if model == "TF5":
        temp = Xif[train].tocoo()
        train_st = tf.sparse.SparseTensor(
            np.concatenate((temp.row.reshape(-1,1), temp.col.reshape(-1,1)), axis=1),
            values = temp.data, dense_shape=temp.shape)
        temp = Xif[test].tocoo()
        test_st = tf.sparse.SparseTensor(
            np.concatenate((temp.row.reshape(-1,1), temp.col.reshape(-1,1)), axis=1),
            values = temp.data, dense_shape=temp.shape)
        y_pred = train_tf(train_st, test_st, y[train], y[test])

if not y_pred is None:
    save = pd.DataFrame({"model":model, "setting":setting, "fold": fold, "index":test, "y_pred":y_pred, "y": y[test]})
    save.to_csv(fn, header=False, index=False, mode='a')
