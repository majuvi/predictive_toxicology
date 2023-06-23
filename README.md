# Predictive Toxicology

Machine Learning algorithms to predict LC50 values based on ECOTOX data.

## Model naming 

| Code | Paper |
| ----------- | ----------- |
| LM0 | Null model | 
| LM1 | Linear model LM (mean species) |
| LM2 | Linear model LM (mean chemical) |
| LM3 | Linear model LM (mean) |
| LM4 | Linear model LM (basic) |
| LM5 | Linear model LM |
| RF | Random Forest RF |
| SVR | Support Vector Regression SVR |
| TF3 | Neural Network NN (no features) |
| TF5 | Neural Network NN |
| XGB | XGBoost XGB |
| XL3a | (Not field-aware) FM (no features) with early stopping  |
| XL3b | FM (no features) with early stopping |
| XL5a | (Not field-aware) FM with early stopping |
| XL5b | FM with early stopping |
| XL3a\_final | (Not field-aware) FM (no features) |
| XL3b\_final | FM (no features) |
| XL5a\_final | (Not field-aware) FM |
| XL5b\_final | FM |

## Running the models

First unzip the file **MIGRATION\_data\_curation.zip**. Data curation steps based on the public ECOTOX database and the resulting data set are then described in the files:
* MIGRATION\_data\_curation/READ-ME - MIGRATION\_data\_curation.txt
* MIGRATION\_data\_curation/ECOTOX\_MIGRATION\_DATASET\_2022-11-15.csv

One then needs to run the files below in the subsequent order to obtain the results. Temporary files used by the high performance cluster are not included in the repository, but all data and results are included as .csv files.

## File description

### data\_processing.ipynb

Add categorical duration, calculate fingerprints, add cross validation fold identifiers

INPUT: 
* MIGRATION\_data\_curation/ECOTOX\_MIGRATION\_DATASET\_2022-11-15.csv

OUTPUT: 
* df\_tox.csv
* fingerprints.csv
* fingerprints2.csv

### data\_out\_of\_core.ipynb
Produce special format files for out-of-core learning in LibSVM, Xlearn, XGBoost. 

For both setting1 and setting2, produce both train and test files for each 5 cross validation folds. Then for hyperparameter tuning, for both setting1 and setting2, produce both train and test files for each 5 inner cross validation folds in each of the 5 cross validation train folds. These files are used to find optimal hyperparameters in each 5 cross validation train folds.


INPUT: 
* df\_tox.csv
* fingerprints.csv

OUTPUT:
Xlearn
* /mnt/scratch\_dir/viljanem/xlearn\_data/train[\_nofeat]\_setting1\_{fold}.ffm
* /mnt/scratch\_dir/viljanem/xlearn\_data/test[\_nofeat]\_setting1\_{fold}.ffm
* /mnt/scratch\_dir/viljanem/xlearn\_data/train\_setting2\_{fold}.ffm
* /mnt/scratch\_dir/viljanem/xlearn\_data/test\_setting2\_{fold}.ffm
* /mnt/scratch\_dir/viljanem/xlearn\_fold1/train[\_nofeat]\_setting1\_{fold}\_{inner}.ffm
 * /mnt/scratch\_dir/viljanem/xlearn\_fold1/test[\_nofeat]\_setting1\_{fold}\_{inner}.ffm
* /mnt/scratch\_dir/viljanem/xlearn\_fold1/train\_setting2\_{fold}\_{inner}.ffm
* /mnt/scratch\_dir/viljanem/xlearn\_fold1/test\_setting2\_{fold}\_{inner}.ffm

SVM
* /mnt/scratch\_dir/viljanem/xlearn\_data/train[\_nofeat]\_setting1\_{fold}.svm
* /mnt/scratch\_dir/viljanem/xlearn\_data/test[\_nofeat]\_setting1\_{fold}.svm
* /mnt/scratch\_dir/viljanem/xlearn\_data/train\_setting2\_{fold}.svm
* /mnt/scratch\_dir/viljanem/xlearn\_data/test\_setting2\_{fold}.svm
* /mnt/scratch\_dir/viljanem/xlearn\_fold1/train[\_nofeat]\_setting1\_{fold}\_{inner}.svm
 * /mnt/scratch\_dir/viljanem/xlearn\_fold1/test[\_nofeat]\_setting1\_{fold}\_{inner}.svm
* /mnt/scratch\_dir/viljanem/xlearn\_fold1/train\_setting2\_{fold}\_{inner}.svm
* /mnt/scratch\_dir/viljanem/xlearn\_fold1/test\_setting2\_{fold}\_{inner}.svm

XGBoost
* /mnt/scratch\_dir/viljanem/xlearn\_data/train\_setting1\_{fold}.xgb
* /mnt/scratch\_dir/viljanem/xlearn\_data/test\_setting1\_{fold}.xgb
* /mnt/scratch\_dir/viljanem/xlearn\_data/train\_setting2\_{fold}.xgb
* /mnt/scratch\_dir/viljanem/xlearn\_data/test\_setting2\_{fold}.xgb
* /mnt/scratch\_dir/viljanem/xlearn\_fold1/train\_setting1\_{fold}\_{inner}.xgb
* /mnt/scratch\_dir/viljanem/xlearn\_fold1/test\_setting1\_{fold}\_{inner}.xgb
* /mnt/scratch\_dir/viljanem/xlearn\_fold1/train\_setting2\_{fold}\_{inner}.xgb
* /mnt/scratch\_dir/viljanem/xlearn\_fold1/test\_setting2\_{fold}\_{inner}.xgb

### run\_hyperparameters.sh
Perform hyperparameter grid search on a high performance LSF cluster. Will submit jobs that run the files below using different command line arguments corresponding to the setting, fold, and hyperparameter values.

#### run\_LM\_hyperparameters.py
Find RMSE of all hyperparameters in each inner cross-validation fold for the linear model.

INPUT
* df\_tox.csv
* fingerprints.csv

OUTPUT:
* /mnt/scratch\_dir/viljanem/hyperparameters/LM\_hyperparameters.csv

#### run\_XL\_hyperparameters.py
Find RMSE of all hyperparameters in each inner cross-validation fold for the factorization machine.

INPUT: 
* /mnt/scratch\_dir/viljanem/xlearn\_fold1/*.ffm
* /mnt/scratch\_dir/viljanem/xlearn\_fold1/*.svm

OUTPUT:
* /mnt/scratch\_dir/viljanem/hyperparameters/XL\_hyperparameters.csv

#### run\_RF\_hyperparameters.py
Find RMSE of all hyperparameters in each inner cross-validation fold for the random forest.

INPUT
* df\_tox.csv
* fingerprints.csv

OUTPUT:
* /mnt/scratch\_dir/viljanem/hyperparameters/RF\_hyperparameters.csv

#### run\_XGB\_hyperparameters.py
Find RMSE of all hyperparameters in each inner cross-validation fold for the XGBoost model.

INPUT:
* /mnt/scratch\_dir/viljanem/xlearn\_fold1/*.xgb
* /mnt/scratch\_dir/viljanem/xlearn\_fold1/*. xgb

OUTPUT: 
* /mnt/scratch\_dir/viljanem/hyperparameters/XGB\_hyperparameters.csv
* 
#### run\_SVR\_hyperparameters.py
Find RMSE of all hyperparameters in each inner cross-validation fold for the support vector regression.

INPUT
* df\_tox.csv
* fingerprints.csv

OUTPUT:
* /mnt/scratch\_dir/viljanem/hyperparameters/SVR\_hyperparameters.csv

#### run\_TF\_hyperparameters.py
Find RMSE of all hyperparameters in each inner cross-validation fold for the neural network.

INPUT
* df\_tox.csv
* fingerprints.csv

OUTPUT:
* /mnt/scratch\_dir/viljanem/hyperparameters/TF\_hyperparameters.csv

### hyperparameter\_tuning.ipynb
Find optimal hyperparameters in each inner cross-validation fold for the models. Also visualize model performance in the first cross-validation fold given different hyperparameter values.

INPUT:
* /mnt/scratch\_dir/viljanem/hyperparameters/LM\_hyperparameters.csv
* /mnt/scratch\_dir/viljanem/hyperparameters/XL\_hyperparameters.csv
* /mnt/scratch\_dir/viljanem/predictions/RF\_hyperparameters.csv
* /mnt/scratch\_dir/viljanem/hyperparameters/XGB\_hyperparameters.csv
* /mnt/scratch\_dir/viljanem/predictions/SVR\_hyperparameters.csv
* /mnt/scratch\_dir/viljanem/hyperparameters/TF\_hyperparameters.csv

OUTPUT:
* LM\_optimal.csv
* XL\_optimal\_epoch.csv
* XGB\_optimal.csv
* TF\_optimal.csv
* SVR\_optimal.csv
* RF\_optimal.csv

### run\_models.sh
Perform model predictions on a high performance LSF cluster. Will submit jobs that run the files below using different command line arguments corresponding to the setting and fold, given optimal hyperparameter values.

#### run\_LM.py
Run the linear model with optimal hyperparameters and save predictions in each cross-validation test fold.

INPUT: 
* df\_tox.csv
* fingerprints.csv
* LM\_optimal.csv

OUTPUT: 
* /mnt/scratch\_dir/viljanem/predictions/LM\_predictions.csv

#### run\_XL.py
Run the factorization machine with optimal hyperparameters and save predictions in each cross-validation test fold.

INPUT: 
* /mnt/scratch\_dir/viljanem/xlearn\_data/*.ffm
* /mnt/scratch\_dir/viljanem/xlearn\_data/*.svm
* XL\_optimal\_epoch.csv

OUTPUT: 
* /mnt/scratch\_dir/viljanem/predictions/XL\_predictions.csv

#### run\_RF.py
Run the random forest with optimal hyperparameters and save predictions in each cross-validation test fold.

INPUT: 
* df\_tox.csv
* fingerprints.csv
* RF\_optimal.csv

OUTPUT: 
* /mnt/scratch\_dir/viljanem/predictions/RF\_predictions.csv

#### run\_XGB.py
Run the XGBoost with optimal hyperparameters and save predictions in each cross-validation test fold.

INPUT: 
* /mnt/scratch\_dir/viljanem/xlearn\_data/*.xgb
* /mnt/scratch\_dir/viljanem/xlearn\_data/*.xgb
* XGB\_optimal.csv

OUTPUT: 
* /mnt/scratch\_dir/viljanem/predictions/XGB\_predictions.csv

#### run\_SVR.py
Run the support vector regression with optimal hyperparameters and save predictions in each cross-validation test fold.

INPUT: 
* df\_tox.csv
* fingerprints.csv
* SVR\_optimal.csv

OUTPUT: 
* /mnt/scratch\_dir/viljanem/predictions/SVR\_predictions.csv

#### run\_TF.py
Run the neural network with optimal hyperparameters and save predictions in each cross-validation test fold.

INPUT: 
* df\_tox.csv
* fingerprints.csv
* TF\_optimal.csv

OUTPUT: 
* /mnt/scratch\_dir/viljanem/predictions/TF\_predictions.csv

### paper\_results.ipynb
Take the data set and the predictions, then visualize the results and calculate predictive accuracy. The notebook produces the figures and tables used in the paper. These predictions are combined and saved to a .csv.

INPUT: 
* df\_tox.csv
* fingerprints.csv
* /mnt/scratch\_dir/viljanem/predictions/LM\_predictions.csv
* /mnt/scratch\_dir/viljanem/predictions/XL\_predictions.csv
* /mnt/scratch\_dir/viljanem/predictions/RF\_predictions.csv
* /mnt/scratch\_dir/viljanem/predictions/XGB\_predictions.csv
* /mnt/scratch\_dir/viljanem/predictions/SVR\_predictions.csv
* /mnt/scratch\_dir/viljanem/predictions/TF\_predictions.csv

OUTPUT:
* predictions.csv

## Notes
Good luck!