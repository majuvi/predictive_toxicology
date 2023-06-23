#!/bin/bash
# Author: Markus Viljanen

rm /mnt/scratch_dir/viljanem/hyperparameters/LM_hyperparameters.csv
for model in "LM0" "LM1" "LM2" "LM3" "LM4" "LM5"
do
 for setting in "setting1" "setting2" 
 do
  for fold in "1" "2" "3" "4" "5"
  do
    for alpha in "0.0001" "0.001" "0.01" "0.1" "1" "10" "100" "1000" "10000"
    do
      bsub python run_LM_hyperparameters.py $model $setting $alpha $fold
      sleep 0.1
    done
  done
 done
done

rm /mnt/scratch_dir/viljanem/hyperparameters/XL_hyperparameters.csv
# First run with some fixed alpha and k, xlearn will create cached binary files
for model in "XL3a" "XL3b" "XL5a" "XL5b" 
do
 for setting in "setting1" "setting2"  
 do
  for fold in "1" "2" "3" "4" "5"
  do
   for alpha in "0.000001" "0.00001" "0.0001" "0.001" "0.01" "0.1"
   do
     for k in "32" "64" "128" "256" 
     do
       bsub python run_XL_hyperparameters.py $model $setting $alpha $k "500" $fold
       sleep 0.1
     done
   done
  done
 done
done


# Annoying: We do not know what was the epoch number with early stopping
# run hyperparameter_tuning.ipynb and repeat over epoch with optimal alpha, k
for epoch in "10" "20" "30" "40" "50" "60" "70" "80" "90" "100" "150" "200" "250" "300" 
do
 for model in "XL3a_final" "XL3b_final" "XL5a_final" "XL5b_final" 
 do
  for setting in "setting1" "setting2"  
  do
   for fold in "1" "2" "3" "4" "5"
   do
     bsub python run_XL_hyperparameters.py $model $setting 0 0 $epoch $fold
   done
  done
 done
done

rm /mnt/scratch_dir/viljanem/hyperparameters/XGB_hyperparameters.csv
for setting in "setting2" "setting1" 
do
 for fold in "1" "2" "3" "4" "5"
 do
  for tree_depth in "3" "4" "5" "7" "9"
  do
    for lambda in "0" "1" "10" "100"
    do
      for gamma in "0" "0.1" "1" 
      do
        for subsample in "0.50" "0.75" "1.00" 
        do
          bsub python run_XGB_hyperparameters.py "XGB" $setting $tree_depth $lambda $gamma $subsample $fold
          sleep 0.1
        done
      done
    done
  done
 done
done

rm /mnt/scratch_dir/viljanem/hyperparameters/TF_hyperparameters.csv
for model in "TF3" "TF5"
do
 for setting in "setting1" "setting2" 
 do
 for fold in "1" "2" "3" "4" "5"
 do
  for M in "64" "128" "256" "512"
  do
    for penalty in "1e-6" "1e-5" "1e-4" "1e-3" "1e-2" 
    do
      bsub python run_TF_hyperparameters.py $model $setting $M $penalty $fold
      sleep 1
    done
  done
 done
 done
done

rm /mnt/scratch_dir/viljanem/hyperparameters/RF_hyperparameters.csv
model="RF" 
for n_estimators in "300" #"100" "200" "300" "400" "500" # past 300 no gains
do
 for setting in "setting1" "setting2" 
 do
 for fold in "1" "2" "3" "4" "5"
 do
  for max_features in "0.25" "0.5" "0.75" "1.00"
  do
    for max_samples in "0.25" "0.5" "0.75" "1.00" "2.00"
    do
      bsub python run_RF_hyperparameters.py $model $setting $n_estimators $max_features $max_samples $fold
      sleep 0.1
    done
  done
 done
 done
done

rm /mnt/scratch_dir/viljanem/hyperparameters/SVR_hyperparameters.csv
model="SVR" 
for M in "0" "1000" "2000" "4000" "8000" 
do
 for setting in "setting1" "setting2" 
 do
 for fold in "1" "2" "3" "4" "5"
 do
  for gamma in "0.01" "0.02" "0.04", "0.08" "0.16"
  do
    for penalty in "0.01" "0.1" "1" "10" "100" "1000" "10000" 
    do
      bsub python run_SVR_hyperparameters.py $model $setting $M $penalty $gamma $fold
      sleep 0.1
    done
  done
 done
 done
done





