#!/bin/bash
# Author: Markus Viljanen

rm /mnt/scratch_dir/viljanem/predictions/LM_predictions.csv
for model in "LM0" "LM1" "LM2" "LM3" "LM4" "LM5"
do
 for setting in "setting1" "setting2" 
 do
  for fold in "1" "2" "3" "4" "5"
  do
   bsub python run_LM.py $model $setting $fold
   sleep 0.1
  done
 done
done

rm /mnt/scratch_dir/viljanem/predictions/XL_predictions.csv
for model in "XL3a_final" "XL3b_final" "XL5a_final" "XL5b_final"
do
 for setting in "setting1" "setting2" 
 do
  for fold in "1" "2" "3" "4" "5"
  do
   bsub python run_XL.py $model $setting $fold
   sleep 0.1
  done
 done
done

rm /mnt/scratch_dir/viljanem/predictions/XGB_predictions.csv
for setting in "setting2" "setting1" 
do
 for fold in "1" "2" "3" "4" "5"
 do
  bsub python run_XGB.py XGB $setting $fold
  sleep 0.1
 done
done

rm /mnt/scratch_dir/viljanem/hyperparameters/TF_predictions.csv
for model in "TF3" "TF5"
do
 for setting in "setting1" "setting2" 
 do
  for fold in "1" "2" "3" "4" "5"
  do
   bsub python run_TF.py $model $setting $fold
   sleep 0.1
  done
 done
done

rm /mnt/scratch_dir/viljanem/predictions/RF_predictions.csv
for setting in "setting1" "setting2" 
do
 for fold in "1" "2" "3" "4" "5" 
 do
  bsub python run_RF.py RF $setting $fold
  sleep 0.1
 done
done

rm /mnt/scratch_dir/viljanem/predictions/SVR_predictions.csv
for setting in "setting2" "setting1" 
do
 for fold in "1" "2" "3" "4" "5"
 do
  bsub python run_SVR.py SVR $setting $fold
  sleep 0.1
 done
done








