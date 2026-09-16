#!/bin/bash

# This is a very hacked script. Idea: Pass the root directory of the mlflow artifacts generated
# by the train_final_configs.sh script, which is then scanned for folders containing both the
# configs and checkpoints to be tested. 

# Usage
#   CUDA_VISIBLE_DEVICES=x ./test_final_configs.sh <mlartifacts_root_dir>

conf_path=$1
for run_dir in $(find $conf_path -type d -name "*epoch*"); do
  ckpt=$(find $run_dir -type f -name "*ckpt"); 
  cfg=$run_dir/../config.yaml; 
  python src/cli.py test --config=$cfg --ckpt_path=$ckpt --trainer.logger.init_args.experiment_name="SelfSupervised_RERC_FINAL_TEST_UNIFIED"; 
done