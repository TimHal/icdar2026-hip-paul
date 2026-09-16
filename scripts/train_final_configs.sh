#!/bin/bash

# Script to run all final config files in the indicated directory 5 times each
# The configs are downloaded from mlflow but we override the experiment name
# to place them all in the same experiment for easier comparison and download

# To test the resulting models use the test_final_configs.sh script

# USAGE;
# bash scripts/train_final_configs.sh <config_dir>
# Make sure to adjust N_GPUS 

config_dir=$1
N_GPUS=8

parallel -j "$N_GPUS" \
  'CUDA_VISIBLE_DEVICES=$(({%} - 1)) PYTHONPATH=src python src/cli.py fit \
    --config {1} \
    --model.init_args.prototypes_per_class=5 \
    --data.init_args.batch_size=1024 \
    --trainer.precision="32" \
    --trainer.max_epochs=100 \
    --trainer.logger.init_args.run_name="{1/.}-{2}" \
    --trainer.logger.init_args.experiment_name="SelfSupervised_RERC_FINAL_UNIFIED" \
    --seed_everything={2}' \
  ::: "$config_dir"/*.yaml \
  ::: 1 2 3 4 5