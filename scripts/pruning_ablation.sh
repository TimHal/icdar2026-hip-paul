#!/bin/bash

# Pruning ablation: same base configs as n_prototype_ablation but vary enable_pruning
# instead of prototypes_per_class. Prototypes are left at the configs' default value.

config_dir=$1
N_GPUS=8

parallel -j "$N_GPUS" \
  'CUDA_VISIBLE_DEVICES=$(({%} - 1)) PYTHONPATH=src python src/cli.py fit \
    --config {1} \
    --model.init_args.enable_pruning={2} \
    --data.init_args.batch_size=1024 \
    --trainer.precision="16-mixed" \
    --trainer.max_epochs=100 \
    --trainer.logger.init_args.run_name="{1/.}-prune{2}-seed{3}" \
    --trainer.logger.init_args.experiment_name="SelfSupervised_RERC_Pruning_Ablation" \
    --seed_everything={3}' \
  ::: "$config_dir"/*.yaml \
  ::: true false \
  ::: 1 2 3 4 5

  # middle axis: enable_pruning, lower axis: seeds
