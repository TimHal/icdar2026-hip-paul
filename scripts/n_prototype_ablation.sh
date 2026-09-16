#!/bin/bash

# Very similar to train_final_configs.sh but with the prototype ablation configs instead of the final configs.

config_dir=$1
N_GPUS=8

parallel -j "$N_GPUS" \
  'CUDA_VISIBLE_DEVICES=$(({%} - 1)) PYTHONPATH=src python src/cli.py fit \
    --config {1} \
    --model.init_args.prototypes_per_class={2} \
    --data.init_args.batch_size=1024 \
    --trainer.precision="16-mixed" \
    --trainer.max_epochs=100 \
    --trainer.logger.init_args.run_name="{1/.}-{2}-seed{3}" \
    --trainer.logger.init_args.experiment_name="SelfSupervised_RERC_N_Prototypes_Ablation" \
    --seed_everything={3}' \
  ::: "$config_dir"/*.yaml \
  ::: 1 2 3 4 5 \
  ::: 1 2 3 4 5

  # the lower one is for seeds, the upper one is for number prototypes