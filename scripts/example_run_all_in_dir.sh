#!/bin/bash

# This is how I run all experiments in a dir.

for f in conf/somedir/*yaml; do
    python src/cli.py fit --config $f
done


# Run them in parallel  (P=4 means 4 at a time)
ls conf/somedir/*yaml | xargs -n 1 -P 4 -I {} python src/cli.py fit --config {}

# Run 16 experiments in parallel distributed on 8 GPUs (2 per GPU)
# This also controls the GPU assignment by setting CUDA_VISIBLE_DEVICES based on the index of the experiment.
parallel -j 16 'CUDA_VISIBLE_DEVICES=$(( ({#}-1) % 8 )) python src/cli.py fit --config "{}"' ::: sweeps/ssl_emnist_digits_conv/self*.yaml

# =============================================================================
# Optuna parallel study (one worker per GPU)
# =============================================================================
#
# All workers share the same journal file (configured via storage_dir in the
# study config). Optuna coordinates which trials to run — each worker picks up
# the next available trial automatically. No pre-creation step needed.
#
# Single worker (1 GPU):
#
#   PYTHONPATH=src python src/cli.py study \
#     --config conf/experiment/selfsupervised_emnist_balanced_conv.yaml \
#     --study_config conf/study/ssl_emnist_balanced_conv.yaml
#
# Parallel workers — one per GPU:

N_GPUS=8
TRIALS_PER_WORKER=13  # total trials = N_GPUS * TRIALS_PER_WORKER

seq 0 $(( N_GPUS - 1 )) | parallel -j "$N_GPUS" \
  'CUDA_VISIBLE_DEVICES={} PYTHONPATH=src python src/cli.py study \
    --config conf/experiment/selfsupervised_emnist_balanced_conv.yaml \
    --study_config conf/study/ssl_emnist_balanced_conv.yaml \
    --n_trials '"$TRIALS_PER_WORKER"

# Continue an existing study with more trials:
#   Just re-run the same command — Optuna loads the existing journal and
#   adds new trials on top of previous ones.
