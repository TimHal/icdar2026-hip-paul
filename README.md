# RERC — Reconstruction Error Ratio Clustering

Self-supervised classification of handwritten-character datasets with **class-wise
autoencoders**. Train K autoencoders, one per class, and let each sample be claimed by
the autoencoder that reconstructs it best. Only a handful of *prototypes* per class —
as few as one, and they can be rendered font glyphs rather than real samples — are
needed to bootstrap the process. No labels are used during training; labels are read
only to score validation and test metrics.

The central quantity is the **Reconstruction Error Ratio (RER)** of Marks et al. (2024):
for a sample, the ratio between its best and second-best class reconstruction error. A
low RER means the sample is ambiguous between two classes; the dataset-level mean, **chi**,
is a label-free difficulty estimate.

<!-- TODO(citation): replace with the final ICDAR 2026 HIP reference and BibTeX entry. -->
> **Paper**: ICDAR 2026, HIP workshop. Citation to follow.

---

## Contents

- [Installation](#installation)
- [Adapting paths to your machine](#adapting-paths-to-your-machine) ← **read this first**
- [Quick start](#quick-start)
- [Reproducing the paper](#reproducing-the-paper)
- [Repository layout](#repository-layout)
- [Configuration reference](#configuration-reference)
- [Hyperparameter search](#hyperparameter-search)
- [Notebooks and results](#notebooks-and-results)
- [Tests](#tests)
- [License and references](#license-and-references)

---

## Installation

Python 3.10+ and a CUDA-capable GPU (compute capability 7.0+ for recent PyTorch builds).

```bash
git clone https://github.com/TimHal/icdar2026-hip-paul.git
cd icdar2026-hip-paul
```

Either recreate the exact environment the experiments were run in:

```bash
conda env create -f mlresearch_env.yaml -n rerc
conda activate rerc
```

or install the package into an environment you already have:

```bash
pip install -e .
```

Every command below assumes `PYTHONPATH=src`, because the project is laid out as a
`src/` tree that the CLI imports by module name.

---

## Adapting paths to your machine

**The configs in `conf/` are the ones used for the paper and they hardcode absolute
paths from the machine they were run on.** They are kept verbatim so the published
numbers stay traceable. You will need to override three things:

| What | Config key | Default in the configs |
|---|---|---|
| Pre-extracted feature cache | `data.init_args.cache_dir` | `/data/thallybu/cache` |
| Raw torchvision dataset root | `data.init_args.root_dir` | `/data/thallybu/datasets` |
| MLflow tracking server | `trainer.logger.init_args.tracking_uri` | `http://localhost:8090` |

Which of `cache_dir` / `root_dir` applies depends on the datamodule a config uses:
`CachedFeatureDatamodule` (DINOv2/CLIP/MAE feature configs) takes `cache_dir`,
`TorchvisionDatamodule` (convolutional configs that read raw pixels) takes `root_dir`
and downloads the dataset there on first use.

Everything is overridable on the command line, so no config has to be edited:

```bash
PYTHONPATH=src python src/cli.py fit \
    --config conf/experiment/mnist_dino_simple.yaml \
    --data.init_args.cache_dir=./cache \
    --trainer.logger.init_args.tracking_uri=file://$PWD/mlruns
```

### Running without an MLflow server

The configs log to an MLflow tracking server on `localhost:8090`. If you do not want to
run one, point the logger at a local directory instead — this is verified to work
end to end:

```bash
--trainer.logger.init_args.tracking_uri=file://$PWD/mlruns
```

Inspect the results afterwards with `mlflow ui --backend-store-uri ./mlruns`.

> Note: `src/cli.py` also has a `DEFAULT_EXPERIMENTS_ROOT` pointing at
> `/data/thallybu/experiments`, which it passes to the logger as `save_dir`. It is
> inert when `tracking_uri` is set explicitly. Checkpoint destinations are controlled
> with `--checkpoint_dir`.

---

## Quick start

### 1. Extract features (for the DINOv2 / CLIP configs)

Convolutional configs train on raw pixels and need no feature extraction — skip to
step 2 for those.

```bash
PYTHONPATH=src python scripts/extract_features.py \
    --dataset MNIST \
    --model dinov2_vits14 \
    --output ./cache \
    --data_root ./data/datasets
```

This writes `./cache/mnist_dinov2_vits14_{train,test}.pt`. The `dataset_name` a config
asks for is exactly that prefix, e.g. `mnist_dinov2_vits14`. `--dataset` accepts MNIST,
FashionMNIST, KMNIST, EMNIST, QMNIST, CIFAR10, CIFAR100 (use `--emnist_split` for
EMNIST), or point `--data_dir` at an ImageFolder tree with `train/` and `test/`.

### 2. Train

```bash
PYTHONPATH=src python src/cli.py fit \
    --config conf/experiment/mnist_dino_simple.yaml \
    --data.init_args.cache_dir=./cache \
    --trainer.logger.init_args.tracking_uri=file://$PWD/mlruns
```

Validation logs `val/loss`, `val/chi` (mean RER) and `val/accuracy`, the last computed
by matching each autoencoder to its majority true class.

### 3. Test

```bash
PYTHONPATH=src python src/cli.py test \
    --config conf/experiment/mnist_dino_simple.yaml \
    --ckpt_path path/to/checkpoint.ckpt
```

Add `--trainer.fast_dev_run=true` to any of the above for a one-batch smoke run.

---

## Reproducing the paper

### Main results

`results/final_rerc_configs/` holds the 15 tuned configs behind the main table — the
cross product of {EMNIST digits, EMNIST balanced, KMNIST, Omniglot-20, Omniglot-50} and
{convolutional, DINOv2, CLIP}. Each was run with five seeds:

```bash
bash scripts/train_final_configs.sh results/final_rerc_configs

# Evaluate: point this at the MLflow artifact root the training run produced.
# It scans for run directories holding both a config.yaml and a checkpoint.
CUDA_VISIBLE_DEVICES=0 bash scripts/test_final_configs.sh <mlartifacts_root>
```

### Ablations

| Study | Configs | Runner |
|---|---|---|
| Prototypes per class (1–5) | `conf/n_prototype_ablation/` | `scripts/n_prototype_ablation.sh` |
| Prototype source: centroid / random / **font glyph** | `conf/fonts_vs_samples_prototype_ablation/` | `scripts/font_prototype_ablation.sh` |
| Pruning on/off | `conf/pruning_ablation/` | `scripts/pruning_ablation.sh` |

Each takes the config directory as its only argument, e.g.
`bash scripts/n_prototype_ablation.sh conf/pruning_ablation`.

**The runner scripts require GNU `parallel` and assume 8 GPUs** (`N_GPUS=8` near the top
of each script — lower it to match your machine). They fan out over configs × seeds and
write every run into one MLflow experiment for easy comparison. To run a single cell of
an ablation instead, just call `src/cli.py fit` with the corresponding config and pass
the swept parameter yourself, e.g. `--model.init_args.prototypes_per_class=3`.

The font-glyph prototypes are checked in at `data/prototypes/emnist_{digits,balanced}_prototypes/`
as `{class_index}/{FontName}.png`. They are rendered from system fonts and are **not**
regenerable with `scripts/extract_prototypes.py`, which samples prototypes out of a
dataset instead:

```bash
python scripts/extract_prototypes.py --dataset MNIST \
    --output_dir data/prototypes/mnist_random_3 --n_prototypes 3 --seed 42
```

### Baselines

`conf/experiment/baseline_{centroid,gmm}_*.yaml` run the nearest-centroid and
Gaussian-mixture baselines from the same prototypes, through the same CLI.
Published comparison numbers for SCAN, JULE and the nearest-centroid baseline are in
`results/`.

---

## Repository layout

```
├── src/
│   ├── cli.py                            # Lightning CLI entry point (fit/validate/test/study)
│   ├── study.py                          # Optuna study driver
│   ├── model/
│   │   ├── ShallowAutoencoder.py         # MLP autoencoder (Marks et al. 2024)
│   │   ├── ShallowConvAutoencoder.py     # Convolutional autoencoder for raw pixels
│   │   ├── ClasswiseAutoencoderManager.py# The K autoencoders + RER computation
│   │   └── PrototypeLedger.py            # Which sample is assigned to which class
│   ├── task/
│   │   ├── SelfSupervisedAETask.py       # Prototype-guided self-supervised training
│   │   ├── ClasswiseAETask.py            # Supervised reference training
│   │   ├── JITClasswiseAETask.py         # Just-in-time feature extraction variant
│   │   └── MAETask.py                    # Masked-autoencoder pretraining
│   ├── baseline/BaselineTask.py          # Nearest-centroid and GMM baselines
│   ├── data/                             # Cached-feature, torchvision and multi-view datamodules
│   ├── feature_extractors/               # DINOv2, CLIP, MAE
│   ├── mae/                              # Masked autoencoder implementation
│   └── util/rer_metrics.py               # RER and chi
├── conf/
│   ├── experiment/                       # 82 experiment configs
│   ├── study/                            # Optuna search spaces
│   ├── n_prototype_ablation/             # ┐
│   ├── fonts_vs_samples_prototype_ablation/ # ├ ablation configs
│   └── pruning_ablation/                 # ┘
├── data/prototypes/                      # Checked-in font-glyph prototypes
├── scripts/                              # Feature/prototype extraction, experiment runners
├── notebooks/                            # Analysis notebooks (see below)
├── results/                              # Exported metrics and paper figures
├── tools/sweeper.py                      # Grid-expansion sweep helper
└── tests/
```

`CONCEPT.md` documents the research motivation and the RER framework in more depth.

---

## Configuration reference

The knobs that matter most for `SelfSupervisedAETask`:

```yaml
model:
  class_path: task.SelfSupervisedAETask.SelfSupervisedAETask
  init_args:
    num_classes: 10
    feature_dim: 384                # 384 for DINOv2 ViT-S; ignored by conv autoencoders

    # Prototypes — the only supervision in the pipeline
    prototypes_per_class: 3
    prototype_selection: centroid   # random (default) | first | centroid
    auto_select_prototypes: true    # sample prototypes from the data ...
    prototype_source: null          # ... or load them from a folder of PNGs

    # Assignment and pruning at each epoch end
    rer_threshold: 2.0
    enable_pruning: true
    pruning_mode: fixed_k           # fixed_k | threshold

    # Losses
    use_prototype_anchoring: true
    prototype_anchor_weight: 0.1
    use_umap_loss: true
    umap_loss_weight: 0.05
    learning_rate: 0.001

    autoencoder_class: ShallowConvAutoencoder
    autoencoder_kwargs: {image_shape: [1, 28, 28]}

    # Optional: dump reconstruction grids and per-sample errors to MLflow
    log_samples: false
    log_samples_frequency: 0
```

`prototype_source` points at a folder laid out as `{class_index}/{name}.png` and takes
precedence over `auto_select_prototypes`. When `autoencoder_kwargs.image_shape` is set,
the prototype images are logged to MLflow as run artifacts.

---

## Hyperparameter search

```bash
PYTHONPATH=src python src/cli.py study \
  --config conf/experiment/mnist_dino_simple.yaml \
  --study_config conf/study/example_mnist_dino.yaml
```

A study config declares `study.direction`, `study.metric`, `study.n_trials`, a sampler
(`TPESampler`, `RandomSampler`, `CmaEsSampler`, `NSGAIISampler`, `GridSampler`), an
optional pruner, and a `search_space` of dot-path parameters with `type`
(`float`/`int`/`categorical`) plus `low`/`high`/`log`/`choices`. `groups` expresses
atomic multi-parameter choices. See `conf/study/example_mnist_dino.yaml` for HPO and
`conf/study/example_grid_search.yaml` for an ablation-style grid.

Each trial logs to MLflow as `<study_name>_trial_<N>`, tags Optuna parameters with an
`hpo/` prefix and attaches the fully resolved config as an artifact; failed trials get a
`trial_error.txt`. Studies use Optuna's `JournalStorage` (`.journal` files under
`storage_dir`), so several processes can extend the same study concurrently, and
re-running the same command continues it rather than starting over.

---

## Notebooks and results

| Notebook | What it does |
|---|---|
| `notebooks/01_basic_training.ipynb` | Minimal training walkthrough |
| `notebooks/02_rer_analysis.ipynb` | RER and chi statistics on a trained model |
| `notebooks/03_mae_feature_extraction.ipynb` | Using MAE-pretrained features |
| `notebooks/04_misclassification_analysis.ipynb` | Per-class chi, borderline samples, confusion pairs |
| `notebooks/05_prototype_vs_font_ablation.ipynb` | Prototype-strategy accuracy curves |
| `results/confusion_matrix/viz_confusion.ipynb` | Combined confusion-matrix figure |

Notebook 04 reloads trained models through `notebooks/error_analysis_*.yaml`; the
`ckpt_path` in those two files points at the original machine's MLflow artifact store
and has to be repointed at your own checkpoint to re-run it.

`results/` contains the exported metrics and the figures used in the paper —
per-dataset sweep CSVs under `ssl_sweeps/`, the ablation CSVs, confusion matrices, and
the misclassification-analysis SVGs.

---

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

71 tests. The two GPU tests in `tests/test_integration.py` require a GPU of compute
capability 7.0 or newer and fail on older hardware; the other 69 run on CPU.

---

## License and references

MIT — see [LICENSE](LICENSE).

Marks, M., et al. (2024). *Reconstruction Error Ratios for Dataset Analysis.*
