# tools/sweeper.py

Generates one config YAML per hyperparameter combination from a base config and a sweep definition. The original config is never modified.

## Usage

```bash
conda activate mlresearch
python tools/sweeper.py <base_config.yaml> <sweep_config.yaml>
```

Output: one `{base_stem}_{index:04d}.yaml` per combination + a `manifest.yaml` (index → overrides) in the configured output directory.

## Sweep definition

```yaml
output_dir: sweeps/my_experiment   # where to write generated configs

# params: independent parameters, cartesian-producted with each other and with groups.
# Keys are dot-notation paths into the config (hydra-compatible).
params:
  model.init_args.learning_rate: [1.0e-4, 3.0e-4, 1.0e-3]
  trainer.max_epochs: [50, 100]

# groups: each group is ONE atomic dimension in the cartesian product.
# Entries within a choice dict always stay together — they are never split
# and recombined with other choices from the same group.
# Multiple groups are cartesian-producted with each other and with params.
groups:
  - - model.init_args.model.init_args.decoder_embed_dim: 128   # choice A
      model.init_args.model.init_args.decoder_depth: 2
      model.init_args.model.init_args.decoder_heads: 2
    - model.init_args.model.init_args.decoder_embed_dim: 64    # choice B
      model.init_args.model.init_args.decoder_depth: 1
      model.init_args.model.init_args.decoder_heads: 1
```

The example above produces **3 × 2 × 2 = 12 configs** (3 LRs × 2 epoch values × 2 decoder configurations).

See [test_sweep.yaml](test_sweep.yaml) for a working example.

## Running all generated configs

```bash
# Generate
python tools/sweeper.py conf/experiment/mae_omniglot_simple.yaml tools/test_sweep.yaml

# Run all
for cfg in sweeps/my_experiment/*.yaml; do
    [[ "$cfg" == *manifest* ]] && continue
    PYTHONPATH=src python src/cli.py fit --config "$cfg"
done

# Clean up sweep directory when done
rm -rf sweeps/my_experiment
```
