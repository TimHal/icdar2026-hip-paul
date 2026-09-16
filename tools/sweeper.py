#!/usr/bin/env python3
"""sweeper.py — Generate one config YAML per sweep combination.

Accepts a base config YAML and a sweep definition YAML, then writes one
merged config per parameter combination to an output directory.

Usage:
    python tools/sweeper.py <base_config.yaml> <sweep_config.yaml>

Sweep definition format:

    output_dir: path/to/output

    # Independent params: cartesian-producted with each other and with groups.
    # Keys use dot-notation (hydra-compatible path into the config).
    params:
      model.init_args.learning_rate: [1.0e-4, 3.0e-4]
      trainer.max_epochs: [50, 100]

    # Groups: each group is ONE atomic dimension in the cartesian product.
    # Params within a choice stay together — they are never split and recombined.
    # Multiple groups are cartesian-producted with each other and with params.
    groups:
      - - model.init_args.model.init_args.decoder_embed_dim: 128  # choice A
          model.init_args.model.init_args.decoder_depth: 2
        - model.init_args.model.init_args.decoder_embed_dim: 64   # choice B
          model.init_args.model.init_args.decoder_depth: 1

    # The above produces 2 * 2 * 2 = 8 configs.
"""

import argparse
import itertools
from pathlib import Path

import yaml
from omegaconf import OmegaConf


def build_dimensions(sweep: dict) -> list:
    """Return a list of dimensions for the cartesian product.

    Each dimension is a list of override dicts.  One dict is selected from
    each dimension per generated config.
    """
    dims = []

    params = sweep.get("params", {})
    if isinstance(params, list):
        params = {k: v for d in params for k, v in d.items()}
    for key, values in params.items():
        if not isinstance(values, list):
            values = [values]
        dims.append([{key: v} for v in values])

    for group in sweep.get("groups", []):
        # group is List[Dict]: each dict is one atomic choice for this dimension
        dims.append(group)

    return dims


def apply_overrides(cfg, overrides: dict) -> None:
    """Apply dot-notation overrides to an OmegaConf DictConfig in-place."""
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=True, force_add=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate one config YAML per sweep combination.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("base_config", type=Path, help="Base config YAML file")
    parser.add_argument("sweep_config", type=Path, help="Sweep definition YAML file")
    args = parser.parse_args()

    if not args.base_config.exists():
        parser.error(f"Base config not found: {args.base_config}")
    if not args.sweep_config.exists():
        parser.error(f"Sweep config not found: {args.sweep_config}")

    # Load base config via OmegaConf (preserves structure and scalar types)
    base = OmegaConf.load(args.base_config)

    # Load sweep definition as plain YAML (it is meta-config, not an app config)
    with open(args.sweep_config) as f:
        sweep = yaml.safe_load(f)

    if "output_dir" not in sweep:
        parser.error("Sweep config must specify 'output_dir'.")

    output_dir = Path(sweep["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dims = build_dimensions(sweep)
    if not dims:
        print("No sweep parameters defined. Nothing to do.")
        return

    base_stem = args.base_config.stem
    combinations = list(itertools.product(*dims))

    print(f"Base config : {args.base_config}")
    print(f"Sweep config: {args.sweep_config}")
    print(f"Output dir  : {output_dir}")
    print(f"Dimensions  : {[len(d) for d in dims]}  →  {len(combinations)} configs\n")

    manifest = {}

    for i, combo in enumerate(combinations):
        # Merge all override dicts from this combination (later entries win)
        merged: dict = {}
        for override_dict in combo:
            merged.update(override_dict)

        # Deep-copy base config and apply overrides
        cfg = OmegaConf.create(
            OmegaConf.to_container(base, resolve=False, throw_on_missing=False)
        )
        apply_overrides(cfg, merged)

        out_path = output_dir / f"{base_stem}_{i:04d}.yaml"
        out_path.write_text(OmegaConf.to_yaml(cfg))

        manifest[f"{base_stem}_{i:04d}.yaml"] = merged
        print(f"  [{i:04d}] {out_path.name}  {merged}")

    # Write manifest so users can trace which file corresponds to which overrides
    manifest_path = output_dir / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    print(f"\n{len(combinations)} configs written to {output_dir}/")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
