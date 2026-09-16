"""Utilities for loading models from checkpoints with config-based instantiation.

This module provides functions to load trained models from Lightning checkpoints
using stored configuration, avoiding the need to infer architecture from state_dict.
"""

import importlib
from pathlib import Path
from omegaconf import OmegaConf
from typing import Any, Dict, Optional, Union

import torch


def import_class(class_path: str):
    """Import a class from a dotted path string.

    Args:
        class_path: Fully qualified class path (e.g., 'mae.mae_model.MaskedAutoencoder')

    Returns:
        The imported class
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def instantiate_from_config(config_dict: Dict[str, Any]) -> Any:
    """Recursively instantiate a class from a config dict with class_path/init_args.

    Supports Lightning CLI style configs with nested class instantiation.

    Args:
        config_dict: Dict with 'class_path' and optional 'init_args' keys

    Returns:
        Instantiated object
    """
    if not isinstance(config_dict, dict):
        return config_dict

    if "class_path" not in config_dict:
        return config_dict

    class_path = config_dict["class_path"]
    init_args = config_dict.get("init_args", {})

    # Recursively instantiate nested class configs
    processed_args = {}
    for key, value in init_args.items():
        if isinstance(value, dict) and "class_path" in value:
            processed_args[key] = instantiate_from_config(value)
        else:
            processed_args[key] = value

    cls = import_class(class_path)
    return cls(**processed_args)


def load_from_config(config_path: str, checkpoint_path: str | None = None):
    """
    Load model and datamodule from a config file, optionally loading checkpoint weights.

    Args:
        config_path: Path to the experiment YAML config
        checkpoint_path: Optional path to checkpoint file (.ckpt)

    Returns:
        tuple: (model/task, datamodule)
    """
    config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(config, resolve=True)

    if not isinstance(config_dict, dict):
        raise ValueError("Config file must contain a dictionary at the top level.")

    # Instantiate model/task
    model = instantiate_from_config(config_dict["model"])

    # Instantiate datamodule
    datamodule = instantiate_from_config(config_dict["data"])

    # Load checkpoint weights
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])  # type: ignore
        print(f"Loaded weights from: {checkpoint_path}")

    return model, datamodule


def load_checkpoint_config(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """Extract hyperparameters/config from a Lightning checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file

    Returns:
        Dictionary of hyperparameters stored in the checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("hyper_parameters", {})


def get_checkpoint_model_config(
    checkpoint_path: Union[str, Path],
    config_key: str = "mae_config",
) -> Optional[Dict[str, Any]]:
    """Get the model config dict from a checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file
        config_key: Key name for the config in hyper_parameters

    Returns:
        Config dict if found, None otherwise
    """
    hparams = load_checkpoint_config(checkpoint_path)
    return hparams.get(config_key)


def checkpoint_has_config(
    checkpoint_path: Union[str, Path],
    config_key: str = "mae_config",
) -> bool:
    """Check if a checkpoint contains stored model config.

    Args:
        checkpoint_path: Path to the .ckpt file
        config_key: Key name for the config in hyper_parameters

    Returns:
        True if config is present
    """
    return get_checkpoint_model_config(checkpoint_path, config_key) is not None


def load_mae_model(
    checkpoint_path: Union[str, Path],
    map_location: str = "cpu",
):
    """Load a MaskedAutoencoder from checkpoint using stored config.

    This is a convenience wrapper around MaskedAutoencoder.from_checkpoint().

    Args:
        checkpoint_path: Path to Lightning checkpoint
        map_location: Device to load tensors to

    Returns:
        MaskedAutoencoder with loaded weights
    """
    from mae.mae_model import MaskedAutoencoder

    return MaskedAutoencoder.from_checkpoint(str(checkpoint_path), map_location)
