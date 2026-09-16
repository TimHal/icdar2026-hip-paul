"""MAE-based feature extractor.

Loads a pretrained MAE model from checkpoint and extracts features
using the encoder (without masking).
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from feature_extractors.base import FeatureExtractor
from mae.mae_model import MaskedAutoencoder, MAEConfig
from mae.multiview_mae import MultiViewMAE


class MAEFeatureExtractor(FeatureExtractor):
    """
    Feature extractor using a pretrained MAE encoder.

    Loads a trained MAE model from checkpoint and uses the encoder
    (without masking) to extract features. Compatible with the existing
    FeatureExtractor interface for use with ClasswiseAETask.

    There are two ways to create an MAEFeatureExtractor:

    1. From checkpoint (recommended):
        extractor = MAEFeatureExtractor.from_checkpoint("path/to/checkpoint.ckpt")

    2. With explicit config (for new models):
        extractor = MAEFeatureExtractor(
            config=MAEConfig(img_size=64, patch_size=8, embed_dim=128, ...)
        )

    Args:
        config: MAEConfig with model architecture parameters.
            If not provided, uses default MAEConfig().
        checkpoint_path: Optional path to load weights from. If the checkpoint
            contains 'mae_config', it will be used instead of the config argument.
        pooling: Feature pooling method ('cls', 'mean', 'both')
        device: Device to load model on
    """

    def __init__(
        self,
        config: Optional[MAEConfig] = None,
        checkpoint_path: Optional[str] = None,
        pooling: str = "cls",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self._pooling = pooling
        self._device = torch.device(device)

        # Load from checkpoint with stored config (preferred path)
        if checkpoint_path is not None:
            self.model, self._config = self._load_from_checkpoint(checkpoint_path)
        else:
            # Create model from provided config
            self._config = config or MAEConfig()
            self.model = MaskedAutoencoder.from_config(self._config)

        self.model = self.model.to(self._device)
        self.model.eval()

        # Compute feature dim based on pooling
        if pooling == "both":
            self._feature_dim = self._config.embed_dim * 2
        else:
            self._feature_dim = self._config.embed_dim

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        pooling: str = "cls",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> "MAEFeatureExtractor":
        """Create a feature extractor from a checkpoint.

        This is the recommended way to load a pretrained MAE for feature extraction.
        The architecture is loaded from the stored config in the checkpoint.

        Args:
            checkpoint_path: Path to Lightning checkpoint (.ckpt file)
            pooling: Feature pooling method ('cls', 'mean', 'both')
            device: Device to load model on

        Returns:
            MAEFeatureExtractor ready for feature extraction
        """
        return cls(
            checkpoint_path=checkpoint_path,
            pooling=pooling,
            device=device,
        )

    def _load_from_checkpoint(self, checkpoint_path: str) -> tuple[MaskedAutoencoder, MAEConfig]:
        """Load MAE model from Lightning checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Tuple of (model, config)
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})

        # Try config-based loading first (preferred)
        if "mae_config" in hparams:
            config = MAEConfig.from_dict(hparams["mae_config"])
            model = MaskedAutoencoder.from_config(config)

            # Load state dict
            state_dict = checkpoint["state_dict"]
            prefix = self._get_state_dict_prefix(state_dict)
            mae_state_dict = self._extract_mae_state_dict(state_dict, prefix)
            model.load_state_dict(mae_state_dict)

            return model, config

    def _get_state_dict_prefix(self, state_dict: dict) -> str:
        """Determine the prefix for MAE weights in state_dict."""
        if any(k.startswith("model.mae.") for k in state_dict.keys()):
            return "model.mae."
        elif any(k.startswith("model.") for k in state_dict.keys()):
            return "model."
        return ""

    def _extract_mae_state_dict(self, state_dict: dict, prefix: str) -> dict:
        """Extract MAE model weights from full state_dict."""
        if prefix:
            return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        return state_dict

    def extract(self, images: Tensor) -> Tensor:
        """
        Extract features from images.

        Args:
            images: Input images [batch, channels, height, width]

        Returns:
            features: Extracted features [batch, feature_dim]
        """
        images = images.to(self._device)

        with torch.no_grad():
            if self._pooling == "cls" and self._config.use_cls_token:
                features = self.model.encode(images)
            elif self._pooling == "mean":
                tokens = self.model.encode_patches(images)
                features = tokens.mean(dim=1)
            elif self._pooling == "both" and self._config.use_cls_token:
                cls_features = self.model.encode(images)
                tokens = self.model.encode_patches(images)
                mean_features = tokens.mean(dim=1)
                features = torch.cat([cls_features, mean_features], dim=1)
            else:
                # Default to mean pooling
                tokens = self.model.encode_patches(images)
                features = tokens.mean(dim=1)

        return features

    def extract_from_dataloader(
        self,
        dataloader: DataLoader,
        device: Optional[torch.device] = None,
        show_progress: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """
        Extract features from all samples in a DataLoader.

        Args:
            dataloader: DataLoader providing (images, labels, ...) batches
            device: Device to run extraction on (default: use self.device)
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (features, labels) tensors
        """
        if device is not None:
            self.model = self.model.to(device)
            self._device = device

        all_features = []
        all_labels = []

        iterator = tqdm(dataloader, desc="Extracting MAE features") if show_progress else dataloader

        self.model.eval()
        with torch.no_grad():
            for batch in iterator:
                # Handle different batch formats
                if len(batch) == 2:
                    images, labels = batch
                elif len(batch) == 3:  # for the custom (X, y, ID) dataloaders
                    images, labels, _ = batch
                else:
                    images, labels = batch[0], batch[1]

                images = images.to(self._device)
                features = self.extract(images)
                all_features.append(features.cpu())
                all_labels.append(labels)

        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

    def get_transform(self, image_size: int = 224) -> transforms.Compose:
        """Get the image transform for MAE feature extraction.

        Args:
            image_size: Size to resize images to (default: 224)

        Returns:
            torchvision transforms.Compose object with the necessary preprocessing
        """

        mae_expected_size = self.model.img_size
        mae_expected_channels = self.model.in_chans

        return transforms.Compose(
            [
                transforms.Resize((mae_expected_size, mae_expected_size)),
                transforms.Grayscale(num_output_channels=mae_expected_channels)
                if mae_expected_channels == 1
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ]
        )

    def get_transform_for_grayscale(self, image_size: int = 224) -> transforms.Compose:
        """Get the image transform for MAE feature extraction with grayscale conversion.

        Args:
            image_size: Size to resize images to (default: 224)

        Returns:
            torchvision transforms.Compose object with the necessary preprocessing including grayscale conversion
        """

        mae_expected_size = self.model.img_size
        mae_expected_channels = self.model.in_chans

        return transforms.Compose(
            [
                transforms.Resize((mae_expected_size, mae_expected_size)),
                transforms.Grayscale(num_output_channels=mae_expected_channels),
                transforms.ToTensor(),
            ]
        )

    @property
    def config(self) -> MAEConfig:
        """Return the model configuration."""
        return self._config

    @property
    def feature_dim(self) -> int:
        """Return the dimensionality of extracted features."""
        return self._feature_dim

    @property
    def model_name(self) -> str:
        """Return the model name for identification."""
        return (
            f"mae_vit_{self._config.embed_dim}d_"
            f"{self._config.encoder_depth}l_{self._config.img_size}px"
        )

    @property
    def device(self) -> torch.device:
        """Return the current device."""
        return self._device

    @device.setter
    def device(self, value: torch.device):
        """Set the device and move model."""
        self._device = value
        self.model = self.model.to(value)

    def to(self, device: Union[str, torch.device]) -> "MAEFeatureExtractor":
        """Move model to device."""
        self._device = torch.device(device)
        self.model = self.model.to(self._device)
        return self
