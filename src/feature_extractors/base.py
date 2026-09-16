"""Abstract base class for feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class FeatureExtractor(ABC):
    """Abstract base class for feature extraction from foundation models.

    Feature extractors wrap pretrained models (DINO, CLIP) and provide
    a unified interface for extracting features from images.
    """

    @abstractmethod
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch of images.

        Args:
            images: Input images [batch_size, channels, height, width]

        Returns:
            Features [batch_size, feature_dim]
        """
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the dimensionality of extracted features."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model for identification."""
        pass

    def extract_from_dataloader(
        self,
        dataloader: DataLoader,
        device: Optional[torch.device] = None,
        show_progress: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract features from all samples in a DataLoader.

        Args:
            dataloader: DataLoader providing (images, labels) batches
            device: Device to run extraction on (default: auto-detect)
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (features, labels) tensors
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        all_features = []
        all_labels = []

        iterator = tqdm(dataloader, desc="Extracting features") if show_progress else dataloader

        with torch.no_grad():
            for images, labels in iterator:
                images = images.to(device)
                features = self.extract(images)
                all_features.append(features.cpu())
                all_labels.append(labels)

        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

    def normalize_features(
        self, features: torch.Tensor, method: str = "minmax"
    ) -> tuple[torch.Tensor, dict]:
        """Normalize features to [0, 1] range.

        Args:
            features: Features tensor [N, D]
            method: Normalization method ('minmax' or 'zscore')

        Returns:
            Tuple of (normalized_features, normalization_params)
        """
        if method == "minmax":
            min_vals = features.min(dim=0, keepdim=True).values
            max_vals = features.max(dim=0, keepdim=True).values
            range_vals = max_vals - min_vals
            range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
            normalized = (features - min_vals) / range_vals
            params = {"method": "minmax", "min": min_vals, "max": max_vals}
        elif method == "zscore":
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True)
            std = torch.where(std == 0, torch.ones_like(std), std)
            normalized = (features - mean) / std
            params = {"method": "zscore", "mean": mean, "std": std}
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized.clamp(0, 1), params

    def apply_normalization(
        self, features: torch.Tensor, params: dict
    ) -> torch.Tensor:
        """Apply pre-computed normalization to features.

        Args:
            features: Features tensor [N, D]
            params: Normalization parameters from normalize_features()

        Returns:
            Normalized features
        """
        method = params["method"]
        if method == "minmax":
            min_vals = params["min"]
            max_vals = params["max"]
            range_vals = max_vals - min_vals
            range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
            normalized = (features - min_vals) / range_vals
        elif method == "zscore":
            mean = params["mean"]
            std = params["std"]
            std = torch.where(std == 0, torch.ones_like(std), std)
            normalized = (features - mean) / std
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized.clamp(0, 1)
