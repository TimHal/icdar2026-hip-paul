"""Lightning DataModule for loading pre-extracted cached features."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class CachedFeatureDatamodule(L.LightningDataModule):
    """Lightning DataModule for loading pre-extracted and cached features.

    This module loads features that were extracted and saved by
    scripts/extract_features.py. Features are expected to be in .pt files
    with the structure:
        {
            'features': Tensor[N, D],
            'labels': Tensor[N],
            'metadata': dict,
            'norm_params': dict (optional)
        }
    """

    def __init__(
        self,
        cache_dir: str = "./cache",
        dataset_name: str = "mnist_dinov2_vits14",
        batch_size: int = 64,
        num_workers: int = 4,
        seed: int = 42,
        train_val_split: tuple[float, float] = (0.9, 0.1),
    ):
        """Initialize the CachedFeatureDatamodule.

        Args:
            cache_dir: Directory containing cached feature files
            dataset_name: Base name for feature files (e.g., 'mnist_dinov2_vits14')
                          Files expected: {dataset_name}_train.pt, {dataset_name}_test.pt
            batch_size: Batch size for dataloaders
            num_workers: Number of data loading workers
            seed: Random seed for train/val split
            train_val_split: Fraction of data for (train, val)
        """
        super().__init__()
        self.save_hyperparameters()

        self.cache_dir = Path(cache_dir)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_val_split = train_val_split

        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Metadata
        self._metadata = None
        self._feature_dim = None
        self._num_classes = None

    def _load_cached_features(self, split: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Load cached features from file.

        Args:
            split: 'train' or 'test'

        Returns:
            Tuple of (features, labels, metadata)
        """
        cache_file = self.cache_dir / f"{self.dataset_name}_{split}.pt"
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Cache file not found: {cache_file}\n"
                f"Run scripts/extract_features.py first to generate cached features."
            )

        data = torch.load(cache_file, weights_only=False)
        features = data["features"]
        labels = data["labels"]
        metadata = data.get("metadata", {})

        return features, labels, metadata

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation, and testing."""
        if stage == "fit" or stage is None:
            # Load training features
            features, labels, metadata = self._load_cached_features("train")
            self._metadata = metadata
            self._feature_dim = features.shape[1]
            self._num_classes = len(torch.unique(labels))

            # Create TensorDataset with sample IDs
            ids = torch.arange(len(features))
            full_train = TensorDataset(features, labels, ids)

            # Split into train and val
            train_size = int(len(full_train) * self.train_val_split[0])
            val_size = len(full_train) - train_size

            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset = random_split(
                full_train, [train_size, val_size], generator=generator
            )

        if stage == "test" or stage is None:
            # Load test features
            features, labels, metadata = self._load_cached_features("test")
            if self._metadata is None:
                self._metadata = metadata
                self._feature_dim = features.shape[1]
                self._num_classes = len(torch.unique(labels))

            ids = torch.arange(len(features))
            self.test_dataset = TensorDataset(features, labels, ids)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def feature_dim(self) -> int:
        """Return the dimensionality of features."""
        if self._feature_dim is None:
            # Load metadata to get feature dim
            _, _, metadata = self._load_cached_features("train")
            self._feature_dim = metadata.get("feature_dim")
        return self._feature_dim

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        if self._num_classes is None:
            _, labels, _ = self._load_cached_features("train")
            self._num_classes = len(torch.unique(labels))
        return self._num_classes

    @property
    def metadata(self) -> dict:
        """Return metadata from the cached features."""
        if self._metadata is None:
            _, _, self._metadata = self._load_cached_features("train")
        return self._metadata
