"""Lightning DataModule for torchvision datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class OracleMNIST(datasets.MNIST):
    """Oracle-MNIST dataset (ancient Chinese characters, MNIST-compatible format).

    Expects data at {root}/OracleMNIST/raw/ in standard MNIST binary format.
    """

    mirrors = []  # No auto-download; data must be pre-placed
    resources = [
        ("train-images-idx3-ubyte.gz", None),
        ("train-labels-idx1-ubyte.gz", None),
        ("t10k-images-idx3-ubyte.gz", None),
        ("t10k-labels-idx1-ubyte.gz", None),
    ]

    @property
    def raw_folder(self) -> str:
        return str(Path(self.root) / "OracleMNIST" / "raw")


class IndexedDataset(Dataset):
    """Wrapper that adds sample indices to dataset returns."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, idx: int):
        data, label = self.dataset[idx]
        return data, label, idx

    def __len__(self) -> int:
        return len(self.dataset)


class TorchvisionDatamodule(L.LightningDataModule):
    """Lightning DataModule for MNIST-style torchvision datasets.

    Supports:
        - MNIST, FashionMNIST, KMNIST, EMNIST, QMNIST
        - CIFAR10, CIFAR100
        - SVHN, STL10

    Features:
        - Automatic grayscale to RGB conversion
        - Configurable train/val split
        - EMNIST split selection
    """

    GRAYSCALE_DATASETS = {"MNIST", "FashionMNIST", "KMNIST", "EMNIST", "QMNIST", "OracleMNIST", "OMNIGLOT20", "OMNIGLOT50"}
    RGB_DATASETS = {"CIFAR10", "CIFAR100", "SVHN", "STL10"}
    # Datasets stored as ImageFolder: {root_dir}/{dataset_name}/{train|test}/{class}/
    IMAGEFOLDER_DATASETS = {"KMNIST", "OracleMNIST", "OMNIGLOT20", "OMNIGLOT50"}
    SUPPORTED_DATASETS = GRAYSCALE_DATASETS | RGB_DATASETS

    def __init__(
        self,
        dataset_name: str = "MNIST",
        root_dir: str = "/data/thallybu/datasets",
        img_size: tuple[int, int] = (224, 224),
        img_mode: Literal["L", "RGB"] = "RGB",
        batch_size: int = 64,
        num_workers: int = 4,
        seed: int = 42,
        train_val_split: tuple[float, float] = (0.9, 0.1),
        emnist_split: str = "digits",
        norm_mean: Optional[tuple[float, ...]] = None,
        norm_std: Optional[tuple[float, ...]] = None,
        flatten: bool = False,
        raw_pixels: bool = False,
        pin_memory: bool = False,
    ):
        """Initialize the DataModule.

        Args:
            dataset_name: Name of the dataset (e.g., 'MNIST', 'EMNIST')
            root_dir: Root directory for dataset storage
            img_size: Target image size (height, width)
            img_mode: Image mode ('L' for grayscale, 'RGB' for color)
            batch_size: Batch size for dataloaders
            num_workers: Number of data loading workers
            seed: Random seed for train/val split
            train_val_split: Fraction of data for (train, val)
            emnist_split: EMNIST split if using EMNIST
            norm_mean: Normalization mean (default: ImageNet for RGB)
            norm_std: Normalization std (default: ImageNet for RGB)
            flatten: Flatten images to 1D vectors (default: False)
            raw_pixels: Keep original size and use [0,1] range without normalization (default: False)
            pin_memory: Whether to pin memory in dataloaders (default: False)
        """
        super().__init__()
        self.save_hyperparameters()

        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Supported: {self.SUPPORTED_DATASETS}"
            )

        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.img_size = img_size
        self.img_mode = img_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_val_split = train_val_split
        self.emnist_split = emnist_split
        self.flatten = flatten
        self.raw_pixels = raw_pixels
        self.pin_memory = pin_memory

        # Set default normalization values
        if norm_mean is None or norm_std is None:
            if img_mode == "RGB":
                self.norm_mean = (0.485, 0.456, 0.406)
                self.norm_std = (0.229, 0.224, 0.225)
            else:
                self.norm_mean = (0.5,)
                self.norm_std = (0.5,)
        else:
            self.norm_mean = norm_mean
            self.norm_std = norm_std

        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _get_base_transform(self) -> transforms.Compose:
        """Get base transform for the dataset."""
        transform_list = []

        # Only resize if not using raw pixels
        if not self.raw_pixels:
            transform_list.append(transforms.Resize(self.img_size))

        # Handle grayscale datasets
        is_grayscale = self.dataset_name in self.GRAYSCALE_DATASETS
        is_imagefolder = self.dataset_name in self.IMAGEFOLDER_DATASETS

        if is_grayscale and self.img_mode == "RGB":
            transform_list.append(transforms.Grayscale(num_output_channels=3))
        elif is_grayscale and self.img_mode == "L" and is_imagefolder:
            # ImageFolder loads PNGs as RGB; convert back to grayscale
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        elif not is_grayscale and self.img_mode == "L":
            transform_list.append(transforms.Grayscale(num_output_channels=1))

        # Handle EMNIST rotation (EMNIST images need rotation)
        if self.dataset_name == "EMNIST":
            transform_list.append(transforms.Lambda(lambda x: transforms.functional.rotate(x, -90)))
            transform_list.append(transforms.Lambda(lambda x: transforms.functional.hflip(x)))

        # ToTensor converts to [0, 1] range
        transform_list.append(transforms.ToTensor())

        # Only normalize if not using raw pixels (autoencoders prefer [0, 1])
        if not self.raw_pixels:
            transform_list.append(transforms.Normalize(mean=self.norm_mean, std=self.norm_std))

        # Flatten if requested
        if self.flatten:
            transform_list.append(transforms.Lambda(lambda x: x.flatten()))

        return transforms.Compose(transform_list)

    def _get_dataset_class(self):
        """Get the torchvision dataset class."""
        mapping = {
            "MNIST": datasets.MNIST,
            "FashionMNIST": datasets.FashionMNIST,
            "KMNIST": datasets.KMNIST,
            "EMNIST": datasets.EMNIST,
            "QMNIST": datasets.QMNIST,
            "CIFAR10": datasets.CIFAR10,
            "CIFAR100": datasets.CIFAR100,
            "SVHN": datasets.SVHN,
            "STL10": datasets.STL10,
            "OracleMNIST": OracleMNIST,
        }
        return mapping[self.dataset_name]

    def _create_dataset(self, train: bool, transform):
        """Create a dataset instance."""
        # ImageFolder-based datasets: {root_dir}/{dataset_name}/{train|test}/{class}/
        if self.dataset_name in self.IMAGEFOLDER_DATASETS:
            split = "train" if train else "test"
            folder = Path(self.root_dir) / self.dataset_name / split
            return datasets.ImageFolder(str(folder), transform=transform)

        dataset_class = self._get_dataset_class()
        kwargs = {
            "root": self.root_dir,
            "transform": transform,
            "download": True,
        }

        if self.dataset_name == "QMNIST":
            kwargs["what"] = "train" if train else "test"
        elif self.dataset_name in ("SVHN", "STL10"):
            kwargs["split"] = "train" if train else "test"
        else:
            kwargs["train"] = train

        if self.dataset_name == "EMNIST":
            kwargs["split"] = self.emnist_split

        return dataset_class(**kwargs)

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation, and testing."""
        transform = self._get_base_transform()

        if stage == "fit" or stage is None:
            # Load full training data and wrap with IndexedDataset
            base_train = self._create_dataset(train=True, transform=transform)
            full_train = IndexedDataset(base_train)

            # Split into train and val
            train_size = int(len(full_train) * self.train_val_split[0])
            val_size = len(full_train) - train_size

            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset = random_split(
                full_train, [train_size, val_size], generator=generator
            )

        if stage == "test" or stage is None:
            base_test = self._create_dataset(train=False, transform=transform)
            self.test_dataset = IndexedDataset(base_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        class_counts = {
            "MNIST": 10,
            "FashionMNIST": 10,
            "KMNIST": 10,
            "CIFAR10": 10,
            "CIFAR100": 100,
            "SVHN": 10,
            "STL10": 10,
            "OracleMNIST": 10,
            "OMNIGLOT20": 20,
            "OMNIGLOT50": 50,
        }
        if self.dataset_name in class_counts:
            return class_counts[self.dataset_name]
        elif self.dataset_name == "EMNIST":
            emnist_counts = {
                "digits": 10,
                "letters": 26,
                "balanced": 47,
                "byclass": 62,
                "bymerge": 47,
                "mnist": 10,
            }
            return emnist_counts[self.emnist_split]
        elif self.dataset_name == "QMNIST":
            return 10
        else:
            raise ValueError(f"Unknown number of classes for {self.dataset_name}")

    @property
    def feature_dim(self) -> int:
        """Return the feature dimension (for flattened images).

        Only available when flatten=True.
        """
        if not self.flatten:
            raise ValueError("feature_dim only available when flatten=True")

        # Original dataset sizes (C, H, W)
        original_sizes = {
            "MNIST": (1, 28, 28),
            "FashionMNIST": (1, 28, 28),
            "KMNIST": (1, 28, 28),
            "EMNIST": (1, 28, 28),
            "QMNIST": (1, 28, 28),
            "CIFAR10": (3, 32, 32),
            "CIFAR100": (3, 32, 32),
            "SVHN": (3, 32, 32),
            "STL10": (3, 96, 96),
            "OracleMNIST": (1, 28, 28),
            "OMNIGLOT20": (1, 105, 105),
            "OMNIGLOT50": (1, 105, 105),
        }

        if self.raw_pixels:
            # Use original dataset size
            c, h, w = original_sizes.get(self.dataset_name, (1, 28, 28))
        else:
            # Use resized dimensions
            h, w = self.img_size
            c = 3 if self.img_mode == "RGB" else 1

        return c * h * w
