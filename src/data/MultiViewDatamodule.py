"""
Multi-View DataModule for MAE training.

Provides multiple augmented views of each sample for multi-view MAE training.
All augmentations are handled in the dataloader, not the model.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import lightning as L

# Try to import torchvision datasets
try:
    from torchvision import datasets
except ImportError:
    datasets = None


class MultiViewTransform:
    """
    Creates multiple augmented views of an image.

    Augmentations suitable for handwritten characters:
    - Random rotation (small angles)
    - Random translation
    - Random scale
    - Gaussian blur
    - Gaussian noise
    - Elastic distortion (optional)

    Args:
        num_views: Number of views to generate
        img_size: Output image size
        rotation_range: Max rotation angle in degrees
        translate_range: Max translation as fraction of image size
        scale_range: Range of scaling factors (min, max)
        blur_sigma: Range of blur sigma (min, max)
        blur_prob: Probability of applying blur
        noise_std: Standard deviation of Gaussian noise
        noise_prob: Probability of applying noise
        brightness_range: Range of brightness adjustment
        contrast_range: Range of contrast adjustment
    """

    def __init__(
        self,
        num_views: int = 2,
        img_size: int = 64,
        rotation_range: float = 15.0,
        translate_range: float = 0.1,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        blur_sigma: Tuple[float, float] = (0.1, 2.0),
        blur_prob: float = 0.5,
        noise_std: float = 0.05,
        noise_prob: float = 0.5,
        brightness_range: float = 0.2,
        contrast_range: float = 0.2,
    ):
        self.num_views = num_views
        self.img_size = img_size
        self.rotation_range = rotation_range
        self.translate_range = translate_range
        self.scale_range = scale_range
        self.blur_sigma = blur_sigma
        self.blur_prob = blur_prob
        self.noise_std = noise_std
        self.noise_prob = noise_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

        # Base transform (resize and to tensor)
        self.base_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def _apply_augmentation(self, img: Tensor) -> Tensor:
        """Apply random augmentations to a single image tensor."""
        # Random rotation
        if self.rotation_range > 0:
            angle = torch.empty(1).uniform_(-self.rotation_range, self.rotation_range).item()
            img = TF.rotate(img, angle)

        # Random affine (scale and translate)
        if self.translate_range > 0 or self.scale_range != (1.0, 1.0):
            scale = torch.empty(1).uniform_(*self.scale_range).item()
            translate_x = torch.empty(1).uniform_(-self.translate_range, self.translate_range).item()
            translate_y = torch.empty(1).uniform_(-self.translate_range, self.translate_range).item()
            translate = (int(translate_x * self.img_size), int(translate_y * self.img_size))

            img = TF.affine(
                img, angle=0, translate=translate, scale=scale, shear=0,
                interpolation=T.InterpolationMode.BILINEAR
            )

        # Gaussian blur
        if self.blur_prob > 0 and torch.rand(1).item() < self.blur_prob:
            sigma = torch.empty(1).uniform_(*self.blur_sigma).item()
            kernel_size = int(2 * round(3 * sigma) + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = max(3, kernel_size)
            img = TF.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)

        # Gaussian noise
        if self.noise_std > 0 and torch.rand(1).item() < self.noise_prob:
            noise = torch.randn_like(img) * self.noise_std
            img = img + noise
            img = img.clamp(0, 1)

        # Brightness/contrast
        if self.brightness_range > 0:
            brightness = torch.empty(1).uniform_(
                1 - self.brightness_range, 1 + self.brightness_range
            ).item()
            img = TF.adjust_brightness(img, brightness)

        if self.contrast_range > 0:
            contrast = torch.empty(1).uniform_(
                1 - self.contrast_range, 1 + self.contrast_range
            ).item()
            img = TF.adjust_contrast(img, contrast)

        return img.clamp(0, 1)

    def __call__(self, img) -> List[Tensor]:
        """
        Generate multiple augmented views.

        Args:
            img: PIL Image or tensor

        Returns:
            List of augmented tensors, each [channels, height, width]
        """
        # Convert to tensor if needed
        if not isinstance(img, Tensor):
            img = self.base_transform(img)
        elif img.dim() == 2:
            img = img.unsqueeze(0)

        # Generate views
        views = []
        for _ in range(self.num_views):
            view = self._apply_augmentation(img.clone())
            views.append(view)

        return views


class SingleViewTransform:
    """
    Simple transform for single-view training (no augmentation).

    Args:
        img_size: Output image size
    """

    def __init__(self, img_size: int = 64):
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def __call__(self, img) -> Tensor:
        return self.transform(img)


class MultiViewDataset(Dataset):
    """
    Wraps a dataset to return multiple views per sample.

    Args:
        dataset: Base dataset
        transform: MultiViewTransform or similar
    """

    def __init__(self, dataset: Dataset, transform: MultiViewTransform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        views = self.transform(img)
        return views, label, idx


class SingleViewDataset(Dataset):
    """
    Wraps a dataset with single view transform.

    Args:
        dataset: Base dataset
        transform: SingleViewTransform or similar
    """

    def __init__(self, dataset: Dataset, transform: SingleViewTransform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img, label, idx


def multiview_collate_fn(batch):
    """
    Collate function for multi-view batches.

    Converts list of (views, label, idx) to stacked tensors.
    """
    views_list, labels, indices = zip(*batch)

    # views_list is list of [num_views] lists of tensors
    num_views = len(views_list[0])

    # Stack each view separately
    views = [torch.stack([v[i] for v in views_list]) for i in range(num_views)]

    labels = torch.tensor(labels, dtype=torch.long)
    indices = torch.tensor(indices, dtype=torch.long)

    return views, labels, indices


class MultiViewDatamodule(L.LightningDataModule):
    """
    Lightning DataModule for multi-view MAE training.

    Supports common image classification datasets with multi-view augmentation.

    Args:
        dataset_name: Name of torchvision dataset (MNIST, EMNIST, FashionMNIST, Omniglot, etc.)
        root_dir: Root directory for dataset storage
        img_size: Image size (assumes square)
        num_views: Number of augmented views per sample
        batch_size: Batch size
        num_workers: Number of data loading workers
        augmentation_config: Dict of augmentation parameters
        split: Dataset split (for EMNIST: 'balanced', 'letters', etc.)
    """

    SUPPORTED_DATASETS = {
        "MNIST": "MNIST",
        "FashionMNIST": "FashionMNIST",
        "KMNIST": "KMNIST",
        "EMNIST": "EMNIST",
        "QMNIST": "QMNIST",
        "Omniglot": "Omniglot",
    }

    def __init__(
        self,
        dataset_name: str = "MNIST",
        root_dir: str = "./data",
        img_size: int = 64,
        num_views: int = 2,
        batch_size: int = 64,
        num_workers: int = 4,
        augmentation_config: Optional[dict] = None,
        split: str = "balanced",
        persistent_workers: bool = True,
        pin_memory: bool = True,
        download: bool = True,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.img_size = img_size
        self.num_views = num_views
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.persistent_workers = persistent_workers and num_workers > 0
        self.pin_memory = pin_memory
        self.download = download

        # Default augmentation config
        default_aug = {
            "rotation_range": 15.0,
            "translate_range": 0.1,
            "scale_range": (0.9, 1.1),
            "blur_sigma": (0.1, 2.0),
            "blur_prob": 0.5,
            "noise_std": 0.05,
            "noise_prob": 0.3,
            "brightness_range": 0.1,
            "contrast_range": 0.1,
        }
        if augmentation_config:
            default_aug.update(augmentation_config)
        self.augmentation_config = default_aug

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage."""
        if datasets is None:
            raise ImportError("torchvision is required for MultiViewDatamodule")

        # Create transforms
        if self.num_views > 1:
            train_transform = MultiViewTransform(
                num_views=self.num_views,
                img_size=self.img_size,
                **self.augmentation_config,
            )
        else:
            train_transform = SingleViewTransform(img_size=self.img_size)

        val_transform = SingleViewTransform(img_size=self.img_size)

        # Get dataset class
        dataset_cls = getattr(datasets, self.dataset_name, None)
        if dataset_cls is None:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Create datasets with dataset-specific kwargs
        dataset_kwargs = {"root": self.root_dir, "download": self.download}
        if self.dataset_name == "EMNIST":
            dataset_kwargs["split"] = self.split

        # Handle Omniglot (uses background=True/False instead of train=True/False)
        if self.dataset_name == "Omniglot":
            if stage == "fit" or stage is None:
                train_base = dataset_cls(background=True, **dataset_kwargs)
                val_base = dataset_cls(background=False, **dataset_kwargs)

                if self.num_views > 1:
                    self.train_dataset = MultiViewDataset(train_base, train_transform)
                else:
                    self.train_dataset = SingleViewDataset(train_base, train_transform)

                self.val_dataset = SingleViewDataset(val_base, val_transform)

            if stage == "test" or stage is None:
                test_base = dataset_cls(background=False, **dataset_kwargs)
                self.test_dataset = SingleViewDataset(test_base, val_transform)
        else:
            # Standard train/test split datasets
            if stage == "fit" or stage is None:
                train_base = dataset_cls(train=True, **dataset_kwargs)
                val_base = dataset_cls(train=False, **dataset_kwargs)

                if self.num_views > 1:
                    self.train_dataset = MultiViewDataset(train_base, train_transform)
                else:
                    self.train_dataset = SingleViewDataset(train_base, train_transform)

                self.val_dataset = SingleViewDataset(val_base, val_transform)

            if stage == "test" or stage is None:
                test_base = dataset_cls(train=False, **dataset_kwargs)
                self.test_dataset = SingleViewDataset(test_base, val_transform)

    def train_dataloader(self):
        collate = multiview_collate_fn if self.num_views > 1 else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    @property
    def num_classes(self) -> int:
        """Return number of classes in the dataset."""
        class_counts = {
            "MNIST": 10,
            "FashionMNIST": 10,
            "KMNIST": 10,
            "EMNIST": {"balanced": 47, "byclass": 62, "bymerge": 47, "letters": 26, "digits": 10, "mnist": 10},
            "QMNIST": 10,
            "Omniglot": 964,  # Background set has 964 characters (30 alphabets)
        }
        count = class_counts.get(self.dataset_name, 10)
        if isinstance(count, dict):
            return count.get(self.split, 47)
        return count

    @property
    def in_channels(self) -> int:
        """Return number of input channels."""
        return 1  # All supported datasets are grayscale
