"""Prototype management for self-supervised class-wise autoencoder training.

Provides functionality to manage prototype samples that anchor the
self-supervised training process.
"""

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset


class PrototypeProvider:
    """Manages prototype samples for each class in self-supervised training.

    Prototypes are representative samples that anchor each class autoencoder.
    They can be provided explicitly or sampled from a dataset.

    Args:
        num_classes: Number of classes
        feature_dim: Feature dimension (for validation)
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: Optional[int] = None,
    ):
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Storage: class_idx -> list of (sample_id, features)
        self._prototypes: dict[int, list[tuple[int, Tensor]]] = {
            c: [] for c in range(num_classes)
        }

        # Cached tensors
        self._features_cache: Optional[Tensor] = None
        self._classes_cache: Optional[Tensor] = None
        self._ids_cache: Optional[Tensor] = None

    def register_prototype(
        self,
        class_idx: int,
        sample_id: int,
        features: Tensor,
    ) -> None:
        """Register a prototype for a class.

        Args:
            class_idx: Class index (0 to num_classes-1)
            sample_id: Sample ID for this prototype
            features: Feature tensor [feature_dim]
        """
        if class_idx < 0 or class_idx >= self.num_classes:
            raise ValueError(f"Invalid class_idx: {class_idx}")

        # Validate feature dimension
        if self.feature_dim is not None and features.shape[0] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, "
                f"got {features.shape[0]}"
            )

        if self.feature_dim is None:
            self.feature_dim = features.shape[0]

        self._prototypes[class_idx].append((sample_id, features.clone()))
        self._invalidate_cache()

    def register_prototypes_batch(
        self,
        class_indices: Tensor,
        sample_ids: Tensor,
        features: Tensor,
    ) -> None:
        """Register multiple prototypes at once.

        Args:
            class_indices: [N] tensor of class indices
            sample_ids: [N] tensor of sample IDs
            features: [N, feature_dim] tensor of features
        """
        for i in range(len(class_indices)):
            self.register_prototype(
                class_idx=class_indices[i].item(),
                sample_id=sample_ids[i].item(),
                features=features[i],
            )

    def get_prototype_features(self) -> tuple[Tensor, Tensor, Tensor]:
        """Get all prototype features and their metadata.

        Returns:
            features: [num_prototypes, feature_dim]
            classes: [num_prototypes] class indices
            sample_ids: [num_prototypes] sample IDs
        """
        if self._features_cache is not None:
            return self._features_cache, self._classes_cache, self._ids_cache

        all_features = []
        all_classes = []
        all_ids = []

        for class_idx in range(self.num_classes):
            for sample_id, features in self._prototypes[class_idx]:
                all_features.append(features)
                all_classes.append(class_idx)
                all_ids.append(sample_id)

        if not all_features:
            # Return empty tensors
            return (
                torch.tensor([]),
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.long),
            )

        self._features_cache = torch.stack(all_features)
        self._classes_cache = torch.tensor(all_classes, dtype=torch.long)
        self._ids_cache = torch.tensor(all_ids, dtype=torch.long)

        return self._features_cache, self._classes_cache, self._ids_cache

    def get_prototypes_for_class(self, class_idx: int) -> tuple[Tensor, Tensor]:
        """Get prototypes for a specific class.

        Args:
            class_idx: Class index

        Returns:
            features: [N, feature_dim]
            sample_ids: [N]
        """
        protos = self._prototypes.get(class_idx, [])
        if not protos:
            return torch.tensor([]), torch.tensor([], dtype=torch.long)

        features = torch.stack([f for _, f in protos])
        sample_ids = torch.tensor([sid for sid, _ in protos], dtype=torch.long)

        return features, sample_ids

    def get_prototype_count(self, class_idx: Optional[int] = None) -> int:
        """Get number of prototypes.

        Args:
            class_idx: If provided, count for specific class; otherwise total

        Returns:
            Number of prototypes
        """
        if class_idx is not None:
            return len(self._prototypes.get(class_idx, []))
        return sum(len(p) for p in self._prototypes.values())

    def get_prototype_ids(self) -> set[int]:
        """Get all prototype sample IDs."""
        ids = set()
        for protos in self._prototypes.values():
            for sample_id, _ in protos:
                ids.add(sample_id)
        return ids

    def _invalidate_cache(self) -> None:
        """Invalidate cached tensors."""
        self._features_cache = None
        self._classes_cache = None
        self._ids_cache = None

    @classmethod
    def from_indices(
        cls,
        prototype_indices: dict[int, list[int]],
        features: Tensor,
        sample_ids: Optional[Tensor] = None,
    ) -> "PrototypeProvider":
        """Create provider from explicit indices and features.

        Args:
            prototype_indices: {class_idx: [feature_indices]} mapping
            features: [N, feature_dim] tensor of all features
            sample_ids: Optional [N] tensor mapping feature index to sample ID

        Returns:
            Configured PrototypeProvider
        """
        num_classes = len(prototype_indices)
        feature_dim = features.shape[1]

        provider = cls(num_classes=num_classes, feature_dim=feature_dim)

        for class_idx, indices in prototype_indices.items():
            for idx in indices:
                sample_id = idx if sample_ids is None else sample_ids[idx].item()
                provider.register_prototype(
                    class_idx=class_idx,
                    sample_id=sample_id,
                    features=features[idx],
                )

        return provider

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        prototype_indices: dict[int, list[int]],
        feature_extractor=None,
    ) -> "PrototypeProvider":
        """Create provider by extracting features from dataset.

        Args:
            dataset: Dataset to extract prototypes from
            prototype_indices: {class_idx: [sample_indices]} mapping
            feature_extractor: Optional extractor; if None, assumes
                              dataset returns (features, label, idx)

        Returns:
            Configured PrototypeProvider
        """
        # Collect all indices
        all_indices = []
        index_to_class = {}
        for class_idx, indices in prototype_indices.items():
            for idx in indices:
                all_indices.append(idx)
                index_to_class[idx] = class_idx

        # Extract features
        all_features = []
        for idx in all_indices:
            sample = dataset[idx]
            if feature_extractor is not None:
                # Assume sample is (image, label, ...)
                image = sample[0].unsqueeze(0)
                features = feature_extractor.extract(image).squeeze(0)
            else:
                # Assume sample is (features, label, ...)
                features = sample[0]
            all_features.append(features)

        features = torch.stack(all_features)

        # Create provider
        num_classes = len(prototype_indices)
        provider = cls(num_classes=num_classes, feature_dim=features.shape[1])

        for i, idx in enumerate(all_indices):
            provider.register_prototype(
                class_idx=index_to_class[idx],
                sample_id=idx,
                features=features[i],
            )

        return provider

    @classmethod
    def from_labels(
        cls,
        features: Tensor,
        labels: Tensor,
        sample_ids: Tensor,
        prototypes_per_class: int = 1,
        selection: str = "first",
    ) -> "PrototypeProvider":
        """Create provider by selecting prototypes from labeled data.

        Args:
            features: [N, feature_dim] tensor
            labels: [N] tensor of class labels
            sample_ids: [N] tensor of sample IDs
            prototypes_per_class: Number of prototypes per class
            selection: Selection strategy - "first", "random", or "centroid"

        Returns:
            Configured PrototypeProvider
        """
        num_classes = labels.max().item() + 1
        feature_dim = features.shape[1]

        provider = cls(num_classes=num_classes, feature_dim=feature_dim)

        for class_idx in range(num_classes):
            mask = labels == class_idx
            class_features = features[mask]
            class_ids = sample_ids[mask]

            if len(class_features) == 0:
                continue

            if selection == "first":
                indices = list(range(min(prototypes_per_class, len(class_features))))
            elif selection == "random":
                perm = torch.randperm(len(class_features))
                indices = perm[:prototypes_per_class].tolist()
            elif selection == "centroid":
                # Select samples closest to centroid
                centroid = class_features.mean(dim=0, keepdim=True)
                distances = ((class_features - centroid) ** 2).sum(dim=1)
                indices = distances.argsort()[:prototypes_per_class].tolist()
            else:
                raise ValueError(f"Unknown selection strategy: {selection}")

            for i in indices:
                provider.register_prototype(
                    class_idx=class_idx,
                    sample_id=class_ids[i].item(),
                    features=class_features[i],
                )

        return provider

    @classmethod
    def from_pth_file(
        cls,
        pth_path: str | Path,
        image_shape: tuple[int, int, int] | None = None,
    ) -> "PrototypeProvider":
        """Create provider from .pth file containing images and class IDs.

        Args:
            pth_path: Path to .pth file with structure:
                      {"images": Tensor[N, C, H, W], "class_ids": Tensor[N]}
            image_shape: Optional (C, H, W) for validation

        Returns:
            Configured PrototypeProvider with synthetic IDs (negative integers)
        """
        data = torch.load(pth_path, weights_only=False)

        images = data["images"]  # [N, C, H, W]
        class_ids = data["class_ids"]  # [N]

        if image_shape is not None:
            expected = tuple(image_shape)
            actual = tuple(images.shape[1:])
            if expected != actual:
                raise ValueError(
                    f"Image shape mismatch: expected {expected}, got {actual}"
                )

        # Flatten images to [N, C*H*W]
        features = images.flatten(start_dim=1).float()

        num_classes = class_ids.max().item() + 1
        feature_dim = features.shape[1]

        provider = cls(num_classes=num_classes, feature_dim=feature_dim)

        for i in range(len(features)):
            provider.register_prototype(
                class_idx=int(class_ids[i].item()),
                sample_id=-(i + 1),  # Synthetic negative ID
                features=features[i],
            )

        return provider

    @classmethod
    def from_folder(
        cls,
        folder_path: str | Path,
        image_shape: tuple[int, int, int],
        img_mode: str = "L",
    ) -> "PrototypeProvider":
        """Create provider from folder with class subfolders containing images.

        Args:
            folder_path: Path to folder with structure:
                         folder/0/*.png, folder/1/*.png, etc.
                         or folder/class_0/*.png, folder/class_1/*.png, etc.
            image_shape: Required (C, H, W) for expected image dimensions
            img_mode: Image mode ("L" for grayscale, "RGB" for color)

        Returns:
            Configured PrototypeProvider with synthetic IDs (negative integers)
        """
        from PIL import Image
        from torchvision import transforms

        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise ValueError(f"Not a directory: {folder_path}")

        # Find class subfolders (0, 1, 2... or class_0, class_1...)
        class_folders = {}
        for subdir in sorted(folder_path.iterdir()):
            if not subdir.is_dir():
                continue
            name = subdir.name
            # Try to parse class index
            if name.isdigit():
                class_idx = int(name)
            elif name.startswith("class_") and name[6:].isdigit():
                class_idx = int(name[6:])
            else:
                continue
            class_folders[class_idx] = subdir

        if not class_folders:
            raise ValueError(f"No class subfolders found in {folder_path}")

        num_classes = max(class_folders.keys()) + 1
        c, h, w = image_shape
        feature_dim = c * h * w

        # Build transform
        transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),  # Converts to [0, 1]
        ])

        provider = cls(num_classes=num_classes, feature_dim=feature_dim)
        sample_counter = 0

        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

        for class_idx, class_dir in sorted(class_folders.items()):
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in image_extensions:
                    continue

                img = Image.open(img_path).convert(img_mode)
                tensor = transform(img)  # [C, H, W]
                features = tensor.flatten()  # [C*H*W]

                provider.register_prototype(
                    class_idx=class_idx,
                    sample_id=-(sample_counter + 1),  # Synthetic negative ID
                    features=features,
                )
                sample_counter += 1

        return provider

    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
            "prototypes": {
                class_idx: [(sid, f.tolist()) for sid, f in protos]
                for class_idx, protos in self._prototypes.items()
            },
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self.num_classes = state_dict["num_classes"]
        self.feature_dim = state_dict["feature_dim"]

        self._prototypes = {}
        for class_idx, protos in state_dict["prototypes"].items():
            self._prototypes[int(class_idx)] = [
                (sid, torch.tensor(f)) for sid, f in protos
            ]

        self._invalidate_cache()
