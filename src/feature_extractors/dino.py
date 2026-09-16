"""DINO/DINOv2 feature extractor using torch.hub."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import FeatureExtractor


# Model configurations: model_name -> (hub_source, feature_dim)
DINO_MODELS = {
    # DINOv2 models (recommended)
    "dinov2_vits14": ("facebookresearch/dinov2", 384),
    "dinov2_vitb14": ("facebookresearch/dinov2", 768),
    "dinov2_vitl14": ("facebookresearch/dinov2", 1024),
    "dinov2_vitg14": ("facebookresearch/dinov2", 1536),
    # Original DINO models
    "dino_vits16": ("facebookresearch/dino:main", 384),
    "dino_vits8": ("facebookresearch/dino:main", 384),
    "dino_vitb16": ("facebookresearch/dino:main", 768),
    "dino_vitb8": ("facebookresearch/dino:main", 768),
}


class DINOFeatureExtractor(FeatureExtractor):
    """Extract features using DINO or DINOv2 models from torch.hub.

    Supported models:
        - dinov2_vits14 (384-dim, recommended for efficiency)
        - dinov2_vitb14 (768-dim, good balance)
        - dinov2_vitl14 (1024-dim)
        - dinov2_vitg14 (1536-dim)
        - dino_vits16, dino_vits8, dino_vitb16, dino_vitb8 (original DINO)
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        device: Optional[torch.device] = None,
    ):
        """Initialize DINO feature extractor.

        Args:
            model_name: Name of the DINO model to use
            device: Device to load model on (default: auto-detect)
        """
        if model_name not in DINO_MODELS:
            available = ", ".join(DINO_MODELS.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")

        self._model_name = model_name
        self._hub_source, self._feature_dim = DINO_MODELS[model_name]

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load model from torch.hub
        self.model = torch.hub.load(self._hub_source, model_name)
        self.model = self.model.to(device)
        self.model.eval()

        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLS token features from images.

        Args:
            images: Input images [batch_size, channels, height, width]
                    Expected to be normalized with ImageNet stats

        Returns:
            Features [batch_size, feature_dim]
        """
        images = images.to(self.device)

        with torch.no_grad():
            # DINOv2 and DINO models return CLS token by default
            features = self.model(images)

        return features

    @staticmethod
    def get_transform(image_size: int = 224):
        """Get the preprocessing transform for DINO models.

        Args:
            image_size: Target image size (default: 224)

        Returns:
            torchvision transform for preprocessing
        """
        from torchvision import transforms

        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @staticmethod
    def get_transform_for_grayscale(image_size: int = 224):
        """Get preprocessing transform for grayscale images (MNIST variants).

        Converts grayscale to RGB by repeating channels.

        Args:
            image_size: Target image size (default: 224)

        Returns:
            torchvision transform for preprocessing
        """
        from torchvision import transforms

        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
