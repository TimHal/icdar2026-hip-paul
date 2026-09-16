"""CLIP feature extractor using open-clip-torch."""

from __future__ import annotations

from typing import Optional

import torch

from .base import FeatureExtractor


# Model configurations: (model_name, pretrained) -> feature_dim
CLIP_MODELS = {
    ("ViT-B-32", "openai"): 512,
    ("ViT-B-32", "laion2b_s34b_b79k"): 512,
    ("ViT-B-16", "openai"): 512,
    ("ViT-B-16", "laion2b_s34b_b88k"): 512,
    ("ViT-L-14", "openai"): 768,
    ("ViT-L-14", "laion2b_s32b_b82k"): 768,
    ("ViT-H-14", "laion2b_s32b_b79k"): 1024,
}


class CLIPFeatureExtractor(FeatureExtractor):
    """Extract features using CLIP models from open-clip-torch.

    Supported models:
        - ViT-B-32 (512-dim, fastest)
        - ViT-B-16 (512-dim)
        - ViT-L-14 (768-dim, recommended for quality)
        - ViT-H-14 (1024-dim)
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[torch.device] = None,
    ):
        """Initialize CLIP feature extractor.

        Args:
            model_name: CLIP model architecture (e.g., 'ViT-B-32')
            pretrained: Pretrained weights source (e.g., 'openai', 'laion2b_s34b_b79k')
            device: Device to load model on (default: auto-detect)
        """
        try:
            import open_clip
        except ImportError:
            raise ImportError("Please install open-clip-torch: pip install open-clip-torch")

        key = (model_name, pretrained)
        if key not in CLIP_MODELS:
            available = ", ".join(f"{m}:{p}" for m, p in CLIP_MODELS.keys())
            raise ValueError(f"Unknown model: {model_name}:{pretrained}. Available: {available}")

        self._model_name = f"{model_name}_{pretrained}"
        self._feature_dim = CLIP_MODELS[key]

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load model from open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device)
        self.model.eval()

        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad = False

        # Store model info
        self._arch = model_name
        self._pretrained = pretrained

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features from CLIP vision encoder.

        Args:
            images: Input images [batch_size, channels, height, width]
                    Expected to be preprocessed with CLIP transforms

        Returns:
            Features [batch_size, feature_dim]
        """
        images = images.to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(images)
            # Normalize features (CLIP typically does this internally but we ensure it)
            features = features / features.norm(dim=-1, keepdim=True)

        return features

    def get_transform(self):
        """Get the preprocessing transform for this CLIP model.

        Returns:
            torchvision transform for preprocessing
        """
        return self.preprocess

    def get_transform_for_grayscale(self, image_size: int = 224):
        """Get preprocessing transform for grayscale images (MNIST variants).

        Converts grayscale to RGB by repeating channels, then applies CLIP preprocessing.

        Args:
            image_size: Target image size (default: 224)

        Returns:
            torchvision transform for preprocessing
        """
        from torchvision import transforms

        # Get the normalization values from CLIP preprocess
        # CLIP uses specific normalization values
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
