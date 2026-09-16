"""Feature extraction modules for DINO, CLIP, and MAE models."""

from .base import FeatureExtractor
from .dino import DINOFeatureExtractor
from .clip import CLIPFeatureExtractor
from .mae import MAEFeatureExtractor

__all__ = [
    "FeatureExtractor",
    "DINOFeatureExtractor",
    "CLIPFeatureExtractor",
    "MAEFeatureExtractor",
]
