"""
Multi-View Masked Autoencoder (MAE) for handwritten character images.

This module provides a modular, minimal MAE implementation designed for:
- Learning embeddings from handwritten character images
- Multi-view training with independent masking per view
- Easy extension with additional losses (contrastive, perceptual, etc.)

Components:
- encoder: ViT-style encoder with patch embedding
- decoder: Lightweight decoder for reconstruction
- masking: Random patch masking strategies
- mae_model: Main MaskedAutoencoder combining all components
- multiview_mae: Multi-view wrapper for processing multiple augmented views
- losses: Reconstruction and future loss functions
"""

from mae.encoder import MAEEncoder
from mae.decoder import MAEDecoder
from mae.masking import RandomMasking
from mae.mae_model import MaskedAutoencoder, MAEConfig
from mae.multiview_mae import MultiViewMAE

__all__ = [
    "MAEEncoder",
    "MAEDecoder",
    "RandomMasking",
    "MaskedAutoencoder",
    "MAEConfig",
    "MultiViewMAE",
]
