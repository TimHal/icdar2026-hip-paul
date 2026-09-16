"""
Loss functions for MAE training.

Currently implemented:
- MaskedMSELoss: MSE reconstruction loss on masked patches only

Future extensions:
- ContrastiveLoss: SimCLR-style contrastive learning
- CrossViewLoss: Consistency between views
- PerceptualLoss: VGG feature matching
- EdgeLoss: Edge preservation
"""

from mae.losses.reconstruction import MaskedMSELoss

__all__ = ["MaskedMSELoss"]
