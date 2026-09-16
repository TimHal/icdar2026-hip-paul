"""
Masking strategies for Masked Autoencoders.

The masking module provides different strategies for selecting which patches
to mask during MAE training. All strategies follow a common interface.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class MaskingStrategy(ABC, nn.Module):
    """Abstract base class for masking strategies."""

    def __init__(self, mask_ratio: float = 0.75):
        super().__init__()
        self.mask_ratio = mask_ratio

    @abstractmethod
    def forward(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply masking to patch sequence.

        Args:
            x: Input tensor [batch, num_patches, embed_dim]

        Returns:
            x_masked: Visible patches [batch, num_visible, embed_dim]
            mask: Binary mask [batch, num_patches] where 1 = masked
            ids_restore: Indices to restore original order [batch, num_patches]
        """
        pass


class RandomMasking(MaskingStrategy):
    """
    Random patch masking strategy.

    Randomly selects patches to mask with a configurable ratio.
    Uses noise-based shuffling for efficient per-sample randomization.
    """

    def __init__(self, mask_ratio: float = 0.75):
        super().__init__(mask_ratio)

    def forward(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Randomly mask patches.

        Args:
            x: [batch, num_patches, embed_dim]

        Returns:
            x_masked: [batch, num_visible, embed_dim]
            mask: [batch, num_patches] binary (1 = masked, 0 = visible)
            ids_restore: [batch, num_patches] indices to unshuffle
        """
        batch_size, num_patches, embed_dim = x.shape
        num_visible = int(num_patches * (1 - self.mask_ratio))

        # Generate random noise for shuffling
        noise = torch.rand(batch_size, num_patches, device=x.device)

        # Sort noise to get shuffled indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first num_visible patches (smallest noise values)
        ids_keep = ids_shuffle[:, :num_visible]

        # Gather visible patches
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, embed_dim)
        )

        # Create binary mask: 0 = visible, 1 = masked
        mask = torch.ones(batch_size, num_patches, device=x.device)
        mask[:, :num_visible] = 0
        # Unshuffle to get mask in original patch order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


class NoMasking(MaskingStrategy):
    """
    No masking - returns all patches.

    Useful for inference or when masking should be disabled.
    """

    def __init__(self):
        super().__init__(mask_ratio=0.0)

    def forward(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Return all patches without masking.

        Args:
            x: [batch, num_patches, embed_dim]

        Returns:
            x (unchanged), mask (all zeros), ids_restore (identity)
        """
        batch_size, num_patches, _ = x.shape
        mask = torch.zeros(batch_size, num_patches, device=x.device)
        ids_restore = torch.arange(num_patches, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return x, mask, ids_restore
