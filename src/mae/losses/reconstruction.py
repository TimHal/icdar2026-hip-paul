"""
Reconstruction losses for MAE training.

The primary loss is MSE computed only on masked patches.
Optionally supports per-patch normalization for stability.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class MAELoss(ABC, nn.Module):
    """Abstract base class for MAE losses."""

    @abstractmethod
    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """
        Compute loss.

        Args:
            pred: Predicted patches [batch, num_patches, patch_pixels]
            target: Target patches [batch, num_patches, patch_pixels]
            mask: Binary mask [batch, num_patches] where 1 = masked

        Returns:
            loss: Scalar loss value
        """
        pass


class MaskedMSELoss(MAELoss):
    """
    MSE loss computed only on masked patches.

    This is the standard MAE reconstruction loss. The loss is computed
    per-patch, averaged over masked patches only.

    Args:
        norm_pix_loss: If True, normalize target patches to zero mean,
            unit variance before computing loss. This can improve training
            stability by making all patches equally weighted regardless
            of their content.
    """

    def __init__(self, norm_pix_loss: bool = False):
        super().__init__()
        self.norm_pix_loss = norm_pix_loss

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """
        Compute masked MSE loss.

        Args:
            pred: Predicted patches [batch, num_patches, patch_pixels]
            target: Target patches [batch, num_patches, patch_pixels]
            mask: Binary mask [batch, num_patches] where 1 = masked

        Returns:
            loss: Mean MSE over masked patches
        """
        if self.norm_pix_loss:
            # Normalize each patch independently
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        # Per-pixel squared error
        loss = (pred - target) ** 2  # [batch, num_patches, patch_pixels]

        # Mean over pixels in each patch
        loss = loss.mean(dim=-1)  # [batch, num_patches]

        # Mean over masked patches only
        # mask is 1 for masked, 0 for visible
        loss = (loss * mask).sum() / mask.sum()

        return loss


class MaskedL1Loss(MAELoss):
    """
    L1 loss computed only on masked patches.

    Alternative to MSE that may be more robust to outliers.
    """

    def __init__(self, norm_pix_loss: bool = False):
        super().__init__()
        self.norm_pix_loss = norm_pix_loss

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """
        Compute masked L1 loss.

        Args:
            pred: Predicted patches [batch, num_patches, patch_pixels]
            target: Target patches [batch, num_patches, patch_pixels]
            mask: Binary mask [batch, num_patches] where 1 = masked

        Returns:
            loss: Mean L1 over masked patches
        """
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = torch.abs(pred - target)
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return loss


class CombinedLoss(nn.Module):
    """
    Combine multiple losses with weights.

    Useful for adding auxiliary losses like contrastive or perceptual.

    Args:
        losses: Dict mapping loss names to (loss_fn, weight) tuples
    """

    def __init__(self, losses: dict):
        super().__init__()
        self.loss_fns = nn.ModuleDict()
        self.weights = {}

        for name, (loss_fn, weight) in losses.items():
            self.loss_fns[name] = loss_fn
            self.weights[name] = weight

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> dict:
        """
        Compute all losses.

        Returns:
            dict with 'loss' (total) and individual loss values
        """
        results = {}
        total = 0.0

        for name, loss_fn in self.loss_fns.items():
            loss_val = loss_fn(pred, target, mask)
            weight = self.weights[name]
            results[name] = loss_val
            total = total + weight * loss_val

        results["loss"] = total
        return results
