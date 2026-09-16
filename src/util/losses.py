"""Shared loss functions for class-wise autoencoder tasks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_umap_loss(
    features: torch.Tensor,
    latents: torch.Tensor,
    min_dist: float = 0.1,
) -> torch.Tensor:
    """Compute UMAP-style graph layout loss for manifold preservation.

    Encourages local neighborhood preservation in the latent space,
    similar to the parametric UMAP loss used in Marks et al. (2024).

    Args:
        features: Original features [batch_size, feature_dim]
        latents: Latent representations [batch_size, n_components]
        min_dist: Minimum distance for repulsive loss

    Returns:
        UMAP loss scalar
    """
    batch_size = features.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=features.device)

    # Pairwise distances
    feat_dists = torch.cdist(features, features, p=2)
    latent_dists = torch.cdist(latents, latents, p=2)

    # Soft neighborhood weights
    sigma = feat_dists.mean() + 1e-8
    feat_weights = torch.exp(-(feat_dists**2) / (2 * sigma**2))

    # Mask diagonal
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)
    feat_weights = feat_weights * mask.float()
    feat_weights = feat_weights / (feat_weights.sum(dim=1, keepdim=True) + 1e-8)

    # Attractive loss: neighbors should stay close
    attractive_loss = (feat_weights * latent_dists**2).sum() / batch_size

    # Repulsive loss: non-neighbors should stay apart
    repulsive_weights = (1 - feat_weights) * mask.float()
    repulsive_weights = repulsive_weights / (repulsive_weights.sum(dim=1, keepdim=True) + 1e-8)
    repulsive_loss = (
        repulsive_weights * F.relu(min_dist - latent_dists) ** 2
    ).sum() / batch_size

    return attractive_loss + repulsive_loss
