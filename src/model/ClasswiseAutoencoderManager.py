"""Manager for class-wise autoencoders with RER computation."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ShallowAutoencoder import ShallowAutoencoder
from .ShallowConvAutoencoder import ShallowConvAutoencoder


# Registry of available autoencoder classes
_AUTOENCODER_CLASSES = {
    "ShallowAutoencoder": ShallowAutoencoder,
    "ShallowConvAutoencoder": ShallowConvAutoencoder,
}


class ClasswiseAutoencoderManager(nn.Module):
    """Manages K autoencoders, one per class, for RER computation.

    This module creates and manages separate autoencoder instances
    for each class, enabling:
    - Class-specific training: Each AE trains only on its class samples
    - RER computation: Reconstruction errors from all AEs are compared

    The Reconstruction Error Ratio (RER) is defined as (Equation 3 in paper):
        chi(x^c) = Delta^c(x^c) / min_{c' != c} Delta^{c'}(x^c)

    Where Delta^c(x) is the reconstruction error for sample x using class c's autoencoder.
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        n_components: int = 10,
        hidden_dims: list[int] = None,
        dropout: float = 0.01,
        l2_reg: float = 1e-6,
        autoencoder_class: str = "ShallowAutoencoder",
        autoencoder_kwargs: Optional[dict] = None,
    ):
        """Initialize the class-wise autoencoder manager.

        Args:
            num_classes: Number of classes (K autoencoders will be created)
            feature_dim: Dimensionality of input features
            n_components: Bottleneck dimension (default: 10)
            hidden_dims: Hidden layer dimensions for ShallowAutoencoder (default: [256])
            dropout: Dropout rate (default: 0.01)
            l2_reg: L2 regularization weight (default: 1e-6)
            autoencoder_class: Name of autoencoder class to use (default: "ShallowAutoencoder")
            autoencoder_kwargs: Additional kwargs to pass to autoencoder constructor
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256]

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.n_components = n_components
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.autoencoder_class_name = autoencoder_class

        # Resolve autoencoder class
        if autoencoder_class not in _AUTOENCODER_CLASSES:
            raise ValueError(
                f"Unknown autoencoder_class '{autoencoder_class}'. "
                f"Available: {list(_AUTOENCODER_CLASSES.keys())}"
            )
        AEClass = _AUTOENCODER_CLASSES[autoencoder_class]

        # Build kwargs for autoencoder constructor
        ae_kwargs = dict(
            feature_dim=feature_dim,
            n_components=n_components,
            dropout=dropout,
            l2_reg=l2_reg,
        )

        # Only pass hidden_dims to ShallowAutoencoder
        if autoencoder_class == "ShallowAutoencoder":
            ae_kwargs["hidden_dims"] = hidden_dims

        # Merge with any extra kwargs
        if autoencoder_kwargs:
            ae_kwargs.update(autoencoder_kwargs)

        # Create K autoencoders
        self.autoencoders = nn.ModuleList([
            AEClass(**ae_kwargs)
            for _ in range(num_classes)
        ])

    def forward_class(
        self,
        x: torch.Tensor,
        class_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through a specific class autoencoder.

        Args:
            x: Input features [batch_size, feature_dim]
            class_idx: Index of the class autoencoder to use

        Returns:
            Tuple of (reconstruction, latent)
        """
        return self.autoencoders[class_idx](x)

    def compute_all_reconstruction_errors(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction errors for all class autoencoders.

        Args:
            x: Input features [batch_size, feature_dim]

        Returns:
            Reconstruction errors [batch_size, num_classes]
        """
        errors = []
        for ae in self.autoencoders:
            errors.append(ae.reconstruction_error(x))
        return torch.stack(errors, dim=1)

    def compute_class_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        class_idx: int,
    ) -> tuple[torch.Tensor, int]:
        """Compute reconstruction loss for a specific class.

        Args:
            features: All features in batch [batch_size, feature_dim]
            labels: All labels in batch [batch_size]
            class_idx: Index of class to compute loss for

        Returns:
            Tuple of (loss, num_samples) where num_samples is the count
            of samples belonging to class_idx
        """
        mask = labels == class_idx
        num_samples = mask.sum().item()

        if num_samples == 0:
            return torch.tensor(0.0, device=features.device), 0

        class_features = features[mask]
        ae = self.autoencoders[class_idx]
        loss = ae.reconstruction_loss(class_features)

        return loss, num_samples

    def compute_total_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        include_l2: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """Compute total loss across all class autoencoders.

        Args:
            features: Features [batch_size, feature_dim]
            labels: Labels [batch_size]
            include_l2: Whether to include L2 regularization

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains per-class losses
        """
        total_loss = torch.tensor(0.0, device=features.device)
        loss_dict = {}

        for class_idx in range(self.num_classes):
            class_loss, num_samples = self.compute_class_loss(features, labels, class_idx)
            if num_samples > 0:
                total_loss = total_loss + class_loss
                loss_dict[f"loss_class_{class_idx}"] = class_loss.item()

                if include_l2:
                    l2_loss = self.autoencoders[class_idx].get_l2_regularization()
                    total_loss = total_loss + l2_loss

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict

    def get_class_autoencoder(self, class_idx: int) -> nn.Module:
        """Get the autoencoder for a specific class.

        Args:
            class_idx: Index of the class

        Returns:
            Autoencoder instance for the specified class
        """
        return self.autoencoders[class_idx]
