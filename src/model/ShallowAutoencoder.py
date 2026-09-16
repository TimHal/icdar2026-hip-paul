"""Shallow autoencoder matching the paper architecture (Marks et al. 2024)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowAutoencoder(nn.Module):
    """Shallow autoencoder for class-wise feature reconstruction.

    Architecture from paper (Table 1):
        Encoder: feature_dim -> Dense(256, ReLU, dropout=0.01) -> n_components
        Decoder: n_components -> Dense(256, ReLU, dropout=0.01) -> feature_dim (Sigmoid)

    The autoencoder is trained to reconstruct normalized features (in [0, 1] range).
    """

    def __init__(
        self,
        feature_dim: int,
        n_components: int = 10,
        hidden_dims: list[int] = None,
        dropout: float = 0.01,
        l2_reg: float = 1e-6,
    ):
        """Initialize the shallow autoencoder.

        Args:
            feature_dim: Dimensionality of input features (e.g., 384 for DINO ViT-S)
            n_components: Bottleneck dimension (default: 10 from paper)
            hidden_dims: Hidden layer dimensions (default: [256] from paper)
            dropout: Dropout rate (default: 0.01 from paper)
            l2_reg: L2 regularization weight (default: 1e-6 from paper)
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256]

        self.feature_dim = feature_dim
        self.n_components = n_components
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.l2_reg = l2_reg

        # Build encoder
        encoder_layers = []
        in_dim = feature_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, n_components))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (symmetric to encoder)
        decoder_layers = []
        in_dim = n_components
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        decoder_layers.extend([
            nn.Linear(in_dim, feature_dim),
            nn.Sigmoid(),  # Output in [0, 1] range for normalized features
        ])
        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode features to latent space.

        Args:
            x: Input features [batch_size, feature_dim]

        Returns:
            Latent representation [batch_size, n_components]
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to feature space.

        Args:
            z: Latent representation [batch_size, n_components]

        Returns:
            Reconstructed features [batch_size, feature_dim]
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder.

        Args:
            x: Input features [batch_size, feature_dim]

        Returns:
            Tuple of (reconstruction, latent):
                - reconstruction: [batch_size, feature_dim]
                - latent: [batch_size, n_components]
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction error (MSE).

        Args:
            x: Input features [batch_size, feature_dim]

        Returns:
            Reconstruction errors [batch_size]
        """
        x_hat, _ = self.forward(x)
        # Per-sample MSE
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mean reconstruction loss (MSE) over batch.

        Args:
            x: Input features [batch_size, feature_dim]

        Returns:
            Scalar MSE loss
        """
        x_hat, _ = self.forward(x)
        return F.mse_loss(x_hat, x)

    def get_l2_regularization(self) -> torch.Tensor:
        """Compute L2 regularization term for all parameters.

        Returns:
            L2 regularization loss (weighted by self.l2_reg)
        """
        l2_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            l2_loss = l2_loss + torch.sum(param ** 2)
        return self.l2_reg * l2_loss
