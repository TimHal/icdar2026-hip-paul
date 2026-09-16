"""Shallow convolutional autoencoder for raw image inputs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowConvAutoencoder(nn.Module):
    """Shallow convolutional autoencoder for class-wise image reconstruction.

    This autoencoder works with raw image pixels rather than pre-extracted features.
    It accepts flat vectors (for compatibility with ClasswiseAutoencoderManager)
    but internally reshapes them to images for convolutional processing.

    Architecture (super shallow - 2 conv layers):
        Encoder: [B, H*W*C] -> reshape -> Conv -> Conv -> flatten -> Linear -> [B, n_components]
        Decoder: [B, n_components] -> Linear -> reshape -> ConvT -> ConvT -> flatten -> [B, H*W*C]
    """

    def __init__(
        self,
        feature_dim: int,
        n_components: int = 10,
        image_shape: tuple[int, int, int] = None,
        conv_channels: list[int] = None,
        dropout: float = 0.01,
        l2_reg: float = 1e-6,
    ):
        """Initialize the shallow convolutional autoencoder.

        Args:
            feature_dim: Total flat feature dimension (C*H*W, e.g., 784 for 28x28 grayscale)
            n_components: Bottleneck dimension (default: 10)
            image_shape: (C, H, W) tuple for reshaping flat input. Required.
            conv_channels: List of conv layer channel sizes (default: [16, 32])
            dropout: Dropout rate (default: 0.01)
            l2_reg: L2 regularization weight (default: 1e-6)
        """
        super().__init__()

        if image_shape is None:
            raise ValueError("image_shape (C, H, W) is required for ShallowConvAutoencoder")

        if conv_channels is None:
            conv_channels = [16, 32]

        self.feature_dim = feature_dim
        self.n_components = n_components
        self.image_shape = tuple(image_shape)
        self.conv_channels = conv_channels
        self.dropout_rate = dropout
        self.l2_reg = l2_reg

        c, h, w = self.image_shape

        # Verify feature_dim matches image_shape
        expected_dim = c * h * w
        if feature_dim != expected_dim:
            raise ValueError(
                f"feature_dim ({feature_dim}) must equal C*H*W ({c}*{h}*{w}={expected_dim})"
            )

        # Build encoder convolutions
        # Each conv with stride=2 halves spatial dimensions
        encoder_convs = []
        in_channels = c
        current_h, current_w = h, w

        for out_channels in conv_channels:
            encoder_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )
            encoder_convs.append(nn.ReLU())
            encoder_convs.append(nn.Dropout2d(dropout))
            in_channels = out_channels
            current_h = (current_h + 1) // 2  # ceil(h/2)
            current_w = (current_w + 1) // 2

        self.encoder_conv = nn.Sequential(*encoder_convs)

        # Store spatial dimensions after convolutions
        self.encoded_h = current_h
        self.encoded_w = current_w
        self.encoded_channels = conv_channels[-1]
        self.flat_encoded_dim = self.encoded_channels * self.encoded_h * self.encoded_w

        # Linear layer to bottleneck
        self.encoder_fc = nn.Linear(self.flat_encoded_dim, n_components)

        # Build decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(n_components, self.flat_encoded_dim),
            nn.ReLU(),
        )

        # Build decoder convolutions (transposed)
        decoder_convs = []
        reversed_channels = list(reversed(conv_channels))

        for i, out_channels in enumerate(reversed_channels[1:] + [c]):
            in_channels = reversed_channels[i]
            decoder_convs.append(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
                )
            )
            if i < len(reversed_channels) - 1:
                # ReLU for intermediate layers
                decoder_convs.append(nn.ReLU())
                decoder_convs.append(nn.Dropout2d(dropout))
            else:
                # Sigmoid for final layer (output in [0, 1])
                decoder_convs.append(nn.Sigmoid())

        self.decoder_conv = nn.Sequential(*decoder_convs)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode flat input to latent space.

        Args:
            x: Flat input [batch_size, feature_dim]

        Returns:
            Latent representation [batch_size, n_components]
        """
        batch_size = x.shape[0]
        c, h, w = self.image_shape

        # Reshape to image format
        x_img = x.view(batch_size, c, h, w)

        # Apply conv encoder
        encoded = self.encoder_conv(x_img)

        # Flatten and apply FC
        encoded_flat = encoded.view(batch_size, -1)
        z = self.encoder_fc(encoded_flat)

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to flat output.

        Args:
            z: Latent representation [batch_size, n_components]

        Returns:
            Reconstructed flat output [batch_size, feature_dim]
        """
        batch_size = z.shape[0]

        # FC layer
        decoded = self.decoder_fc(z)

        # Reshape to spatial format
        decoded = decoded.view(batch_size, self.encoded_channels, self.encoded_h, self.encoded_w)

        # Apply transposed convolutions
        decoded = self.decoder_conv(decoded)

        # Crop/pad to exact original size if needed (due to stride arithmetic)
        c, h, w = self.image_shape
        decoded = decoded[:, :, :h, :w]

        # Flatten to match input format
        x_hat = decoded.view(batch_size, -1)

        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder.

        Args:
            x: Flat input [batch_size, feature_dim]

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
            x: Flat input [batch_size, feature_dim]

        Returns:
            Reconstruction errors [batch_size]
        """
        x_hat, _ = self.forward(x)
        # Per-sample MSE
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mean reconstruction loss (MSE) over batch.

        Args:
            x: Flat input [batch_size, feature_dim]

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
