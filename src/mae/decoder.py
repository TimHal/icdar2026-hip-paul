"""
Lightweight decoder for Masked Autoencoders.

The decoder reconstructs masked patches from encoder outputs.
Designed to be minimal (1-2 layers) and easily swappable.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mae.encoder import TransformerBlock
from util.pos_embed import get_2d_sincos_pos_embed


class MAEDecoder(nn.Module):
    """
    Lightweight MAE decoder.

    Takes encoder output and mask tokens, reconstructs original patches.
    Uses separate positional embeddings and optional transformer layers.

    Args:
        num_patches: Number of patches in image
        patch_size: Patch size (for reconstruction dimension)
        in_chans: Number of image channels
        embed_dim: Encoder embedding dimension
        decoder_embed_dim: Decoder embedding dimension
        depth: Number of transformer blocks (0 = linear only)
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        use_cls_token: Whether encoder uses CLS token
    """

    def __init__(
        self,
        num_patches: int,
        patch_size: int = 8,
        in_chans: int = 1,
        embed_dim: int = 128,
        decoder_embed_dim: int = 64,
        depth: int = 1,
        num_heads: int = 2,
        mlp_ratio: float = 4.0,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.use_cls_token = use_cls_token
        self.patch_pixels = patch_size * patch_size * in_chans

        # Project from encoder dim to decoder dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Decoder positional embedding
        grid_size = int(num_patches**0.5)
        pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(decoder_embed_dim, grid_size, cls_token=use_cls_token)
        ).float()
        self.register_buffer("decoder_pos_embed", pos_embed.unsqueeze(0))

        # Transformer blocks (optional, can be 0 for linear decoder)
        if depth > 0:
            self.blocks = nn.ModuleList([
                TransformerBlock(
                    dim=decoder_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(decoder_embed_dim)
        else:
            self.blocks = nn.ModuleList()
            self.norm = nn.Identity()

        # Prediction head: project to pixel values
        self.pred = nn.Linear(decoder_embed_dim, self.patch_pixels)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.decoder_embed.weight)
        nn.init.xavier_uniform_(self.pred.weight)
        nn.init.zeros_(self.pred.bias)

    def forward(
        self,
        x: Tensor,
        ids_restore: Tensor,
        cls_token: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Decode masked patches.

        Args:
            x: Encoder output [batch, num_visible, embed_dim]
            ids_restore: Indices to restore original order [batch, num_patches]
            cls_token: Optional CLS token from encoder [batch, embed_dim]

        Returns:
            pred: Reconstructed patches [batch, num_patches, patch_pixels]
        """
        batch_size = x.shape[0]

        # Project to decoder dimension
        x = self.decoder_embed(x)  # [batch, num_visible, decoder_embed_dim]

        # Append mask tokens for masked positions
        num_visible = x.shape[1]
        num_masked = self.num_patches - num_visible

        mask_tokens = self.mask_token.expand(batch_size, num_masked, -1)
        x = torch.cat([x, mask_tokens], dim=1)  # [batch, num_patches, decoder_embed_dim]

        # Unshuffle to original patch order
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim)
        )

        # Prepend CLS token if used
        if self.use_cls_token and cls_token is not None:
            cls_token = self.decoder_embed(cls_token).unsqueeze(1)
            x = torch.cat([cls_token, x], dim=1)

        # Add positional embedding
        x = x + self.decoder_pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Remove CLS token if present
        if self.use_cls_token and cls_token is not None:
            x = x[:, 1:, :]

        # Predict pixel values
        pred = self.pred(x)  # [batch, num_patches, patch_pixels]

        return pred


class LinearDecoder(nn.Module):
    """
    Minimal linear-only decoder.

    Even simpler than MAEDecoder with depth=0. Just projects and predicts.
    Useful as a baseline or when decoder capacity should be minimal.

    Args:
        num_patches: Number of patches
        patch_size: Patch size
        in_chans: Number of channels
        embed_dim: Encoder embedding dimension
    """

    def __init__(
        self,
        num_patches: int,
        patch_size: int = 8,
        in_chans: int = 1,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.patch_pixels = patch_size * patch_size * in_chans

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Direct prediction
        self.pred = nn.Linear(embed_dim, self.patch_pixels)

        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.pred.weight)
        nn.init.zeros_(self.pred.bias)

    def forward(
        self,
        x: Tensor,
        ids_restore: Tensor,
        cls_token: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Decode with linear projection only.

        Args:
            x: Encoder output [batch, num_visible, embed_dim]
            ids_restore: Indices to restore order [batch, num_patches]
            cls_token: Unused (for interface compatibility)

        Returns:
            pred: Reconstructed patches [batch, num_patches, patch_pixels]
        """
        batch_size = x.shape[0]
        embed_dim = x.shape[2]

        # Append mask tokens
        num_visible = x.shape[1]
        num_masked = self.num_patches - num_visible

        mask_tokens = self.mask_token.expand(batch_size, num_masked, -1)
        x = torch.cat([x, mask_tokens], dim=1)

        # Unshuffle
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, embed_dim)
        )

        # Predict
        pred = self.pred(x)

        return pred
