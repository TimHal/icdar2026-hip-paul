"""
ViT-style encoder for Masked Autoencoders.

The encoder converts images to patch embeddings and processes them
through transformer blocks. Designed for small-scale handwritten images.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from util.pos_embed import get_2d_sincos_pos_embed


class PatchEmbed(nn.Module):
    """
    Convert image to patch embeddings using Conv2d.

    Args:
        img_size: Input image size (assumes square)
        patch_size: Patch size (assumes square)
        in_chans: Number of input channels
        embed_dim: Embedding dimension
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 1,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, channels, height, width]

        Returns:
            patches: [batch, num_patches, embed_dim]
        """
        x = self.proj(x)  # [batch, embed_dim, grid_h, grid_w]
        x = x.flatten(2).transpose(1, 2)  # [batch, num_patches, embed_dim]
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: If True, add bias to QKV projection
        attn_drop: Attention dropout rate
        proj_drop: Output projection dropout rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, seq_len, dim]

        Returns:
            out: [batch, seq_len, dim]
        """
        batch_size, seq_len, dim = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention weights
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """
    MLP block with GELU activation.

    Args:
        in_features: Input dimension
        hidden_features: Hidden dimension (default: 4x input)
        out_features: Output dimension (default: same as input)
        drop: Dropout rate
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with pre-norm architecture.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        drop: Dropout rate
        attn_drop: Attention dropout rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAEEncoder(nn.Module):
    """
    ViT-style encoder for MAE.

    Converts images to patch embeddings, adds positional embeddings,
    and processes through transformer blocks. Supports optional CLS token.

    Args:
        img_size: Input image size
        patch_size: Patch size
        in_chans: Number of input channels (1 for grayscale)
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        use_cls_token: Whether to use CLS token
        learn_pos_embed: Whether positional embeddings are learnable
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 1,
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_cls_token: bool = True,
        learn_pos_embed: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(embed_dim, self.grid_size, cls_token=use_cls_token)
        ).float()
        if learn_pos_embed:
            self.pos_embed = nn.Parameter(pos_embed.unsqueeze(0))
        else:
            self.register_buffer("pos_embed", pos_embed.unsqueeze(0))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)

        # Initialize patch embed
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))

        # Initialize transformer blocks
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None, ids_keep: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through encoder.

        Args:
            x: Input images [batch, channels, height, width]
            mask: Optional mask (for visualization only)
            ids_keep: Optional indices of patches to keep (for masked forward)

        Returns:
            tokens: Encoded tokens [batch, num_tokens, embed_dim]
            cls_token: CLS token if use_cls_token, else None
        """
        # Patch embedding
        x = self.patch_embed(x)  # [batch, num_patches, embed_dim]

        # Add positional embedding (without CLS if masking)
        if ids_keep is not None:
            # For masked encoding: add pos embed then gather
            if self.use_cls_token:
                x = x + self.pos_embed[:, 1:, :]  # Skip CLS pos embed
            else:
                x = x + self.pos_embed
            # Gather visible patches
            x = torch.gather(
                x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            )
        else:
            # For full encoding
            if self.use_cls_token:
                x = x + self.pos_embed[:, 1:, :]
            else:
                x = x + self.pos_embed

        # Prepend CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            if ids_keep is None:
                cls_tokens = cls_tokens + self.pos_embed[:, :1, :]
            x = torch.cat([cls_tokens, x], dim=1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract CLS token
        if self.use_cls_token:
            cls_token = x[:, 0]
            tokens = x[:, 1:]
        else:
            cls_token = None
            tokens = x

        return tokens, cls_token

    def forward_features(self, x: Tensor) -> Tensor:
        """
        Extract pooled features for downstream tasks.

        Args:
            x: Input images [batch, channels, height, width]

        Returns:
            features: Pooled features [batch, embed_dim]
        """
        tokens, cls_token = self.forward(x)

        if self.use_cls_token and cls_token is not None:
            return cls_token
        else:
            # Mean pool over patch tokens
            return tokens.mean(dim=1)
