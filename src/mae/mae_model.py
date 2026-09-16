"""
Main Masked Autoencoder model.

Combines encoder, decoder, masking, and loss into a unified model.
Provides both training (with masking) and inference (feature extraction) APIs.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from mae.encoder import MAEEncoder
from mae.decoder import MAEDecoder
from mae.masking import RandomMasking, NoMasking
from mae.losses.reconstruction import MaskedMSELoss


@dataclass
class MAEConfig:
    """Configuration for MaskedAutoencoder.

    Stores all hyperparameters needed to reconstruct the model architecture.
    """
    img_size: int = 64
    patch_size: int = 8
    in_chans: int = 1
    embed_dim: int = 128
    encoder_depth: int = 2
    encoder_heads: int = 4
    decoder_embed_dim: int = 64
    decoder_depth: int = 1
    decoder_heads: int = 2
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.75
    norm_pix_loss: bool = False
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    use_cls_token: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MAEConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder for self-supervised learning.

    Implements the MAE approach: mask random patches, encode visible patches,
    decode to reconstruct masked patches. The encoder learns useful features
    by solving this pretext task.

    Args:
        img_size: Input image size (assumes square)
        patch_size: Patch size
        in_chans: Number of input channels (1 for grayscale)
        embed_dim: Encoder embedding dimension
        encoder_depth: Number of encoder transformer blocks
        encoder_heads: Number of encoder attention heads
        decoder_embed_dim: Decoder embedding dimension
        decoder_depth: Number of decoder transformer blocks
        decoder_heads: Number of decoder attention heads
        mlp_ratio: MLP hidden dimension ratio
        mask_ratio: Fraction of patches to mask (0-1)
        norm_pix_loss: Normalize patches before computing loss
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        use_cls_token: Whether to use CLS token
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 1,
        embed_dim: int = 128,
        encoder_depth: int = 2,
        encoder_heads: int = 4,
        decoder_embed_dim: int = 64,
        decoder_depth: int = 1,
        decoder_heads: int = 2,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_cls_token: bool = True,
    ):
        super().__init__()

        # Save config
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.use_cls_token = use_cls_token

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_pixels = patch_size * patch_size * in_chans

        # Encoder
        self.encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            use_cls_token=use_cls_token,
        )

        # Decoder
        self.decoder = MAEDecoder(
            num_patches=self.num_patches,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            mlp_ratio=mlp_ratio,
            use_cls_token=use_cls_token,
        )

        # Masking
        self.masking = RandomMasking(mask_ratio=mask_ratio)
        self.no_masking = NoMasking()

        # Loss
        self.loss_fn = MaskedMSELoss(norm_pix_loss=norm_pix_loss)

    def patchify(self, imgs: Tensor) -> Tensor:
        """
        Convert images to patch sequences.

        Args:
            imgs: [batch, channels, height, width]

        Returns:
            patches: [batch, num_patches, patch_pixels]
        """
        p = self.patch_size
        c = self.in_chans
        h = w = self.img_size // p

        x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [batch, h, w, c, p, p]
        x = x.reshape(imgs.shape[0], h * w, c * p * p)

        return x

    def unpatchify(self, patches: Tensor) -> Tensor:
        """
        Convert patch sequences back to images.

        Args:
            patches: [batch, num_patches, patch_pixels]

        Returns:
            imgs: [batch, channels, height, width]
        """
        p = self.patch_size
        c = self.in_chans
        h = w = self.img_size // p

        x = patches.reshape(patches.shape[0], h, w, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)  # [batch, c, h, p, w, p]
        x = x.reshape(patches.shape[0], c, h * p, w * p)

        return x

    def forward_encoder(
        self, x: Tensor, mask_ratio: Optional[float] = None
    ) -> tuple:
        """
        Encode with masking.

        Args:
            x: Input images [batch, channels, height, width]
            mask_ratio: Optional override for mask ratio

        Returns:
            latent: Encoded visible patches [batch, num_visible, embed_dim]
            mask: Binary mask [batch, num_patches]
            ids_restore: Indices to restore order [batch, num_patches]
            cls_token: CLS token if use_cls_token [batch, embed_dim]
        """
        # Patch embed
        patches = self.encoder.patch_embed(x)  # [batch, num_patches, embed_dim]

        # Add position embedding (without CLS)
        if self.use_cls_token:
            patches = patches + self.encoder.pos_embed[:, 1:, :]
        else:
            patches = patches + self.encoder.pos_embed

        # Apply masking
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        if mask_ratio > 0:
            masking = RandomMasking(mask_ratio=mask_ratio)
            x_masked, mask, ids_restore = masking(patches)
        else:
            x_masked, mask, ids_restore = self.no_masking(patches)

        # Prepend CLS token
        if self.use_cls_token:
            cls_tokens = self.encoder.cls_token.expand(x.shape[0], -1, -1)
            x_masked = torch.cat([cls_tokens, x_masked], dim=1)

        # Transformer blocks
        for block in self.encoder.blocks:
            x_masked = block(x_masked)

        x_masked = self.encoder.norm(x_masked)

        # Extract CLS token
        if self.use_cls_token:
            cls_token = x_masked[:, 0]
            latent = x_masked[:, 1:]
        else:
            cls_token = None
            latent = x_masked

        return latent, mask, ids_restore, cls_token

    def forward_decoder(
        self, latent: Tensor, ids_restore: Tensor, cls_token: Optional[Tensor] = None
    ) -> Tensor:
        """
        Decode to reconstruct patches.

        Args:
            latent: Encoded patches [batch, num_visible, embed_dim]
            ids_restore: Indices to restore order [batch, num_patches]
            cls_token: Optional CLS token [batch, embed_dim]

        Returns:
            pred: Predicted patches [batch, num_patches, patch_pixels]
        """
        pred = self.decoder(latent, ids_restore, cls_token)
        return pred

    def forward_loss(self, imgs: Tensor, pred: Tensor, mask: Tensor) -> Tensor:
        """
        Compute reconstruction loss on masked patches.

        Args:
            imgs: Original images [batch, channels, height, width]
            pred: Predicted patches [batch, num_patches, patch_pixels]
            mask: Binary mask [batch, num_patches]

        Returns:
            loss: Scalar loss value
        """
        target = self.patchify(imgs)
        loss = self.loss_fn(pred, target, mask)
        return loss

    def forward(
        self, x: Tensor, mask_ratio: Optional[float] = None
    ) -> Dict[str, Tensor]:
        """
        Full forward pass with masking and reconstruction.

        Args:
            x: Input images [batch, channels, height, width]
            mask_ratio: Optional override for mask ratio

        Returns:
            dict with:
                - loss: Reconstruction loss
                - pred: Predicted patches [batch, num_patches, patch_pixels]
                - mask: Binary mask [batch, num_patches]
                - latent: Encoded features [batch, num_visible, embed_dim]
                - cls_token: CLS token if use_cls_token [batch, embed_dim]
        """
        # Encode with masking
        latent, mask, ids_restore, cls_token = self.forward_encoder(x, mask_ratio)

        # Decode
        pred = self.forward_decoder(latent, ids_restore, cls_token)

        # Compute loss
        loss = self.forward_loss(x, pred, mask)

        return {
            "loss": loss,
            "pred": pred,
            "mask": mask,
            "latent": latent,
            "cls_token": cls_token,
            "ids_restore": ids_restore,
        }

    def encode(self, x: Tensor) -> Tensor:
        """
        Extract features without masking.

        Use this for downstream tasks (retrieval, classification, etc.)

        Args:
            x: Input images [batch, channels, height, width]

        Returns:
            features: Pooled features [batch, embed_dim]
        """
        return self.encoder.forward_features(x)

    def encode_patches(self, x: Tensor) -> Tensor:
        """
        Extract all patch embeddings without masking.

        Args:
            x: Input images [batch, channels, height, width]

        Returns:
            patch_features: [batch, num_patches, embed_dim]
        """
        tokens, _ = self.encoder(x)
        return tokens

    def reconstruct(self, x: Tensor, mask_ratio: Optional[float] = None) -> Tensor:
        """
        Reconstruct images (for visualization).

        Args:
            x: Input images [batch, channels, height, width]
            mask_ratio: Optional override for mask ratio

        Returns:
            recon: Reconstructed images [batch, channels, height, width]
        """
        outputs = self.forward(x, mask_ratio)
        pred = outputs["pred"]
        return self.unpatchify(pred)

    def visualize(self, x: Tensor, mask_ratio: Optional[float] = None) -> Dict[str, Tensor]:
        """
        Generate visualization of masking and reconstruction.

        Args:
            x: Input images [batch, channels, height, width]
            mask_ratio: Optional override for mask ratio

        Returns:
            dict with:
                - original: Original images
                - masked: Images with masked patches zeroed
                - reconstruction: Reconstructed images
                - combined: Side-by-side visualization
        """
        outputs = self.forward(x, mask_ratio)
        pred = outputs["pred"]
        mask = outputs["mask"]

        # Reconstructed image
        recon = self.unpatchify(pred)

        # Masked image (zero out masked patches)
        target = self.patchify(x)
        masked_target = target * (1 - mask.unsqueeze(-1))
        masked_img = self.unpatchify(masked_target)

        # Visible patches from original, masked from reconstruction
        combined_patches = target * (1 - mask.unsqueeze(-1)) + pred * mask.unsqueeze(-1)
        combined = self.unpatchify(combined_patches)

        return {
            "original": x,
            "masked": masked_img,
            "reconstruction": recon,
            "combined": combined,
            "mask": mask,
        }

    @property
    def feature_dim(self) -> int:
        """Return the feature dimension for downstream tasks."""
        return self.embed_dim

    def get_config(self) -> MAEConfig:
        """Get the configuration of this model.

        Returns:
            MAEConfig with all hyperparameters needed to recreate the architecture.
        """
        return MAEConfig(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            encoder_depth=len(self.encoder.blocks),
            encoder_heads=self.encoder.blocks[0].attn.num_heads if self.encoder.blocks else 4,
            decoder_embed_dim=self.decoder.decoder_embed.out_features,
            decoder_depth=len(self.decoder.blocks),
            decoder_heads=self.decoder.blocks[0].attn.num_heads if self.decoder.blocks else 2,
            mlp_ratio=self.encoder.blocks[0].mlp.fc1.out_features / self.embed_dim if self.encoder.blocks else 4.0,
            mask_ratio=self.mask_ratio,
            norm_pix_loss=self.loss_fn.norm_pix_loss,
            use_cls_token=self.use_cls_token,
        )

    @classmethod
    def from_config(cls, config: MAEConfig) -> "MaskedAutoencoder":
        """Create a MaskedAutoencoder from a config.

        Args:
            config: MAEConfig with model hyperparameters.

        Returns:
            New MaskedAutoencoder instance.
        """
        return cls(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            encoder_depth=config.encoder_depth,
            encoder_heads=config.encoder_heads,
            decoder_embed_dim=config.decoder_embed_dim,
            decoder_depth=config.decoder_depth,
            decoder_heads=config.decoder_heads,
            mlp_ratio=config.mlp_ratio,
            mask_ratio=config.mask_ratio,
            norm_pix_loss=config.norm_pix_loss,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            use_cls_token=config.use_cls_token,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: str = "cpu",
    ) -> "MaskedAutoencoder":
        """Load a MaskedAutoencoder from a Lightning checkpoint.

        The checkpoint must contain 'mae_config' in hyper_parameters or
        'model_config' for proper loading. Architecture is inferred from
        config, not from state_dict.

        Args:
            checkpoint_path: Path to Lightning checkpoint (.ckpt file).
            map_location: Device to load tensors to.

        Returns:
            MaskedAutoencoder with loaded weights.

        Raises:
            ValueError: If checkpoint doesn't contain model config.
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        # Get config from hyperparameters
        hparams = checkpoint.get("hyper_parameters", {})
        config_dict = hparams.get("mae_config") or hparams.get("model_config")

        if config_dict is None:
            raise ValueError(
                f"Checkpoint at {checkpoint_path} does not contain 'mae_config' or 'model_config' "
                "in hyper_parameters. This checkpoint was saved without config storage. "
                "Use MAETask with save_model_config=True or the legacy loading method."
            )

        # Create model from config
        config = MAEConfig.from_dict(config_dict)
        model = cls.from_config(config)

        # Extract and load state dict
        state_dict = checkpoint["state_dict"]

        # Determine prefix based on checkpoint structure
        if any(k.startswith("model.mae.") for k in state_dict.keys()):
            prefix = "model.mae."
        elif any(k.startswith("model.") for k in state_dict.keys()):
            prefix = "model."
        else:
            prefix = ""

        # Extract model weights
        if prefix:
            mae_state_dict = {
                k[len(prefix):]: v for k, v in state_dict.items()
                if k.startswith(prefix)
            }
        else:
            mae_state_dict = state_dict

        model.load_state_dict(mae_state_dict)
        return model
