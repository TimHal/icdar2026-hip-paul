"""
Multi-View MAE wrapper.

Processes multiple augmented views of the same image independently,
allowing for future extensions like cross-view losses or teacher-student setups.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from mae.mae_model import MaskedAutoencoder, MAEConfig


class MultiViewMAE(nn.Module):
    """
    Multi-View Masked Autoencoder.

    Wraps a base MAE model to process multiple augmented views of the same image.
    Each view is masked independently and processed through the shared encoder.
    The reconstruction loss is computed per view and averaged.

    This design supports future extensions:
    - Cross-view consistency loss
    - Contrastive learning between views
    - DINO-style teacher-student (add momentum encoder)
    - Different masking per view

    Args:
        base_mae: Base MaskedAutoencoder model
        share_masking: If True, use same mask for all views (default: False)
    """

    def __init__(
        self,
        base_mae: MaskedAutoencoder,
        share_masking: bool = False,
    ):
        super().__init__()
        self.mae = base_mae
        self.share_masking = share_masking

    def forward(
        self, views: Union[List[Tensor], Tensor], mask_ratio: Optional[float] = None
    ) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        Process multiple views through the MAE.

        Args:
            views: Either:
                - List of tensors, each [batch, channels, height, width]
                - Single tensor [batch, num_views, channels, height, width]
            mask_ratio: Optional override for mask ratio

        Returns:
            dict with:
                - loss: Mean loss across all views
                - losses: List of per-view losses
                - predictions: List of per-view predicted patches
                - masks: List of per-view masks
                - latents: List of per-view encoded features
                - cls_tokens: List of per-view CLS tokens (if use_cls_token)
        """
        # Handle both list and stacked tensor input
        if isinstance(views, Tensor) and views.dim() == 5:
            # [batch, num_views, C, H, W] -> list of [batch, C, H, W]
            views = [views[:, i] for i in range(views.shape[1])]

        num_views = len(views)

        # Process each view
        losses = []
        predictions = []
        masks = []
        latents = []
        cls_tokens = []
        ids_restores = []

        # For shared masking, generate mask from first view and reuse
        shared_mask = None
        shared_ids_restore = None

        for i, view in enumerate(views):
            if self.share_masking and shared_mask is not None:
                # Use shared mask
                outputs = self._forward_with_mask(
                    view, shared_mask, shared_ids_restore
                )
            else:
                # Independent masking
                outputs = self.mae(view, mask_ratio)

                if self.share_masking and i == 0:
                    # Save mask from first view
                    shared_mask = outputs["mask"]
                    shared_ids_restore = outputs["ids_restore"]

            losses.append(outputs["loss"])
            predictions.append(outputs["pred"])
            masks.append(outputs["mask"])
            latents.append(outputs["latent"])
            if outputs["cls_token"] is not None:
                cls_tokens.append(outputs["cls_token"])
            ids_restores.append(outputs["ids_restore"])

        # Average loss across views
        total_loss = torch.stack(losses).mean()

        result = {
            "loss": total_loss,
            "losses": losses,
            "predictions": predictions,
            "masks": masks,
            "latents": latents,
            "ids_restores": ids_restores,
        }

        if cls_tokens:
            result["cls_tokens"] = cls_tokens

        return result

    def _forward_with_mask(
        self, x: Tensor, mask: Tensor, ids_restore: Tensor
    ) -> Dict[str, Tensor]:
        """
        Forward pass with predetermined mask.

        Used for shared masking across views.
        """
        # Patch embed
        patches = self.mae.encoder.patch_embed(x)

        # Add position embedding
        if self.mae.use_cls_token:
            patches = patches + self.mae.encoder.pos_embed[:, 1:, :]
        else:
            patches = patches + self.mae.encoder.pos_embed

        # Apply predetermined mask (gather visible patches)
        visible_mask = 1 - mask  # 0 = masked, 1 = visible
        ids_keep = torch.argsort(
            visible_mask + torch.rand_like(visible_mask.float()) * 0.1, dim=1, descending=True
        )
        num_visible = int((1 - self.mae.mask_ratio) * self.mae.num_patches)
        ids_keep = ids_keep[:, :num_visible]

        x_masked = torch.gather(
            patches, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, self.mae.embed_dim)
        )

        # Prepend CLS token
        if self.mae.use_cls_token:
            cls_tokens = self.mae.encoder.cls_token.expand(x.shape[0], -1, -1)
            x_masked = torch.cat([cls_tokens, x_masked], dim=1)

        # Encoder transformer blocks
        for block in self.mae.encoder.blocks:
            x_masked = block(x_masked)
        x_masked = self.mae.encoder.norm(x_masked)

        # Extract CLS token
        if self.mae.use_cls_token:
            cls_token = x_masked[:, 0]
            latent = x_masked[:, 1:]
        else:
            cls_token = None
            latent = x_masked

        # Decode
        pred = self.mae.forward_decoder(latent, ids_restore, cls_token)

        # Loss
        loss = self.mae.forward_loss(x, pred, mask)

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
        Extract features from a single image (no masking).

        Args:
            x: Input image [batch, channels, height, width]

        Returns:
            features: Pooled features [batch, embed_dim]
        """
        return self.mae.encode(x)

    def encode_views(self, views: Union[List[Tensor], Tensor]) -> List[Tensor]:
        """
        Extract features from multiple views.

        Args:
            views: List of [batch, C, H, W] or [batch, num_views, C, H, W]

        Returns:
            List of feature tensors [batch, embed_dim] per view
        """
        if isinstance(views, Tensor) and views.dim() == 5:
            views = [views[:, i] for i in range(views.shape[1])]

        return [self.mae.encode(view) for view in views]

    def encode_mean(self, views: Union[List[Tensor], Tensor]) -> Tensor:
        """
        Extract averaged features across views.

        Args:
            views: List of [batch, C, H, W] or [batch, num_views, C, H, W]

        Returns:
            features: Mean features [batch, embed_dim]
        """
        features = self.encode_views(views)
        return torch.stack(features).mean(dim=0)

    @property
    def feature_dim(self) -> int:
        """Return the feature dimension for downstream tasks."""
        return self.mae.feature_dim

    @property
    def encoder(self):
        """Access the encoder for checkpoint saving."""
        return self.mae.encoder

    @property
    def decoder(self):
        """Access the decoder."""
        return self.mae.decoder

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of this model.

        Returns:
            Dict with 'mae_config' and 'share_masking' keys.
        """
        return {
            "mae_config": self.mae.get_config().to_dict(),
            "share_masking": self.share_masking,
        }

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> "MultiViewMAE":
        """Create a MultiViewMAE from a config dict.

        Args:
            config: Dict with 'mae_config' and optional 'share_masking'.

        Returns:
            New MultiViewMAE instance.
        """
        mae_config = MAEConfig.from_dict(config["mae_config"])
        base_mae = MaskedAutoencoder.from_config(mae_config)
        share_masking = config.get("share_masking", False)
        return cls(base_mae=base_mae, share_masking=share_masking)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: str = "cpu",
    ) -> "MultiViewMAE":
        """Load a MultiViewMAE from a Lightning checkpoint.

        The checkpoint must contain 'mae_config' in hyper_parameters.

        Args:
            checkpoint_path: Path to Lightning checkpoint.
            map_location: Device to load tensors to.

        Returns:
            MultiViewMAE with loaded weights.

        Raises:
            ValueError: If checkpoint doesn't contain model config.
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})

        config_dict = hparams.get("mae_config")
        if config_dict is None:
            raise ValueError(
                f"Checkpoint at {checkpoint_path} does not contain 'mae_config'. "
                "Use MAETask with a MultiViewMAE model to save config."
            )

        # Check if multiview-specific config exists
        is_multiview = hparams.get("is_multiview", False)
        share_masking = hparams.get("share_masking", False)

        # Create model from config
        mae_config = MAEConfig.from_dict(config_dict)
        base_mae = MaskedAutoencoder.from_config(mae_config)
        model = cls(base_mae=base_mae, share_masking=share_masking)

        # Extract and load state dict
        state_dict = checkpoint["state_dict"]

        # Determine prefix
        if any(k.startswith("model.mae.") for k in state_dict.keys()):
            prefix = "model."
        elif any(k.startswith("model.") for k in state_dict.keys()):
            prefix = "model."
        else:
            prefix = ""

        # Extract model weights
        if prefix:
            model_state_dict = {
                k[len(prefix):]: v for k, v in state_dict.items()
                if k.startswith(prefix)
            }
        else:
            model_state_dict = state_dict

        model.load_state_dict(model_state_dict)
        return model
