"""
Lightning module for MAE pretraining.

Handles training loop, logging, and visualization for both
single-view and multi-view MAE models.
"""

import math
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT

from mae.mae_model import MaskedAutoencoder, MAEConfig
from mae.multiview_mae import MultiViewMAE
from util import mlflow_utils


class MAETask(L.LightningModule):
    """
    Lightning module for training Masked Autoencoders.

    Supports both single-view (MaskedAutoencoder) and multi-view (MultiViewMAE)
    models. Handles optimizer configuration, learning rate scheduling,
    metric logging, and reconstruction visualization.

    The model configuration is automatically saved to checkpoints, enabling
    clean loading via MaskedAutoencoder.from_checkpoint() without needing
    to infer architecture from state_dict.

    Args:
        model: MaskedAutoencoder or MultiViewMAE instance
        learning_rate: Initial learning rate
        weight_decay: Weight decay for AdamW
        warmup_epochs: Number of warmup epochs for LR scheduler
        min_lr: Minimum learning rate after decay
        log_reconstruction_every: Log reconstruction images every N steps
        mask_ratio: Override mask ratio (if different from model default)
    """

    def __init__(
        self,
        model: Union[MaskedAutoencoder, MultiViewMAE],
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
        min_lr: float = 1e-6,
        log_reconstruction_every: int = 500,
        mask_ratio: Optional[float] = None,
    ):
        super().__init__()

        # Extract and store model config for checkpoint saving
        if isinstance(model, MultiViewMAE):
            mae_config = model.mae.get_config().to_dict()
            is_multiview = True
            share_masking = model.share_masking
        else:
            mae_config = model.get_config().to_dict()
            is_multiview = False
            share_masking = False

        self.save_hyperparameters(ignore=["model"])
        # Manually add config to hparams (save_hyperparameters can't capture it)
        self.hparams["mae_config"] = mae_config
        self.hparams["is_multiview"] = is_multiview
        self.hparams["share_masking"] = share_masking

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.log_reconstruction_every = log_reconstruction_every
        self.mask_ratio = mask_ratio

        # Track if model is multi-view
        self.is_multiview = isinstance(model, MultiViewMAE)

        # Validation outputs for epoch-end processing
        self.validation_step_outputs: List[Dict[str, Tensor]] = []

    def forward(self, x: Union[Tensor, List[Tensor]]) -> Dict[str, Any]:
        """Forward pass through model."""
        return self.model(x, mask_ratio=self.mask_ratio)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """Training step."""
        if self.is_multiview:
            views, labels, _ = batch
            outputs = self.model(views, mask_ratio=self.mask_ratio)
            loss = outputs["loss"]

            # Log per-view losses
            for i, view_loss in enumerate(outputs["losses"]):
                self.log(f"train/loss_view_{i}", view_loss, prog_bar=False)
        else:
            images, labels, _ = batch
            outputs = self.model(images, mask_ratio=self.mask_ratio)
            loss = outputs["loss"]

        # Log main loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train/mask_ratio",
            self.mask_ratio or self.model.mae.mask_ratio
            if self.is_multiview
            else self.model.mask_ratio,
        )

        # Log reconstruction images periodically
        if batch_idx % self.log_reconstruction_every == 0:
            self._log_reconstructions(batch, "train")

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """Validation step."""
        images, labels, _ = batch

        # For multi-view, validate on single view
        if self.is_multiview:
            outputs = self.model.mae(images, mask_ratio=self.mask_ratio)
        else:
            outputs = self.model(images, mask_ratio=self.mask_ratio)

        loss = outputs["loss"]

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # Store for epoch-end processing
        self.validation_step_outputs.append({"loss": loss})

        return {"loss": loss}

    def on_validation_epoch_end(self):
        """Log epoch-level metrics and visualizations."""
        if not self.validation_step_outputs:
            return

        # Compute average loss
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("val/loss_epoch", avg_loss)

        # Clear outputs
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """Test step."""
        images, labels, _ = batch

        if self.is_multiview:
            outputs = self.model.mae(images, mask_ratio=self.mask_ratio)
        else:
            outputs = self.model(images, mask_ratio=self.mask_ratio)

        loss = outputs["loss"]
        self.log("test/loss", loss, prog_bar=True)

        return {"loss": loss}

    def _log_reconstructions(self, batch, prefix: str):
        """Log reconstruction visualizations to logger."""
        if self.logger is None:
            return

        # Get images
        if self.is_multiview:
            views, _, _ = batch
            images = views[0] if isinstance(views, list) else views[:, 0]
        else:
            images, _, _ = batch

        # Take first few samples
        images = images[:8]

        # Get base model
        base_model = self.model.mae if self.is_multiview else self.model

        # Generate visualization
        with torch.no_grad():
            viz = base_model.visualize(images, mask_ratio=self.mask_ratio)

        # Create grid
        try:
            import torchvision.utils as vutils

            # Stack original, masked, reconstruction
            grid_images = torch.cat(
                [
                    viz["original"],
                    viz["masked"],
                    viz["reconstruction"],
                    viz["combined"],
                ],
                dim=0,
            )

            grid = vutils.make_grid(grid_images, nrow=8, normalize=True, pad_value=1)

            # Log to MLFlow if available
            if hasattr(self.logger, "experiment"):
                import mlflow
                import numpy as np
                from PIL import Image

                # Convert to numpy
                grid_np = grid.cpu().permute(1, 2, 0).numpy()
                grid_np = (grid_np * 255).astype(np.uint8)

                # Convert to PIL.Image
                img = Image.fromarray(grid_np)

                # Log artifact
                step = self.global_step
                epoch = self.current_epoch
                artifact_path = f"e{epoch:03d}/reconstructions/{prefix}_step_{step:05d}.png"
                mlflow_utils.log_image_artifact(self.logger, img, artifact_path)
                # mlflow.log_image(img, artifact_path)

        except Exception as e:
            # Silently ignore visualization errors
            pass

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters with and without weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "pos_embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate)

        # Cosine annealing with warmup
        if self.trainer and self.trainer.max_epochs:
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(total_steps * self.warmup_epochs / self.trainer.max_epochs)

            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup
                    return step / max(1, warmup_steps)
                else:
                    # Cosine decay
                    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                    return self.min_lr / self.learning_rate + (
                        1 - self.min_lr / self.learning_rate
                    ) * 0.5 * (1 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return optimizer

    def get_encoder(self) -> nn.Module:
        """Get the encoder for feature extraction."""
        if self.is_multiview:
            return self.model.encoder
        return self.model.encoder

    def encode(self, images: Tensor) -> Tensor:
        """Extract features from images."""
        return self.model.encode(images)

    @property
    def feature_dim(self) -> int:
        """Return the feature dimension."""
        return self.model.feature_dim


def load_mae_from_checkpoint(
    checkpoint_path: str,
    map_location: str = "cpu",
) -> MaskedAutoencoder:
    """
    Load a trained MAE model from checkpoint.

    Prefers config-based loading if the checkpoint contains 'mae_config',
    otherwise falls back to legacy state_dict inference.

    Args:
        checkpoint_path: Path to Lightning checkpoint
        map_location: Device to load to

    Returns:
        MaskedAutoencoder model (or base model from MultiViewMAE)
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})

    # Try config-based loading first (preferred)
    if "mae_config" in hparams:
        return MaskedAutoencoder.from_checkpoint(checkpoint_path, map_location)

    # Legacy fallback: infer architecture from state_dict
    return _load_mae_from_state_dict_legacy(checkpoint)


def _load_mae_from_state_dict_legacy(checkpoint: Dict) -> MaskedAutoencoder:
    """
    Legacy loader that infers architecture from state_dict.

    This is kept for backwards compatibility with older checkpoints
    that don't have config stored. New checkpoints should use
    MaskedAutoencoder.from_checkpoint() instead.

    Args:
        checkpoint: Loaded checkpoint dictionary

    Returns:
        MaskedAutoencoder with loaded weights
    """
    state_dict = checkpoint["state_dict"]

    # Check if it's a multi-view model
    is_multiview = any(k.startswith("model.mae.") for k in state_dict.keys())

    if is_multiview:
        prefix = "model.mae."
    else:
        prefix = "model."

    mae_state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}

    # Infer model config from state dict
    embed_dim = mae_state_dict["encoder.patch_embed.proj.weight"].shape[0]
    patch_size = mae_state_dict["encoder.patch_embed.proj.weight"].shape[2]
    in_chans = mae_state_dict["encoder.patch_embed.proj.weight"].shape[1]

    encoder_depth = sum(
        1
        for k in mae_state_dict.keys()
        if k.startswith("encoder.blocks.") and k.endswith(".norm1.weight")
    )

    decoder_depth = sum(
        1
        for k in mae_state_dict.keys()
        if k.startswith("decoder.blocks.") and k.endswith(".norm1.weight")
    )

    decoder_embed_dim = mae_state_dict["decoder.decoder_embed.weight"].shape[0]

    # Infer img_size from positional embedding
    pos_embed = mae_state_dict["encoder.pos_embed"]
    use_cls_token = "encoder.cls_token" in mae_state_dict
    num_patches = pos_embed.shape[1] - (1 if use_cls_token else 0)
    grid_size = int(num_patches**0.5)
    img_size = grid_size * patch_size

    # Infer encoder heads (can't be exactly determined, use reasonable default)
    # Check if num_heads is stored in attention layer
    encoder_heads = 4  # Default

    model = MaskedAutoencoder(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        encoder_depth=encoder_depth,
        encoder_heads=encoder_heads,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        use_cls_token=use_cls_token,
    )

    model.load_state_dict(mae_state_dict)
    return model
