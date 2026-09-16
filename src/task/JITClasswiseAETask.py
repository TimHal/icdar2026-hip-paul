"""Lightning task for class-wise autoencoders with just-in-time feature extraction."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from feature_extractors.base import FeatureExtractor
from task.ClasswiseAETask import ClasswiseAETask


class JITClasswiseAETask(ClasswiseAETask):
    """ClasswiseAETask that extracts features just-in-time from images.

    Instead of loading pre-computed features, this task receives raw images
    and uses a configurable feature extractor (DINO, CLIP, or custom) to
    compute embeddings before passing them to the autoencoder manager.

    The feature extractor is frozen and never trained.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        num_classes: int,
        n_components: int = 10,
        hidden_dims: list[int] = None,
        dropout: float = 0.01,
        l2_reg: float = 1e-6,
        learning_rate: float = 0.1,
        umap_loss_weight: float = 0.05,
        use_umap_loss: bool = True,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        normalize_features: bool = True,
    ):
        """Initialize the JIT task.

        Args:
            feature_extractor: Feature extractor instance (DINO, CLIP, or custom).
                Configured via class_path/init_args in YAML.
            num_classes: Number of classes (K autoencoders)
            n_components: Bottleneck dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            l2_reg: L2 regularization weight
            learning_rate: Learning rate
            umap_loss_weight: Weight for UMAP loss
            use_umap_loss: Whether to use UMAP graph layout loss
            umap_n_neighbors: Number of neighbors for UMAP loss
            umap_min_dist: Minimum distance for UMAP loss
            normalize_features: Whether to MinMax-normalize features to [0,1]
        """
        super().__init__(
            num_classes=num_classes,
            feature_dim=feature_extractor.feature_dim,
            n_components=n_components,
            hidden_dims=hidden_dims,
            dropout=dropout,
            l2_reg=l2_reg,
            learning_rate=learning_rate,
            umap_loss_weight=umap_loss_weight,
            use_umap_loss=use_umap_loss,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
        )

        self.feature_extractor = feature_extractor
        self.normalize_features = normalize_features

        # Save child-specific hyperparameters
        # Parent already saved its params, we only add our additions
        self.save_hyperparameters(
            ignore=[
                # All parent params (already saved by ClasswiseAETask)
                "num_classes", "feature_dim", "n_components", "hidden_dims",
                "dropout", "l2_reg", "learning_rate", "umap_loss_weight",
                "use_umap_loss", "umap_n_neighbors", "umap_min_dist",
                # Complex object that won't serialize
                "feature_extractor",
            ]
        )

        # Manually add feature extractor metadata
        self.hparams["feature_extractor_type"] = type(self.feature_extractor).__name__
        self.hparams["feature_extractor_dim"] = self.feature_extractor.feature_dim
        self._norm_params: dict | None = None

    def _sync_extractor_device(self) -> None:
        """Move the feature extractor's model to match this module's device."""
        if hasattr(self.feature_extractor, "model"):
            self.feature_extractor.model.to(self.device)
            self.feature_extractor.device = self.device

    def _compute_norm_params(self, dataloader: DataLoader) -> dict:
        """Compute MinMax normalization parameters from the training set.

        Args:
            dataloader: Training dataloader providing (images, labels, ids) batches

        Returns:
            Normalization params dict compatible with FeatureExtractor.apply_normalization
        """
        all_features = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(self.device)
                features = self.feature_extractor.extract(images)
                all_features.append(features.cpu())

        all_features = torch.cat(all_features, dim=0)
        _, params = self.feature_extractor.normalize_features(all_features, method="minmax")
        return params

    @torch.no_grad()
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract and optionally normalize features from images.

        Args:
            images: Raw images [batch_size, C, H, W]

        Returns:
            Features [batch_size, feature_dim], optionally normalized to [0,1]
        """
        features = self.feature_extractor.extract(images)
        if self.normalize_features and self._norm_params is not None:
            # Ensure norm params are on the same device as features
            params = {
                k: v.to(features.device) if isinstance(v, torch.Tensor) else v
                for k, v in self._norm_params.items()
            }
            features = self.feature_extractor.apply_normalization(features, params)
        return features

    def on_fit_start(self) -> None:
        """Sync extractor device and compute normalization params before training."""
        self._sync_extractor_device()
        if self.normalize_features:
            self._norm_params = self._compute_norm_params(
                self.trainer.datamodule.train_dataloader()
            )

    def on_test_start(self) -> None:
        """Sync extractor device before testing."""
        self._sync_extractor_device()

    def training_step(self, batch, batch_idx):
        images, labels, ids = batch
        features = self._extract_features(images)
        return super().training_step((features, labels, ids), batch_idx)

    def validation_step(self, batch, batch_idx):
        images, labels, ids = batch
        features = self._extract_features(images)
        return super().validation_step((features, labels, ids), batch_idx)

    def test_step(self, batch, batch_idx):
        images, labels, ids = batch
        features = self._extract_features(images)
        return super().test_step((features, labels, ids), batch_idx)
