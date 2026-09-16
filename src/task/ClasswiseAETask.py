"""Lightning training task for class-wise autoencoders with RER computation."""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F

from model.ClasswiseAutoencoderManager import ClasswiseAutoencoderManager
from util.classification_metrics import (
    compute_accuracy,
    compute_f1_scores,
    log_confusion_matrix_artifacts,
)
from util.losses import compute_umap_loss
from util.rer_metrics import (
    compute_chi_0,
    compute_chi_rand,
    compute_class_chi,
    compute_dataset_chi,
    compute_rer,
    compute_rer_statistics,
    estimate_noise_rate,
)


class ClasswiseAETask(L.LightningModule):
    """Lightning task for training class-wise autoencoders.

    Trains K autoencoders (one per class) and computes Reconstruction
    Error Ratios (RER) for dataset difficulty estimation.

    Supports optional UMAP-style graph layout loss for manifold preservation.
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        n_components: int = 10,
        hidden_dims: list[int] = None,
        dropout: float = 0.01,
        l2_reg: float = 1e-6,
        learning_rate: float = 0.1,
        umap_loss_weight: float = 0.05,
        use_umap_loss: bool = True,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        autoencoder_class: str = "ShallowAutoencoder",
        autoencoder_kwargs: dict = None,
    ):
        """Initialize the ClasswiseAETask.

        Args:
            num_classes: Number of classes (K autoencoders)
            feature_dim: Dimensionality of input features
            n_components: Bottleneck dimension (default: 10)
            hidden_dims: Hidden layer dimensions for ShallowAutoencoder (default: [256])
            dropout: Dropout rate (default: 0.01)
            l2_reg: L2 regularization weight (default: 1e-6)
            learning_rate: Learning rate for optimizer (default: 0.1)
            umap_loss_weight: Weight for UMAP loss (default: 0.05, paper uses 20:1 MSE:UMAP)
            use_umap_loss: Whether to use UMAP graph layout loss (default: True)
            umap_n_neighbors: Number of neighbors for UMAP loss (default: 15)
            umap_min_dist: Minimum distance for UMAP loss (default: 0.1)
            autoencoder_class: Name of autoencoder class (default: "ShallowAutoencoder")
            autoencoder_kwargs: Additional kwargs for autoencoder constructor
        """
        super().__init__()
        self.save_hyperparameters()

        if hidden_dims is None:
            hidden_dims = [256]

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.umap_loss_weight = umap_loss_weight
        self.use_umap_loss = use_umap_loss
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist

        # Create the class-wise autoencoder manager
        self.ae_manager = ClasswiseAutoencoderManager(
            num_classes=num_classes,
            feature_dim=feature_dim,
            n_components=n_components,
            hidden_dims=hidden_dims,
            dropout=dropout,
            l2_reg=l2_reg,
            autoencoder_class=autoencoder_class,
            autoencoder_kwargs=autoencoder_kwargs,
        )

        # Storage for validation outputs (for epoch-end RER computation)
        self._val_outputs = []

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass computing RER values.

        Args:
            x: Features [batch_size, feature_dim]
            labels: Class labels [batch_size]

        Returns:
            RER values [batch_size]
        """
        all_errors = self.ae_manager.compute_all_reconstruction_errors(x)
        return compute_rer(all_errors, labels)

    def training_step(self, batch, batch_idx):
        """Training step: compute reconstruction loss for each class autoencoder.

        Each autoencoder only trains on samples from its corresponding class.
        """
        features, labels, _ = batch

        total_loss = torch.tensor(0.0, device=self.device)
        total_recon_loss = torch.tensor(0.0, device=self.device)
        total_umap_loss = torch.tensor(0.0, device=self.device)
        total_l2_loss = torch.tensor(0.0, device=self.device)
        classes_with_samples = 0

        for class_idx in range(self.num_classes):
            mask = labels == class_idx
            num_samples = mask.sum().item()

            if num_samples == 0:
                continue

            classes_with_samples += 1
            class_features = features[mask]
            ae = self.ae_manager.autoencoders[class_idx]

            # Reconstruction loss
            x_hat, z = ae(class_features)
            recon_loss = F.mse_loss(x_hat, class_features)
            total_recon_loss = total_recon_loss + recon_loss

            # UMAP loss (if enabled)
            if self.use_umap_loss and num_samples > 1:
                umap_loss = compute_umap_loss(class_features, z, self.umap_min_dist)
                total_umap_loss = total_umap_loss + umap_loss * self.umap_loss_weight

            # L2 regularization
            l2_loss = ae.get_l2_regularization()
            total_l2_loss = total_l2_loss + l2_loss

            # Log per-class loss
            self.log(f"train/loss_class_{class_idx}", recon_loss, on_step=False, on_epoch=True)

        # Combine losses
        total_loss = total_recon_loss + total_umap_loss + total_l2_loss

        # Log aggregate metrics
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss", total_recon_loss, on_step=False, on_epoch=True)
        if self.use_umap_loss:
            self.log("train/umap_loss", total_umap_loss, on_step=False, on_epoch=True)
        self.log("train/l2_loss", total_l2_loss, on_step=False, on_epoch=True)
        self.log(
            "train/classes_with_samples", float(classes_with_samples), on_step=False, on_epoch=True
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step: compute reconstruction errors and store for RER computation."""
        features, labels, _ = batch

        # Compute reconstruction errors from all autoencoders
        all_errors = self.ae_manager.compute_all_reconstruction_errors(features)

        # Store for epoch-end aggregation
        self._val_outputs.append(
            {
                "all_errors": all_errors.detach(),
                "labels": labels.detach(),
            }
        )

        # Log validation reconstruction loss
        total_loss, loss_dict = self.ae_manager.compute_total_loss(
            features, labels, include_l2=False
        )
        self.log("val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"all_errors": all_errors, "labels": labels}

    def _compute_epoch_metrics(self, prefix: str = "val"):
        """Compute RER and classification metrics at epoch end.

        Args:
            prefix: Metric prefix ("val" or "test")
        """
        if not self._val_outputs:
            return

        # Aggregate all validation outputs
        all_errors = torch.cat([x["all_errors"] for x in self._val_outputs], dim=0)
        all_labels = torch.cat([x["labels"] for x in self._val_outputs], dim=0)

        # Compute RER values
        rer_values = compute_rer(all_errors, all_labels)

        # Compute RER statistics
        chi = compute_dataset_chi(rer_values)
        class_chi = compute_class_chi(rer_values, all_labels)

        # Compute chi_0 and chi_rand for noise estimation
        chi_0 = compute_chi_0(all_errors, all_labels)
        chi_rand = compute_chi_rand(all_errors, all_labels)
        eta_est = estimate_noise_rate(chi_0, chi_rand)

        # Log RER metrics
        self.log(f"{prefix}/chi", chi, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/rer_mean", rer_values.mean().detach(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/rer_std", rer_values.std().detach(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/rer_min", rer_values.min().detach(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/rer_max", rer_values.max().detach(), on_step=False, on_epoch=True)
        self.log(f"{prefix}/chi_0", chi_0, on_step=False, on_epoch=True)
        self.log(f"{prefix}/chi_rand", chi_rand, on_step=False, on_epoch=True)
        self.log(f"{prefix}/eta_est", eta_est, on_step=False, on_epoch=True)
        self.log(
            f"{prefix}/frac_above_1",
            (rer_values > 1.0).float().mean().detach(),
            on_step=False,
            on_epoch=True,
        )

        # Log per-class chi
        for c, chi_c in class_chi.items():
            self.log(f"{prefix}/chi_class_{c}", chi_c, on_step=False, on_epoch=True)

        # Classification metrics: predict class with minimum reconstruction error
        predictions = all_errors.argmin(dim=1)

        # Compute accuracy
        accuracy = compute_accuracy(all_labels, predictions)
        self.log(f"{prefix}/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        # Compute F1 scores
        f1_scores = compute_f1_scores(all_labels, predictions)
        self.log(f"{prefix}/f1_micro", f1_scores["f1_micro"], on_step=False, on_epoch=True)
        self.log(f"{prefix}/f1_macro", f1_scores["f1_macro"], on_step=False, on_epoch=True)

        # Log confusion matrix artifact
        if self.logger is not None and hasattr(self.logger, "experiment"):
            artifact_prefix = f"{prefix}_epoch_{self.current_epoch:03d}"
            log_confusion_matrix_artifacts(
                logger=self.logger,
                run_id=self.logger.run_id,
                labels=all_labels,
                predictions=predictions,
                num_classes=self.num_classes,
                prefix=artifact_prefix,
                metadata={
                    "accuracy": accuracy,
                    "f1_micro": f1_scores["f1_micro"],
                    "f1_macro": f1_scores["f1_macro"],
                    "chi": chi,
                },
            )

        # Clear stored outputs
        self._val_outputs.clear()

    def on_validation_epoch_end(self):
        """Compute RER and classification statistics at the end of validation epoch."""
        self._compute_epoch_metrics(prefix="val")

    def test_step(self, batch, batch_idx):
        """Test step: same as validation step."""
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        """Compute final RER and classification statistics at end of test epoch."""
        self._compute_epoch_metrics(prefix="test")

    def configure_optimizers(self):
        """Configure optimizer with Adam."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer

    def get_rer_for_dataset(
        self,
        dataloader,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute RER for an entire dataset.

        Args:
            dataloader: DataLoader providing (features, labels, ids) batches

        Returns:
            Tuple of (rer_values, labels, statistics)
        """
        self.eval()
        all_errors_list = []
        all_labels_list = []

        with torch.no_grad():
            for features, labels, _ in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                all_errors = self.ae_manager.compute_all_reconstruction_errors(features)
                all_errors_list.append(all_errors)
                all_labels_list.append(labels)

        all_errors = torch.cat(all_errors_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0)

        rer_values = compute_rer(all_errors, all_labels)
        stats = compute_rer_statistics(rer_values, all_labels)

        # Add noise estimation
        stats["chi_0"] = compute_chi_0(all_errors, all_labels)
        stats["chi_rand"] = compute_chi_rand(all_errors, all_labels)
        stats["eta_est"] = estimate_noise_rate(stats["chi_0"], stats["chi_rand"])

        return rer_values, all_labels, stats
