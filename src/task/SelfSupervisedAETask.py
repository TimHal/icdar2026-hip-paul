"""Self-supervised class-wise autoencoder training with prototype-guided assignment.

This task implements a fully unsupervised training procedure:
1. Initialize with prototype samples (1..N per class)
   - Prototypes are auto-selected by default (1 random sample per class)
   - Can be disabled or customized via parameters
2. Train each autoencoder on its assigned samples (prototypes only at first)
3. At end of each epoch: re-evaluate all samples, optionally prune, then assign
4. Assignments are based on reconstruction error rate (RER) - lower is better

Assignment modes (at epoch end, after training):
- fixed_k: Assign K best (lowest RER) unassigned samples per class
- threshold: Assign all unassigned samples with RER below threshold

Optional pruning (before new assignments):
- fixed_k: Remove K worst (highest RER) non-prototype samples per class
- threshold: Remove all samples with RER above threshold

True labels are only used for validation/test metrics, not for training.

Example:
    # Basic usage with auto-selected prototypes
    task = SelfSupervisedAETask(num_classes=10, feature_dim=384)
    trainer.fit(task, datamodule)

    # Custom prototype selection
    task = SelfSupervisedAETask(
        num_classes=10,
        feature_dim=384,
        prototypes_per_class=3,
        prototype_selection='centroid',
    )

    # Explicit prototypes
    provider = PrototypeProvider.from_labels(...)
    task = SelfSupervisedAETask(num_classes=10, feature_dim=384)
    task.set_prototype_provider(provider)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor

from data.PrototypeProvider import PrototypeProvider
from model.ClasswiseAutoencoderManager import ClasswiseAutoencoderManager
from model.PrototypeLedger import PrototypeLedger
from util.classification_metrics import (
    compute_accuracy,
    compute_f1_scores,
    log_confusion_matrix_artifacts,
)
from util.losses import compute_umap_loss
from util.rer_metrics import compute_rer


class SelfSupervisedAETask(L.LightningModule):
    """Self-supervised class-wise autoencoder training.

    Training is fully unsupervised - uses prototypes for initialization and
    reconstruction error for iterative assignment refinement.
    True labels are only used for validation/test metrics.

    The training loop:
    1. At epoch start: re-evaluate all samples, optionally prune, then assign
    2. During training: train AEs on their assigned samples (read-only assignments)
    3. Prototypes are always assigned to their designated classes

    Assignment modes (at epoch start):
    - fixed_k: Assign K best (lowest RER) unassigned samples per class
    - threshold: Assign all unassigned samples with RER below threshold

    Optional pruning (before new assignments):
    - fixed_k: Remove K worst (highest RER) samples per class
    - threshold: Remove all samples with RER above threshold

    Args:
        num_classes: Number of classes (K autoencoders)
        feature_dim: Dimensionality of input features
        n_components: Bottleneck dimension
        hidden_dims: Hidden layer dimensions
        dropout: Dropout rate
        l2_reg: L2 regularization weight

        rer_threshold: Maximum RER for sample assignment (legacy)
        top_n_per_class: Top samples per class for epoch refinement
        min_samples_for_training: Minimum samples to train an AE

        assignment_mode: "fixed_k" or "threshold" for epoch-based assignment
        assignments_per_epoch: K samples to assign per class (fixed_k mode)
        assignment_threshold: RER threshold for assignment (threshold mode)

        enable_pruning: Whether to prune poorly assigned samples
        pruning_mode: "fixed_k" or "threshold" for pruning strategy
        prune_per_epoch: K worst samples to prune per class (fixed_k mode)
        prune_threshold: RER threshold for pruning (threshold mode, default 1.5)

        auto_select_prototypes: If True and no prototypes provided, auto-select from data
        prototypes_per_class: Number of prototypes to auto-select per class
        prototype_selection: Selection strategy ('random', 'first', 'centroid')

        learning_rate: Learning rate for optimizer
        warmup_epochs: Epochs before assignment starts (only prototypes trained)

        use_prototype_anchoring: Whether to use prototype anchoring loss
        prototype_anchor_weight: Weight for prototype anchoring loss
        use_umap_loss: Whether to use UMAP manifold loss
        umap_loss_weight: Weight for UMAP loss
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        n_components: int = 10,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.01,
        l2_reg: float = 1e-6,
        # Assignment settings
        rer_threshold: float = 2.0,
        top_n_per_class: int = 500,
        min_samples_for_training: int = 5,
        # Assignment strategy (epoch-based)
        assignment_mode: str = "fixed_k",  # "fixed_k" or "threshold"
        assignments_per_epoch: int = 10,  # K samples per class (fixed_k mode)
        assignment_threshold: float = 0.8,  # RER threshold (threshold mode)
        # Pruning settings
        enable_pruning: bool = False,
        pruning_mode: str = "threshold",  # "fixed_k" or "threshold"
        prune_per_epoch: int = 5,  # Worst K to remove (fixed_k mode)
        prune_threshold: float = 1.5,  # Remove if RER > threshold
        # Prototype settings
        auto_select_prototypes: bool = True,
        prototypes_per_class: int = 1,
        prototype_selection: str = "random",
        prototype_source: str | None = None,  # Path to .pth file or folder
        prototype_img_mode: str = "L",  # For folder loading
        # Threshold annealing
        anneal_assignment_threshold: bool = False,
        anneal_threshold_epochs: int | None = None,  # None = trainer.max_epochs
        # Training settings
        learning_rate: float = 0.001,
        warmup_epochs: int = 0,
        # LR scheduler
        use_cosine_lr: bool = False,
        cosine_lr_min: float = 1e-6,
        cosine_lr_warmup_epochs: int = 0,
        # Loss settings
        use_prototype_anchoring: bool = True,
        prototype_anchor_weight: float = 0.1,
        use_umap_loss: bool = False,
        umap_loss_weight: float = 0.05,
        umap_min_dist: float = 0.1,
        # Sample reconstruction logging
        log_samples: bool = False,
        log_samples_frequency: int = 0,
        autoencoder_class: str = "ShallowAutoencoder",
        autoencoder_kwargs: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if hidden_dims is None:
            hidden_dims = [256]

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.n_components = n_components
        self.rer_threshold = rer_threshold
        self.top_n_per_class = top_n_per_class
        self.min_samples_for_training = min_samples_for_training
        # Assignment strategy
        self.assignment_mode = assignment_mode
        self.assignments_per_epoch = assignments_per_epoch
        self.assignment_threshold = assignment_threshold
        # Pruning settings
        self.enable_pruning = enable_pruning
        self.pruning_mode = pruning_mode
        self.prune_per_epoch = prune_per_epoch
        self.prune_threshold = prune_threshold
        # Prototype settings
        self.auto_select_prototypes = auto_select_prototypes
        self.prototypes_per_class = prototypes_per_class
        self.prototype_selection = prototype_selection
        self.prototype_source = prototype_source
        self.prototype_img_mode = prototype_img_mode
        self.anneal_assignment_threshold = anneal_assignment_threshold
        self.anneal_threshold_epochs = anneal_threshold_epochs
        self._initial_assignment_threshold = assignment_threshold
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.use_cosine_lr = use_cosine_lr
        self.cosine_lr_min = cosine_lr_min
        self.cosine_lr_warmup_epochs = cosine_lr_warmup_epochs
        self.use_prototype_anchoring = use_prototype_anchoring
        self.prototype_anchor_weight = prototype_anchor_weight
        self.use_umap_loss = use_umap_loss
        self.umap_loss_weight = umap_loss_weight
        self.umap_min_dist = umap_min_dist
        self.log_samples = log_samples
        self.log_samples_frequency = log_samples_frequency

        # Create class-wise autoencoder manager
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

        # Ledger for tracking weak assignments (initialized in setup)
        self.ledger: PrototypeLedger | None = None

        # Prototype provider (set externally or via setup)
        self.prototype_provider: PrototypeProvider | None = None

        # Cached prototype data
        self._prototype_features: Tensor | None = None
        self._prototype_classes: Tensor | None = None
        self._prototype_ids: Tensor | None = None

        # Fixed batch for sample reconstruction logging (captured once, reused)
        self._sample_log_batch: tuple[Tensor, Tensor] | None = None

        # Validation/test output storage
        self._val_outputs: list[dict] = []

        # Epoch assignment/pruning statistics
        self._epoch_new_assignments: int = 0
        self._epoch_pruned_samples: int = 0

    def set_prototype_provider(self, provider: PrototypeProvider) -> None:
        """Set the prototype provider.

        Args:
            provider: PrototypeProvider with registered prototypes
        """
        self.prototype_provider = provider
        self._update_prototype_cache()

    def _update_prototype_cache(self) -> None:
        """Update cached prototype tensors from provider."""
        if self.prototype_provider is None:
            return

        features, classes, ids = self.prototype_provider.get_prototype_features()
        if len(features) == 0:
            return

        self._prototype_features = features
        self._prototype_classes = classes
        self._prototype_ids = ids

    def setup(self, stage: str) -> None:
        """Initialize ledger and register prototypes."""
        if stage != "fit":
            return

        # Initialize ledger
        self.ledger = PrototypeLedger(
            num_classes=self.num_classes,
            rer_threshold=self.rer_threshold,
            device=str(self.device),
        )

        # Load external prototypes if a source was given, regardless of
        # auto_select_prototypes (which only governs in-distribution sampling).
        if self.prototype_source is not None:
            self._load_external_prototypes()
        elif self.auto_select_prototypes:
            self._auto_select_prototypes_if_needed()

        # Register prototypes in ledger
        self._initialize_from_prototypes()

        # Log prototype images to MLflow
        self._log_prototype_images()

    def _load_external_prototypes(self) -> None:
        """Load prototypes from external .pth file or folder."""
        source_path = Path(self.prototype_source)

        # Get image_shape from autoencoder_kwargs if available
        image_shape = None
        if self.hparams.get("autoencoder_kwargs"):
            image_shape = self.hparams["autoencoder_kwargs"].get("image_shape")
            if image_shape is not None:
                image_shape = tuple(image_shape)

        if source_path.is_file() and source_path.suffix == ".pth":
            self.prototype_provider = PrototypeProvider.from_pth_file(
                pth_path=source_path,
                image_shape=image_shape,
            )
        elif source_path.is_dir():
            if image_shape is None:
                raise ValueError(
                    "image_shape required for folder prototype loading. "
                    "Set autoencoder_kwargs.image_shape in config."
                )
            self.prototype_provider = PrototypeProvider.from_folder(
                folder_path=source_path,
                image_shape=image_shape,
                img_mode=self.prototype_img_mode,
            )
        else:
            raise ValueError(
                f"Invalid prototype_source: {self.prototype_source}. "
                "Must be a .pth file or a directory."
            )

        self._update_prototype_cache()

    def _auto_select_prototypes_if_needed(self) -> None:
        """Automatically select prototypes from training data if none provided."""
        # Check for external prototype source first
        if self.prototype_source is not None:
            self._load_external_prototypes()
            return

        # Skip if prototypes already provided
        if self.prototype_provider is not None:
            if self.prototype_provider.get_prototype_count() > 0:
                return

        # Need access to training data
        if not hasattr(self.trainer, "datamodule") or self.trainer.datamodule is None:
            return

        datamodule = self.trainer.datamodule

        # Try to access train dataloader
        try:
            # Setup datamodule if needed
            if hasattr(datamodule, "setup") and not hasattr(datamodule, "train_dataset"):
                datamodule.setup("fit")

            # Get train dataloader
            train_loader = datamodule.train_dataloader()

            # Collect samples from training data
            all_features = []
            all_labels = []
            all_ids = []

            # Collect from first few batches (enough to get prototypes)
            max_batches = min(10, len(train_loader))
            for i, batch in enumerate(train_loader):
                if i >= max_batches:
                    break

                # Batch format: (features/images, labels, ids)
                features, labels, ids = batch

                # Move to CPU for processing
                features = features.cpu()
                labels = labels.cpu()
                ids = ids.cpu()

                all_features.append(features)
                all_labels.append(labels)
                all_ids.append(ids)

            if not all_features:
                return

            # Concatenate
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_ids = torch.cat(all_ids, dim=0)

            # Create prototype provider from collected data
            self.prototype_provider = PrototypeProvider.from_labels(
                features=all_features,
                labels=all_labels,
                sample_ids=all_ids,
                prototypes_per_class=self.prototypes_per_class,
                selection=self.prototype_selection,
            )

            # Update cache
            self._update_prototype_cache()

        except Exception as e:
            # If auto-selection fails, just continue without prototypes
            # This allows training to proceed even if datamodule structure is different
            import warnings

            warnings.warn(
                f"Auto-selection of prototypes failed: {e}. "
                "Continuing without prototypes. "
                "Set auto_select_prototypes=False or provide prototypes explicitly.",
                UserWarning,
            )

    def _initialize_from_prototypes(self) -> None:
        """Initialize ledger with prototype assignments."""
        if self.prototype_provider is None:
            return

        self._update_prototype_cache()

        if self._prototype_features is None or len(self._prototype_features) == 0:
            return

        # Register each prototype in the ledger (without computing errors yet)
        # Errors will be computed on first forward pass when device is known
        proto_classes = self._prototype_classes
        proto_ids = self._prototype_ids

        for i in range(len(proto_ids)):
            sid = proto_ids[i].item()
            class_idx = proto_classes[i].item()

            self.ledger.register_prototype(
                sample_id=sid,
                class_idx=class_idx,
                reconstruction_error=0.0,  # Will be updated on first forward
            )

    def _get_effective_assignment_threshold(self) -> float:
        """Return the assignment threshold for the current epoch.

        When annealing is enabled, linearly interpolates from the initial
        assignment_threshold to 1.0 over ``anneal_threshold_epochs`` epochs.
        """
        if not self.anneal_assignment_threshold:
            return self.assignment_threshold

        total = self.anneal_threshold_epochs
        if total is None:
            total = self.trainer.max_epochs
        if total is None or total <= 0:
            return self.assignment_threshold

        t = min(self.current_epoch / total, 1.0)
        init = self._initial_assignment_threshold
        return init + (1.0 - init) * t

    def _perform_epoch_assignment_and_pruning(self) -> None:
        """Handle epoch-based assignment and pruning.

        Called at end of each epoch after training:
        1. Re-evaluate all training samples with current AE states
        2. Prune poorly assigned samples (if enabled)
        3. Assign new samples based on assignment mode

        Assignments are made at epoch END so AEs have trained on data first.
        """
        if self.ledger is None:
            return

        # Skip warmup epochs (train on prototypes only)
        if self.current_epoch < self.warmup_epochs:
            self._epoch_new_assignments = 0
            self._epoch_pruned_samples = 0
            return

        # Get datamodule and evaluate all samples
        if not hasattr(self.trainer, "datamodule") or self.trainer.datamodule is None:
            return

        candidates = self._evaluate_all_training_samples()
        if not candidates:
            return

        # Update RER values for currently assigned samples
        self.ledger.update_rer_values(candidates)

        # Pruning (optional, after re-evaluation, skip first assignment epoch)
        self._epoch_pruned_samples = 0
        if self.enable_pruning and self.current_epoch > self.warmup_epochs:
            if self.pruning_mode == "fixed_k":
                self._epoch_pruned_samples = self.ledger.prune_worst_k_per_class(
                    self.prune_per_epoch
                )
            else:  # threshold mode
                self._epoch_pruned_samples = self.ledger.prune_above_threshold(self.prune_threshold)

        # New assignments based on mode
        if self.assignment_mode == "fixed_k":
            self._epoch_new_assignments = self.ledger.assign_top_k_per_class(
                candidates, self.assignments_per_epoch, self.current_epoch
            )
        else:  # threshold mode
            effective_threshold = self._get_effective_assignment_threshold()
            self._epoch_new_assignments = self.ledger.assign_below_threshold(
                candidates, effective_threshold, self.current_epoch
            )

    def _evaluate_all_training_samples(self) -> dict[int, tuple[int, float, float]]:
        """Evaluate all training samples with current autoencoders.

        Returns:
            {sample_id: (best_class, rer, error)} for all training samples
        """
        datamodule = self.trainer.datamodule
        all_candidates: dict[int, tuple[int, float, float]] = {}

        try:
            train_loader = datamodule.train_dataloader()
        except Exception:
            return all_candidates

        # Temporarily set to eval mode for consistent evaluation
        self.ae_manager.eval()

        with torch.no_grad():
            for batch in train_loader:
                features, _, sample_ids = batch
                features = features.to(self.device)
                sample_ids = sample_ids.to(self.device)

                all_errors = self.ae_manager.compute_all_reconstruction_errors(features)
                batch_candidates = self.ledger.compute_candidate_rer(sample_ids, all_errors)
                all_candidates.update(batch_candidates)

        # Return to train mode
        self.ae_manager.train()

        return all_candidates

    def forward(self, x: Tensor) -> dict:
        """Forward pass computing reconstruction errors and assignments.

        Args:
            x: Features [batch_size, feature_dim]

        Returns:
            Dictionary with errors and predictions
        """
        all_errors = self.ae_manager.compute_all_reconstruction_errors(x)
        predictions = all_errors.argmin(dim=1)

        return {
            "all_errors": all_errors,
            "predictions": predictions,
        }

    def _compute_prototype_anchor_loss(self) -> Tensor:
        """Compute loss that keeps AEs anchored to their prototypes."""
        if self._prototype_features is None or len(self._prototype_features) == 0:
            return torch.tensor(0.0, device=self.device)

        proto_features = self._prototype_features.to(self.device)
        proto_classes = self._prototype_classes.to(self.device)

        total_loss = torch.tensor(0.0, device=self.device)
        num_protos = 0

        for class_idx in range(self.num_classes):
            mask = proto_classes == class_idx
            if not mask.any():
                continue

            class_proto_features = proto_features[mask]
            ae = self.ae_manager.autoencoders[class_idx]

            # Each prototype should reconstruct well by its designated AE
            x_hat, _ = ae(class_proto_features)
            loss = F.mse_loss(x_hat, class_proto_features)
            total_loss = total_loss + loss
            num_protos += mask.sum().item()

        if num_protos > 0:
            total_loss = total_loss / self.num_classes

        return total_loss

    def training_step(self, batch, batch_idx):
        """Unsupervised training step.

        1. Get current assignments from ledger (read-only)
        2. Compute reconstruction errors for assigned samples
        3. Train AEs on their assigned samples

        Note: Assignments are made at epoch start via on_train_epoch_start,
        not during training steps.
        """
        features, true_labels, sample_ids = batch

        # Compute reconstruction errors from all AEs
        # all_errors = self.ae_manager.compute_all_reconstruction_errors(features)

        # Get current assignments from ledger (read-only, no new assignments)
        assigned_classes = self.ledger.get_class_assignments_tensor(sample_ids)

        # Train each AE on its assigned samples
        total_loss = torch.tensor(0.0, device=self.device)
        total_recon_loss = torch.tensor(0.0, device=self.device)
        total_umap_loss = torch.tensor(0.0, device=self.device)
        total_l2_loss = torch.tensor(0.0, device=self.device)
        classes_with_samples = 0

        for class_idx in range(self.num_classes):
            mask = assigned_classes == class_idx
            num_samples = mask.sum().item()
            ae = self.ae_manager.autoencoders[class_idx]

            # L2 regularization applies regardless of assignments,
            # ensuring the loss always has a grad_fn even with empty batches.
            total_l2_loss = total_l2_loss + ae.get_l2_regularization()

            if not mask.any():
                continue

            classes_with_samples += 1
            class_features = features[mask]

            # Reconstruction loss
            x_hat, z = ae(class_features)
            recon_loss = F.mse_loss(x_hat, class_features)
            total_recon_loss = total_recon_loss + recon_loss

            # UMAP loss (if enabled)
            if self.use_umap_loss and num_samples > 1:
                umap_loss = compute_umap_loss(class_features, z, self.umap_min_dist)
                total_umap_loss = total_umap_loss + umap_loss * self.umap_loss_weight

            # Log per-class loss
            self.log(
                f"train/loss_class_{class_idx}",
                recon_loss,
                on_step=False,
                on_epoch=True,
            )

        # Prototype anchoring loss
        anchor_loss = torch.tensor(0.0, device=self.device)
        if self.use_prototype_anchoring:
            anchor_loss = self._compute_prototype_anchor_loss()

        # Combine losses
        total_loss = (
            total_recon_loss
            + total_umap_loss
            + total_l2_loss
            + self.prototype_anchor_weight * anchor_loss
        )

        # Log metrics
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss", total_recon_loss, on_step=False, on_epoch=True)
        self.log("train/l2_loss", total_l2_loss, on_step=False, on_epoch=True)
        self.log("train/anchor_loss", anchor_loss, on_step=False, on_epoch=True)
        self.log(
            "train/classes_with_samples",
            float(classes_with_samples),
            on_step=False,
            on_epoch=True,
        )
        if self.use_umap_loss:
            self.log("train/umap_loss", total_umap_loss, on_step=False, on_epoch=True)

        # Log assignment statistics
        assigned_count = (assigned_classes >= 0).sum()
        self.log("train/assigned_in_batch", assigned_count, on_step=False, on_epoch=True)

        return total_loss

    def on_train_epoch_end(self) -> None:
        """Perform assignment/pruning and log statistics at end of epoch.

        After training completes for this epoch:
        1. Re-evaluate all samples and perform assignment/pruning
        2. Log statistics (reflecting state ready for next epoch)
        """
        if self.ledger is None:
            return

        # Perform assignment/pruning after training (AEs have seen data)
        self._perform_epoch_assignment_and_pruning()

        # Compute and log assignment statistics
        stats = self.ledger.compute_assignment_stats()

        self.log("train/total_assigned", float(stats["assigned_samples"]))
        self.log("train/total_unassigned", float(stats["unassigned_samples"]))
        self.log("train/prototype_count", float(stats["prototype_count"]))

        # Log epoch-level assignment/pruning counts
        self.log("train/new_assignments_this_epoch", float(self._epoch_new_assignments))
        self.log("train/pruned_this_epoch", float(self._epoch_pruned_samples))

        # Log effective assignment threshold (useful when annealing)
        if self.assignment_mode == "threshold":
            self.log("train/effective_assignment_threshold", self._get_effective_assignment_threshold())

        if stats["assigned_samples"] > 0:
            self.log("train/assignment_rer_mean", stats["rer_mean"])
            self.log("train/assignment_rer_std", stats["rer_std"])

        # Log per-class assignment counts and average RER
        for c in range(self.num_classes):
            #    self.log(
            #        f"train/samples_class_{c}",
            #        float(stats.get(f"samples_class_{c}", 0)),
            #    )
            self.log(
                f"train/avg_rer_class_{c}",
                float(stats.get(f"avg_rer_class_{c}", 0.0)),
            )

    def validation_step(self, batch, batch_idx):
        """Validation with true labels for evaluation."""
        features, true_labels, sample_ids = batch

        # Compute reconstruction errors
        all_errors = self.ae_manager.compute_all_reconstruction_errors(features)

        # Store for epoch metrics
        self._val_outputs.append(
            {
                "all_errors": all_errors.detach(),
                "true_labels": true_labels.detach(),
                "sample_ids": sample_ids.detach(),
            }
        )

        # Log validation loss
        predictions = all_errors.argmin(dim=1)
        accuracy = (predictions == true_labels).float().mean()
        self.log("val/accuracy_step", accuracy, on_step=False, on_epoch=True)

        return {"all_errors": all_errors, "true_labels": true_labels}

    def on_validation_epoch_end(self) -> None:
        """Compute metrics using true labels."""
        self._compute_epoch_metrics(prefix="val")
        self._maybe_log_sample_reconstructions()

    def test_step(self, batch, batch_idx):
        """Test step: same as validation."""
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        """Compute test metrics using true labels."""
        self._compute_epoch_metrics(prefix="test")

    def _compute_epoch_metrics(self, prefix: str = "val") -> None:
        """Compute RER and classification metrics at epoch end."""
        if not self._val_outputs:
            return

        # Aggregate outputs
        all_errors = torch.cat([x["all_errors"] for x in self._val_outputs], dim=0)
        true_labels = torch.cat([x["true_labels"] for x in self._val_outputs], dim=0)
        sample_ids = torch.cat([x["sample_ids"] for x in self._val_outputs], dim=0)

        # Predictions: argmin error
        predictions = all_errors.argmin(dim=1)

        # Classification metrics (using true labels)
        accuracy = compute_accuracy(true_labels, predictions)
        f1_scores = compute_f1_scores(true_labels, predictions)

        self.log(f"{prefix}/accuracy", accuracy, prog_bar=True)
        self.log(f"{prefix}/f1_micro", f1_scores["f1_micro"])
        self.log(f"{prefix}/f1_macro", f1_scores["f1_macro"])

        # RER metrics (using true labels as reference)
        rer_values = compute_rer(all_errors, true_labels)
        # chi = compute_dataset_chi(rer_values)
        # class_chi = compute_class_chi(rer_values, true_labels)

        # self.log(f"{prefix}/chi", chi, prog_bar=True)
        self.log(f"{prefix}/rer_mean", rer_values.mean().detach())
        self.log(f"{prefix}/rer_std", rer_values.std().detach())
        self.log(f"{prefix}/rer_min", rer_values.min().detach())
        self.log(f"{prefix}/rer_max", rer_values.max().detach())
        self.log(f"{prefix}/frac_above_1", (rer_values > 1.0).float().mean().detach())

        # Per-class chi
        # for c, chi_c in class_chi.items():
        #    self.log(f"{prefix}/chi_class_{c}", chi_c)

        # Noise estimation metrics
        # chi_0 = compute_chi_0(all_errors, true_labels)
        # chi_rand = compute_chi_rand(all_errors, true_labels)
        # eta_est = estimate_noise_rate(chi_0, chi_rand)

        # self.log(f"{prefix}/chi_0", chi_0)
        # self.log(f"{prefix}/chi_rand", chi_rand)
        # self.log(f"{prefix}/eta_est", eta_est)

        # Assignment agreement (how well weak labels match true labels)
        if self.ledger is not None:
            agreement = self.ledger.get_assignment_agreement(true_labels, sample_ids)
            self.log(f"{prefix}/assignment_agreement", agreement)

        # Confusion matrix
        if self.logger is not None and hasattr(self.logger, "experiment"):
            artifact_prefix = f"{prefix}_epoch_{self.current_epoch:03d}"
            log_confusion_matrix_artifacts(
                logger=self.logger,
                run_id=self.logger.run_id,
                labels=true_labels,
                predictions=predictions,
                num_classes=self.num_classes,
                prefix=artifact_prefix,
                metadata={
                    "accuracy": accuracy,
                    "f1_micro": f1_scores["f1_micro"],
                    "f1_macro": f1_scores["f1_macro"],
                    # "chi": chi,
                },
            )

            # Log ledger state as JSON artifact
            # self._log_ledger_artifact(prefix)

        # Clear outputs
        self._val_outputs.clear()

    def _maybe_log_sample_reconstructions(self) -> None:
        """Log sample reconstructions if this is a logging epoch."""
        if not self.log_samples or self.log_samples_frequency <= 0:
            return
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return

        epoch = self.current_epoch
        if epoch != 0 and epoch % self.log_samples_frequency != 0:
            return

        # Capture fixed batch on first logging epoch
        if self._sample_log_batch is None:
            self._capture_sample_log_batch()
        if self._sample_log_batch is None:
            return

        self._log_sample_reconstructions()

    def _capture_sample_log_batch(self) -> None:
        """Capture one batch from the val dataloader to reuse for all future logs."""
        if not hasattr(self.trainer, "datamodule") or self.trainer.datamodule is None:
            return
        try:
            val_loader = self.trainer.datamodule.val_dataloader()
            batch = next(iter(val_loader))
            features, true_labels, _ = batch
            self._sample_log_batch = (features.cpu(), true_labels.cpu())
        except Exception as e:
            print(f"Warning: Could not capture sample log batch: {e}")

    def _log_sample_reconstructions(self) -> None:
        """Reconstruct the fixed batch through all K autoencoders and log as artifacts."""
        import csv
        import numpy as np
        from PIL import Image

        features, true_labels = self._sample_log_batch
        features = features.to(self.device)
        epoch = self.current_epoch

        # Get image shape for visualization (optional — skip image logging without it)
        image_shape = None
        if self.hparams.get("autoencoder_kwargs"):
            image_shape = self.hparams["autoencoder_kwargs"].get("image_shape")
            if image_shape is not None:
                image_shape = tuple(image_shape)

        self.ae_manager.eval()
        with torch.no_grad():
            all_errors = self.ae_manager.compute_all_reconstruction_errors(features)
            reconstructions = []
            for ae in self.ae_manager.autoencoders:
                x_hat, _ = ae(features)
                reconstructions.append(x_hat.cpu())
        self.ae_manager.train()

        # reconstructions: list of K tensors, each [batch_size, feature_dim]
        # all_errors: [batch_size, num_classes]
        errors_cpu = all_errors.cpu()
        features_cpu = features.cpu()

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                artifact_base = f"sample_reconstructions/epoch_{epoch:03d}"

                # --- 1) Per-sample reconstruction error CSV ---
                csv_path = tmp_path / "reconstruction_errors.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    header = ["sample_idx", "true_label"] + [
                        f"ae_{k}_error" for k in range(self.num_classes)
                    ]
                    writer.writerow(header)
                    for i in range(len(features_cpu)):
                        row = [i, true_labels[i].item()] + [
                            f"{errors_cpu[i, k].item():.6f}" for k in range(self.num_classes)
                        ]
                        writer.writerow(row)
                self.logger.experiment.log_artifact(
                    run_id=self.logger.run_id,
                    local_path=str(csv_path),
                    artifact_path=artifact_base,
                )

                # --- 2) Image grids (only when image_shape is available) ---
                if image_shape is not None:
                    c, h, w = image_shape

                    def _tensor_to_pil(t: Tensor) -> Image.Image:
                        t = t.clamp(0, 1).view(c, h, w)
                        if c == 1:
                            arr = (t.squeeze(0).numpy() * 255).astype(np.uint8)
                            return Image.fromarray(arr, mode="L")
                        arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        return Image.fromarray(arr, mode="RGB")

                    n_samples = min(len(features_cpu), 16)
                    # Grid: rows = samples, cols = original + K reconstructions
                    n_cols = 1 + self.num_classes
                    pad = 2
                    grid_w = n_cols * (w + pad) + pad
                    grid_h = n_samples * (h + pad) + pad
                    bg = 128 if c == 1 else (128, 128, 128)
                    mode = "L" if c == 1 else "RGB"
                    grid = Image.new(mode, (grid_w, grid_h), bg)

                    for i in range(n_samples):
                        y_off = pad + i * (h + pad)
                        # Original
                        grid.paste(_tensor_to_pil(features_cpu[i]), (pad, y_off))
                        # K reconstructions
                        for k in range(self.num_classes):
                            x_off = pad + (1 + k) * (w + pad)
                            grid.paste(_tensor_to_pil(reconstructions[k][i]), (x_off, y_off))

                    grid_path = tmp_path / "grid.png"
                    grid.save(grid_path)
                    self.logger.experiment.log_artifact(
                        run_id=self.logger.run_id,
                        local_path=str(grid_path),
                        artifact_path=artifact_base,
                    )

        except Exception as e:
            print(f"Warning: Could not log sample reconstructions: {e}")

    def _log_ledger_artifact(self, prefix: str) -> None:
        """Log ledger state as JSON artifact to MLflow.

        Args:
            prefix: Prefix for the artifact filename (e.g., "val", "test")
        """
        if self.ledger is None:
            return
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return

        try:
            ledger_data = self.ledger.to_json_dict()
            ledger_data["epoch"] = self.current_epoch
            ledger_data["prefix"] = prefix

            with tempfile.TemporaryDirectory() as tmp_dir:
                json_path = Path(tmp_dir) / f"ledger_{prefix}_epoch_{self.current_epoch:03d}.json"
                with open(json_path, "w") as f:
                    json.dump(ledger_data, f, indent=2)

                self.logger.experiment.log_artifact(
                    run_id=self.logger.run_id,
                    local_path=str(json_path),
                    artifact_path="ledger",
                )
        except Exception as e:
            print(f"Warning: Could not log ledger artifact: {e}")

    def _log_prototype_images(self) -> None:
        """Log prototype images as MLflow artifacts.

        Logs each prototype image under prototypes/class_{c}/proto_{i}.png.
        Called automatically at the end of setup() after prototypes are loaded.
        """
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return
        if self._prototype_features is None or len(self._prototype_features) == 0:
            return

        # Get image shape from autoencoder_kwargs
        image_shape = None
        if self.hparams.get("autoencoder_kwargs"):
            image_shape = self.hparams["autoencoder_kwargs"].get("image_shape")
            if image_shape is not None:
                image_shape = tuple(image_shape)

        if image_shape is None:
            # Cannot reconstruct images without shape
            return

        try:
            import numpy as np
            from PIL import Image

            c, h, w = image_shape
            img_mode = self.prototype_img_mode if hasattr(self, "prototype_img_mode") else "L"

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                for i in range(len(self._prototype_features)):
                    class_idx = self._prototype_classes[i].item()
                    features = self._prototype_features[i]

                    # Reshape from flattened [C*H*W] to [C, H, W]
                    tensor = features.view(c, h, w)

                    # Convert to PIL Image
                    if img_mode == "L" or c == 1:
                        # Grayscale: [1, H, W] -> [H, W]
                        arr = (tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                        img = Image.fromarray(arr, mode="L")
                    else:
                        # RGB: [3, H, W] -> [H, W, 3]
                        arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        img = Image.fromarray(arr, mode="RGB")

                    # Find prototype index within this class
                    class_protos = (self._prototype_classes[:i+1] == class_idx).sum().item()
                    proto_idx = class_protos - 1

                    # Save image
                    class_dir = tmp_path / f"class_{class_idx}"
                    class_dir.mkdir(exist_ok=True)
                    img_path = class_dir / f"proto_{proto_idx}.png"
                    img.save(img_path)

                # Log all prototype images as artifacts
                for class_dir in tmp_path.iterdir():
                    if class_dir.is_dir():
                        for img_file in class_dir.iterdir():
                            artifact_path = f"prototypes/{class_dir.name}"
                            self.logger.experiment.log_artifact(
                                run_id=self.logger.run_id,
                                local_path=str(img_file),
                                artifact_path=artifact_path,
                            )

        except Exception as e:
            print(f"Warning: Could not log prototype images: {e}")

    def configure_optimizers(self):
        """Configure optimizer and optional cosine LR scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

        if not self.use_cosine_lr:
            return optimizer

        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        t_max = self.trainer.max_epochs or 100
        cosine = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=self.cosine_lr_min)

        if self.cosine_lr_warmup_epochs > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=self.cosine_lr_warmup_epochs,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.cosine_lr_warmup_epochs],
            )
        else:
            scheduler = cosine

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def get_ledger(self) -> PrototypeLedger:
        """Get the assignment ledger."""
        return self.ledger

    def state_dict(self, *args, **kwargs) -> dict:
        """Include ledger state in checkpoint."""
        state = super().state_dict(*args, **kwargs)

        # Add ledger state
        if self.ledger is not None:
            state["_ledger_state"] = self.ledger.state_dict()

        # Add prototype provider state
        if self.prototype_provider is not None:
            state["_prototype_provider_state"] = self.prototype_provider.state_dict()

        return state

    def load_state_dict(self, state_dict: dict, *args, **kwargs) -> None:
        """Load ledger state from checkpoint."""
        # Extract custom state before parent load
        ledger_state = state_dict.pop("_ledger_state", None)
        provider_state = state_dict.pop("_prototype_provider_state", None)

        # Load model weights
        super().load_state_dict(state_dict, *args, **kwargs)

        # Restore ledger
        if ledger_state is not None:
            if self.ledger is None:
                self.ledger = PrototypeLedger(
                    num_classes=ledger_state["num_classes"],
                    rer_threshold=ledger_state["rer_threshold"],
                )
            self.ledger.load_state_dict(ledger_state)

        # Restore prototype provider
        if provider_state is not None:
            if self.prototype_provider is None:
                self.prototype_provider = PrototypeProvider(
                    num_classes=provider_state["num_classes"],
                )
            self.prototype_provider.load_state_dict(provider_state)
            self._update_prototype_cache()
