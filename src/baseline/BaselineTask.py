"""Baseline unsupervised classification tasks: Nearest Centroid and GMM.

These baselines operate on pre-extracted features (CLIP, DINO) and produce
an error matrix [N, num_classes] compatible with compute_rer() for direct
comparison against the RERC autoencoder approach.

Neither baseline involves gradient-based training. All fitting happens in
setup(); the Lightning training loop runs for 1 epoch with a dummy loss.

Usage:
    # Nearest Centroid (prototypes as centroids)
    PYTHONPATH=src python src/cli.py fit --config conf/experiment/baseline_centroid_mnist_dino.yaml

    # Gaussian Mixture Model (prototype-initialized)
    PYTHONPATH=src python src/cli.py fit --config conf/experiment/baseline_gmm_mnist_dino.yaml
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor

from data.PrototypeProvider import PrototypeProvider
from util.classification_metrics import (
    compute_accuracy,
    compute_f1_scores,
    log_confusion_matrix_artifacts,
)
from util.rer_metrics import compute_rer


class _BaselineTask(L.LightningModule):
    """Abstract base for non-parametric baseline classifiers.

    Subclasses implement _fit() and _compute_error_matrix().
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        prototypes_per_class: int = 1,
        prototype_selection: str = "random",
        prototype_source: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.prototypes_per_class = prototypes_per_class
        self.prototype_selection = prototype_selection
        self.prototype_source = prototype_source

        # Dummy parameter so Lightning has something to optimize
        self._dummy_param = nn.Parameter(torch.zeros(1))

        self.prototype_provider: PrototypeProvider | None = None
        self._val_outputs: list[dict] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        if stage not in ("fit", "test"):
            return

        datamodule = self.trainer.datamodule
        if hasattr(datamodule, "setup"):
            datamodule.setup("fit" if stage == "fit" else "test")

        # Collect all training features
        loader = datamodule.train_dataloader()
        all_features, all_labels, all_ids = [], [], []
        for features, labels, ids in loader:
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
            all_ids.append(ids.cpu())

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_ids = torch.cat(all_ids, dim=0)

        # Load or auto-select prototypes
        self._init_prototypes(all_features, all_labels, all_ids)

        # Delegate to subclass
        self._fit(all_features, all_labels, all_ids)

    def _init_prototypes(
        self,
        all_features: Tensor,
        all_labels: Tensor,
        all_ids: Tensor,
    ) -> None:
        """Load prototypes from external source or auto-select from data."""
        if self.prototype_source is not None:
            source = Path(self.prototype_source)
            if source.is_file() and source.suffix == ".pth":
                self.prototype_provider = PrototypeProvider.from_pth_file(source)
            elif source.is_dir():
                raise ValueError(
                    "Folder-based prototype_source requires image_shape, "
                    "which is not applicable for pre-extracted features. "
                    "Use a .pth file instead."
                )
            else:
                raise ValueError(
                    f"Invalid prototype_source: {self.prototype_source}. "
                    "Must be a .pth file."
                )
        else:
            self.prototype_provider = PrototypeProvider.from_labels(
                features=all_features,
                labels=all_labels,
                sample_ids=all_ids,
                prototypes_per_class=self.prototypes_per_class,
                selection=self.prototype_selection,
            )

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _fit(
        self,
        all_features: Tensor,
        all_labels: Tensor,
        all_ids: Tensor,
    ) -> None:
        """Fit the baseline model on training data. Called from setup()."""

    @abstractmethod
    def _compute_error_matrix(self, features: Tensor) -> Tensor:
        """Compute per-class error/distance matrix.

        Args:
            features: [B, D] batch of features

        Returns:
            [B, num_classes] distance matrix (lower = better fit)
        """

    # ------------------------------------------------------------------
    # Lightning training loop (dummy)
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        return self._dummy_param.sum() * 0

    def configure_optimizers(self):
        return torch.optim.SGD([self._dummy_param], lr=0.0)

    # ------------------------------------------------------------------
    # Validation / Test
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        features, true_labels, sample_ids = batch
        all_errors = self._compute_error_matrix(features)
        self._val_outputs.append({
            "all_errors": all_errors.detach().cpu(),
            "true_labels": true_labels.detach().cpu(),
        })

    def on_validation_epoch_end(self):
        self._compute_epoch_metrics("val")

    def test_step(self, batch, batch_idx):
        features, true_labels, sample_ids = batch
        all_errors = self._compute_error_matrix(features)
        self._val_outputs.append({
            "all_errors": all_errors.detach().cpu(),
            "true_labels": true_labels.detach().cpu(),
        })

    def on_test_epoch_end(self):
        self._compute_epoch_metrics("test")

    def _compute_epoch_metrics(self, prefix: str) -> None:
        if not self._val_outputs:
            return

        all_errors = torch.cat([x["all_errors"] for x in self._val_outputs], dim=0)
        true_labels = torch.cat([x["true_labels"] for x in self._val_outputs], dim=0)
        self._val_outputs.clear()

        # Predictions: lowest error = best class
        predictions = all_errors.argmin(dim=1)

        # Classification metrics
        accuracy = compute_accuracy(true_labels, predictions)
        f1_scores = compute_f1_scores(true_labels, predictions)

        self.log(f"{prefix}/accuracy", accuracy, prog_bar=True)
        self.log(f"{prefix}/f1_micro", f1_scores["f1_micro"])
        self.log(f"{prefix}/f1_macro", f1_scores["f1_macro"])

        # RER metrics
        rer_values = compute_rer(all_errors, true_labels)
        self.log(f"{prefix}/rer_mean", rer_values.mean())
        self.log(f"{prefix}/rer_std", rer_values.std())
        self.log(f"{prefix}/rer_min", rer_values.min())
        self.log(f"{prefix}/rer_max", rer_values.max())
        self.log(f"{prefix}/frac_above_1", (rer_values > 1.0).float().mean())
        self.log(f"{prefix}/chi", rer_values.mean())

        # Confusion matrix artifact
        if hasattr(self, "logger") and self.logger is not None:
            try:
                log_confusion_matrix_artifacts(
                    logger=self.logger,
                    run_id=self.logger.run_id,
                    labels=true_labels,
                    predictions=predictions,
                    num_classes=self.num_classes,
                    prefix=f"{prefix}_baseline",
                )
            except Exception:
                pass  # MLflow may not be available


class NearestCentroidTask(_BaselineTask):
    """Nearest Centroid baseline using prototypes as centroids.

    Each prototype becomes a centroid. With multiple prototypes per class,
    each class has multiple centroids; classification uses the minimum
    distance to any centroid of that class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.centroids: Tensor | None = None
        self.centroid_classes: Tensor | None = None

    def _fit(self, all_features, all_labels, all_ids):
        proto_features, proto_classes, _ = (
            self.prototype_provider.get_prototype_features()
        )
        self.centroids = proto_features
        self.centroid_classes = proto_classes

        n_per_class = {}
        for c in proto_classes.tolist():
            n_per_class[c] = n_per_class.get(c, 0) + 1
        print(
            f"NearestCentroid: {len(proto_features)} centroids "
            f"({n_per_class})"
        )

    def _compute_error_matrix(self, features: Tensor) -> Tensor:
        centroids = self.centroids.to(features.device)
        centroid_classes = self.centroid_classes.to(features.device)

        # [B, P] distances to all centroids
        dists = torch.cdist(features, centroids)

        # Reduce to per-class minimum distance
        B = features.shape[0]
        error_matrix = torch.full(
            (B, self.num_classes), float("inf"), device=features.device
        )
        for c in range(self.num_classes):
            mask = centroid_classes == c
            if mask.any():
                error_matrix[:, c] = dists[:, mask].min(dim=1).values

        return error_matrix


class GMMTask(_BaselineTask):
    """Gaussian Mixture Model baseline with prototype-initialized components.

    Fully unsupervised: fits a GMM with n_components = num_classes on all
    training features. One Gaussian per class. Multiple prototypes per class
    are used to compute the initial mean (average of prototypes) and initial
    covariance (sample covariance of prototypes) for that class's component.

    Error matrix: negative log-likelihood per component = per class.
    """

    def __init__(
        self,
        covariance_type: str = "diag",
        max_iter: int = 200,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self._gmm = None

    def _fit(self, all_features, all_labels, all_ids):
        from sklearn.mixture import GaussianMixture
        import numpy as np

        proto_features, proto_classes, _ = (
            self.prototype_provider.get_prototype_features()
        )

        # Compute per-class initial mean from prototypes
        means_init = np.zeros((self.num_classes, self.feature_dim), dtype=np.float64)
        for c in range(self.num_classes):
            mask = proto_classes == c
            if mask.any():
                means_init[c] = proto_features[mask].mean(dim=0).numpy()

        # Compute per-class initial precisions from prototypes (where possible)
        # For classes with >1 prototype, use sample variance; otherwise skip
        # and let sklearn use its default initialization for precisions.
        precisions_init = None
        has_covariance_info = all(
            (proto_classes == c).sum() > 1 for c in range(self.num_classes)
        )

        if has_covariance_info and self.covariance_type == "diag":
            # Diagonal precisions: 1 / variance per feature per class
            precisions_init = np.zeros(
                (self.num_classes, self.feature_dim), dtype=np.float64
            )
            for c in range(self.num_classes):
                mask = proto_classes == c
                class_protos = proto_features[mask].numpy()
                var = np.var(class_protos, axis=0) + 1e-6  # avoid div-by-zero
                precisions_init[c] = 1.0 / var

        gmm_kwargs = dict(
            n_components=self.num_classes,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            means_init=means_init,
            random_state=0,
        )
        if precisions_init is not None:
            gmm_kwargs["precisions_init"] = precisions_init

        gmm = GaussianMixture(**gmm_kwargs)
        gmm.fit(all_features.numpy())
        self._gmm = gmm

        protos_per_class = {
            c: int((proto_classes == c).sum()) for c in range(self.num_classes)
        }
        print(
            f"GMM: {self.num_classes} components (1 per class), "
            f"prototypes used for init: {protos_per_class}, "
            f"covariance={self.covariance_type}, "
            f"converged={gmm.converged_}, n_iter={gmm.n_iter_}"
        )

    def _compute_error_matrix(self, features: Tensor) -> Tensor:
        X = features.cpu().numpy()

        # Per-component weighted log-prob: [B, num_classes]
        # Component i = class i, so this is already the per-class matrix
        log_prob = self._gmm._estimate_weighted_log_prob(X)
        log_prob = torch.from_numpy(log_prob.astype("float32"))

        # Negative log-prob as "error" (lower = better fit for that class)
        return -log_prob
