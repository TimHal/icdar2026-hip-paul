"""Tests for JITClasswiseAETask with mock feature extractors."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.task.JITClasswiseAETask import JITClasswiseAETask
from src.feature_extractors.base import FeatureExtractor


class MockFeatureExtractor(FeatureExtractor):
    """Deterministic mock extractor for testing (no model download required)."""

    def __init__(self, feature_dim: int = 64):
        self._feature_dim = feature_dim
        self.device = torch.device("cpu")
        # Fixed random projection to simulate feature extraction
        self._projection = torch.randn(3 * 8 * 8, feature_dim)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def model_name(self) -> str:
        return "mock"

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dims and project to feature_dim
        flat = images.reshape(images.shape[0], -1)[:, : self._projection.shape[0]]
        # Pad if needed
        if flat.shape[1] < self._projection.shape[0]:
            flat = torch.nn.functional.pad(flat, (0, self._projection.shape[0] - flat.shape[1]))
        return flat @ self._projection.to(flat.device)


@pytest.fixture
def extractor():
    return MockFeatureExtractor(feature_dim=64)


@pytest.fixture
def image_batch():
    """Fake image batch: (images, labels, indices)."""
    images = torch.randn(32, 3, 8, 8)
    labels = torch.randint(0, 3, (32,))
    indices = torch.arange(32)
    return images, labels, indices


@pytest.fixture
def image_dataloader(image_batch):
    images, labels, indices = image_batch
    ds = TensorDataset(images, labels, indices)
    return DataLoader(ds, batch_size=16)


class TestJITClasswiseAETask:

    def test_init_infers_feature_dim(self, extractor):
        task = JITClasswiseAETask(
            feature_extractor=extractor,
            num_classes=3,
            n_components=5,
            use_umap_loss=False,
        )
        assert task.feature_dim == 64

    def test_extract_features(self, extractor, image_batch):
        task = JITClasswiseAETask(
            feature_extractor=extractor,
            num_classes=3,
            n_components=5,
            use_umap_loss=False,
            normalize_features=False,
        )
        images, _, _ = image_batch
        features = task._extract_features(images)
        assert features.shape == (32, 64)

    def test_extract_features_with_normalization(self, extractor, image_dataloader):
        task = JITClasswiseAETask(
            feature_extractor=extractor,
            num_classes=3,
            n_components=5,
            use_umap_loss=False,
            normalize_features=True,
        )
        # Compute norm params manually (simulating on_fit_start)
        task._norm_params = task._compute_norm_params(image_dataloader)

        images = next(iter(image_dataloader))[0]
        features = task._extract_features(images)
        assert features.shape == (16, 64)
        # Features should be in [0, 1] after MinMax normalization
        assert features.min() >= 0.0 - 1e-6
        assert features.max() <= 1.0 + 1e-6

    def test_training_step(self, extractor, image_batch):
        task = JITClasswiseAETask(
            feature_extractor=extractor,
            num_classes=3,
            n_components=5,
            use_umap_loss=False,
            normalize_features=False,
        )
        loss = task.training_step(image_batch, batch_idx=0)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_validation_step(self, extractor, image_batch):
        task = JITClasswiseAETask(
            feature_extractor=extractor,
            num_classes=3,
            n_components=5,
            use_umap_loss=False,
            normalize_features=False,
        )
        result = task.validation_step(image_batch, batch_idx=0)
        assert "all_errors" in result
        assert "labels" in result
        assert result["all_errors"].shape == (32, 3)

    def test_forward_computes_rer(self, extractor, image_batch):
        task = JITClasswiseAETask(
            feature_extractor=extractor,
            num_classes=3,
            n_components=5,
            use_umap_loss=False,
            normalize_features=False,
        )
        images, labels, _ = image_batch
        # Forward on the task still expects features (parent behavior),
        # so we extract manually for this test
        features = task._extract_features(images)
        rer = task(features, labels)
        assert rer.shape == (32,)
        assert (rer >= 0).all()

    def test_training_reduces_loss(self, extractor):
        """Verify that a few optimizer steps reduce the loss."""
        images = torch.randn(64, 3, 8, 8)
        labels = torch.randint(0, 3, (64,))
        ids = torch.arange(64)

        task = JITClasswiseAETask(
            feature_extractor=extractor,
            num_classes=3,
            n_components=5,
            use_umap_loss=False,
            normalize_features=False,
        )

        features = task._extract_features(images)
        initial_loss, _ = task.ae_manager.compute_total_loss(features, labels)

        optimizer = torch.optim.Adam(task.parameters(), lr=0.01)
        for _ in range(10):
            optimizer.zero_grad()
            loss = task.training_step((images, labels, ids), batch_idx=0)
            loss.backward()
            optimizer.step()

        features = task._extract_features(images)
        final_loss, _ = task.ae_manager.compute_total_loss(features, labels)
        assert final_loss < initial_loss
