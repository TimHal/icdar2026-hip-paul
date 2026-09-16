"""Tests for ShallowAutoencoder and ClasswiseAutoencoderManager."""

import pytest
import torch

from src.model.ShallowAutoencoder import ShallowAutoencoder
from src.model.ClasswiseAutoencoderManager import ClasswiseAutoencoderManager
from src.util.rer_metrics import compute_rer


class TestShallowAutoencoder:
    """Tests for the ShallowAutoencoder module."""

    @pytest.fixture
    def autoencoder(self):
        """Create a test autoencoder."""
        return ShallowAutoencoder(
            feature_dim=384,
            n_components=10,
            hidden_dims=[256],
            dropout=0.01,
            l2_reg=1e-6,
        )

    def test_forward_shape(self, autoencoder):
        """Test forward pass output shapes."""
        batch_size = 32
        x = torch.randn(batch_size, 384)

        x_hat, z = autoencoder(x)

        assert x_hat.shape == (batch_size, 384)
        assert z.shape == (batch_size, 10)

    def test_encode_shape(self, autoencoder):
        """Test encoder output shape."""
        batch_size = 32
        x = torch.randn(batch_size, 384)

        z = autoencoder.encode(x)

        assert z.shape == (batch_size, 10)

    def test_decode_shape(self, autoencoder):
        """Test decoder output shape."""
        batch_size = 32
        z = torch.randn(batch_size, 10)

        x_hat = autoencoder.decode(z)

        assert x_hat.shape == (batch_size, 384)

    def test_reconstruction_error_shape(self, autoencoder):
        """Test reconstruction error is per-sample."""
        batch_size = 32
        x = torch.randn(batch_size, 384)

        error = autoencoder.reconstruction_error(x)

        assert error.shape == (batch_size,)
        assert (error >= 0).all()

    def test_reconstruction_loss_scalar(self, autoencoder):
        """Test reconstruction loss is a scalar."""
        batch_size = 32
        x = torch.randn(batch_size, 384)

        loss = autoencoder.reconstruction_loss(x)

        assert loss.dim() == 0
        assert loss >= 0

    def test_l2_regularization(self, autoencoder):
        """Test L2 regularization is non-negative."""
        l2 = autoencoder.get_l2_regularization()

        assert l2.dim() == 0
        assert l2 >= 0

    def test_output_range(self, autoencoder):
        """Test decoder output is in [0, 1] due to sigmoid."""
        batch_size = 32
        x = torch.randn(batch_size, 384)

        x_hat, _ = autoencoder(x)

        assert (x_hat >= 0).all()
        assert (x_hat <= 1).all()

    def test_gradient_flow(self, autoencoder):
        """Test gradients flow through the autoencoder."""
        x = torch.randn(8, 384, requires_grad=True)

        loss = autoencoder.reconstruction_loss(x)
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestClasswiseAutoencoderManager:
    """Tests for the ClasswiseAutoencoderManager module."""

    @pytest.fixture
    def manager(self):
        """Create a test manager with 10 classes."""
        return ClasswiseAutoencoderManager(
            num_classes=10,
            feature_dim=384,
            n_components=10,
            hidden_dims=[256],
            dropout=0.01,
            l2_reg=1e-6,
        )

    def test_num_autoencoders(self, manager):
        """Test correct number of autoencoders are created."""
        assert len(manager.autoencoders) == 10

    def test_compute_all_errors_shape(self, manager):
        """Test all reconstruction errors shape."""
        batch_size = 32
        x = torch.randn(batch_size, 384)

        errors = manager.compute_all_reconstruction_errors(x)

        assert errors.shape == (batch_size, 10)
        assert (errors >= 0).all()

    def test_compute_rer_shape(self, manager):
        """Test RER computation via util.compute_rer."""
        batch_size = 32
        x = torch.randn(batch_size, 384)
        labels = torch.randint(0, 10, (batch_size,))

        all_errors = manager.compute_all_reconstruction_errors(x)
        rer = compute_rer(all_errors, labels)

        assert rer.shape == (batch_size,)
        assert (rer >= 0).all()

    def test_compute_class_loss(self, manager):
        """Test per-class loss computation."""
        batch_size = 32
        features = torch.randn(batch_size, 384)
        labels = torch.zeros(batch_size, dtype=torch.long)  # All class 0

        loss, num_samples = manager.compute_class_loss(features, labels, 0)

        assert loss.dim() == 0
        assert loss >= 0
        assert num_samples == batch_size

    def test_compute_class_loss_empty(self, manager):
        """Test per-class loss with no samples."""
        batch_size = 32
        features = torch.randn(batch_size, 384)
        labels = torch.zeros(batch_size, dtype=torch.long)  # All class 0

        loss, num_samples = manager.compute_class_loss(features, labels, 5)

        assert loss == 0.0
        assert num_samples == 0

    def test_compute_total_loss(self, manager):
        """Test total loss computation."""
        batch_size = 32
        features = torch.randn(batch_size, 384)
        labels = torch.randint(0, 10, (batch_size,))

        total_loss, loss_dict = manager.compute_total_loss(features, labels)

        assert total_loss.dim() == 0
        assert total_loss >= 0
        assert "total_loss" in loss_dict

    def test_get_class_autoencoder(self, manager):
        """Test getting individual autoencoders."""
        ae = manager.get_class_autoencoder(5)

        assert isinstance(ae, ShallowAutoencoder)
        assert ae.feature_dim == 384
