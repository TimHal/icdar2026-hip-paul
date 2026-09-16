"""Integration tests for the full training pipeline."""

import pytest
import torch
import tempfile
from pathlib import Path

from src.model.ShallowAutoencoder import ShallowAutoencoder
from src.model.ClasswiseAutoencoderManager import ClasswiseAutoencoderManager
from src.task.ClasswiseAETask import ClasswiseAETask
from src.util.losses import compute_umap_loss
from src.util.rer_metrics import compute_rer, compute_rer_statistics


class TestTrainingPipeline:
    """Integration tests for the training pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        # Simulate features from 3 classes
        num_samples = 300
        feature_dim = 384
        num_classes = 3

        features = torch.randn(num_samples, feature_dim)
        labels = torch.randint(0, num_classes, (num_samples,))

        return features, labels, num_classes, feature_dim

    def test_autoencoder_training_step(self, sample_data):
        """Test a single training step reduces loss."""
        features, labels, num_classes, feature_dim = sample_data

        # Create task
        task = ClasswiseAETask(
            num_classes=num_classes,
            feature_dim=feature_dim,
            n_components=10,
            learning_rate=0.01,
            use_umap_loss=False,  # Disable for faster test
        )

        # Compute initial loss
        initial_loss, _ = task.ae_manager.compute_total_loss(features, labels)

        # Run optimizer step
        optimizer = torch.optim.Adam(task.parameters(), lr=0.01)
        for _ in range(10):  # 10 steps
            optimizer.zero_grad()
            loss, _ = task.ae_manager.compute_total_loss(features, labels)
            loss.backward()
            optimizer.step()

        # Compute final loss
        final_loss, _ = task.ae_manager.compute_total_loss(features, labels)

        assert final_loss < initial_loss

    def test_rer_computation_end_to_end(self, sample_data):
        """Test RER computation from features to statistics."""
        features, labels, num_classes, feature_dim = sample_data

        # Create manager
        manager = ClasswiseAutoencoderManager(
            num_classes=num_classes,
            feature_dim=feature_dim,
            n_components=10,
        )

        # Train briefly
        optimizer = torch.optim.Adam(manager.parameters(), lr=0.01)
        for _ in range(5):
            loss, _ = manager.compute_total_loss(features, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute RER
        all_errors = manager.compute_all_reconstruction_errors(features)
        rer_values = compute_rer(all_errors, labels)

        # Compute statistics
        stats = compute_rer_statistics(rer_values, labels)

        assert "chi" in stats
        assert stats["chi"] > 0
        assert "rer_mean" in stats

    def test_task_forward_pass(self, sample_data):
        """Test full forward pass through the task."""
        features, labels, num_classes, feature_dim = sample_data

        task = ClasswiseAETask(
            num_classes=num_classes,
            feature_dim=feature_dim,
            n_components=10,
            use_umap_loss=False,
        )

        # Forward pass computes RER
        rer = task(features, labels)

        assert rer.shape == (len(features),)
        assert (rer >= 0).all()

    def test_umap_loss_computation(self, sample_data):
        """Test UMAP loss is computed without errors."""
        features, labels, num_classes, feature_dim = sample_data

        task = ClasswiseAETask(
            num_classes=num_classes,
            feature_dim=feature_dim,
            n_components=10,
            use_umap_loss=True,
            umap_loss_weight=0.05,
        )

        # Get some class features and their latent representations
        class_mask = labels == 0
        class_features = features[class_mask][:32]  # Take first 32
        ae = task.ae_manager.autoencoders[0]
        _, latents = ae(class_features)

        # Compute UMAP loss
        umap_loss = compute_umap_loss(class_features, latents)

        assert umap_loss.dim() == 0
        assert umap_loss >= 0
        assert not torch.isnan(umap_loss)


class TestCheckpointSaveLoad:
    """Tests for checkpoint save/load functionality."""

    @pytest.fixture
    def task(self):
        """Create a task for testing."""
        return ClasswiseAETask(
            num_classes=10,
            feature_dim=384,
            n_components=10,
            use_umap_loss=False,
        )

    def test_state_dict_roundtrip(self, task):
        """Test saving and loading state dict."""
        # Get initial state
        state_dict = task.state_dict()

        # Create new task with same config
        new_task = ClasswiseAETask(
            num_classes=10,
            feature_dim=384,
            n_components=10,
            use_umap_loss=False,
        )

        # Load state
        new_task.load_state_dict(state_dict)

        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            task.named_parameters(), new_task.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)

    def test_checkpoint_file_roundtrip(self, task):
        """Test saving to and loading from a checkpoint file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.ckpt"

            # Save checkpoint
            torch.save(task.state_dict(), ckpt_path)

            # Load into new task
            new_task = ClasswiseAETask(
                num_classes=10,
                feature_dim=384,
                n_components=10,
                use_umap_loss=False,
            )
            new_task.load_state_dict(torch.load(ckpt_path, weights_only=True))

            # Verify same outputs
            x = torch.randn(8, 384)
            labels = torch.randint(0, 10, (8,))

            task.eval()
            new_task.eval()

            with torch.no_grad():
                rer1 = task(x, labels)
                rer2 = new_task(x, labels)

            assert torch.allclose(rer1, rer2)


class TestGPUCompatibility:
    """Tests for GPU compatibility (skipped if no GPU available)."""

    @pytest.fixture
    def device(self):
        """Get available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        pytest.skip("CUDA not available")

    def test_task_on_gpu(self, device):
        """Test task runs on GPU."""
        task = ClasswiseAETask(
            num_classes=10,
            feature_dim=384,
            n_components=10,
            use_umap_loss=False,
        ).to(device)

        x = torch.randn(32, 384, device=device)
        labels = torch.randint(0, 10, (32,), device=device)

        rer = task(x, labels)

        assert rer.device.type == device.type
        assert not torch.isnan(rer).any()

    def test_training_step_on_gpu(self, device):
        """Test training step on GPU."""
        task = ClasswiseAETask(
            num_classes=10,
            feature_dim=384,
            n_components=10,
            use_umap_loss=True,
        ).to(device)

        x = torch.randn(32, 384, device=device)
        labels = torch.randint(0, 10, (32,), device=device)
        batch = (x, labels)

        # Simulate training step
        loss = task.training_step(batch, 0)

        assert loss.device.type == device.type
        assert not torch.isnan(loss)
