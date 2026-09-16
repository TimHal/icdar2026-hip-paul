"""Tests for MAE checkpoint saving and loading with config storage.

Run with: PYTHONPATH=src python -m pytest tests/test_mae_checkpoint.py -v
"""

import tempfile
from pathlib import Path

import pytest
import torch
import lightning as L

# Import from the same paths as the source code uses (requires PYTHONPATH=src)
from mae.mae_model import MaskedAutoencoder, MAEConfig
from mae.multiview_mae import MultiViewMAE
from task.MAETask import MAETask, load_mae_from_checkpoint
from feature_extractors.mae import MAEFeatureExtractor
from util.model_loading import (
    load_checkpoint_config,
    get_checkpoint_model_config,
    checkpoint_has_config,
    load_mae_model,
)


class TestMAEConfig:
    """Tests for MAEConfig dataclass."""

    def test_config_default_values(self):
        """Test MAEConfig has sensible defaults."""
        config = MAEConfig()
        assert config.img_size == 64
        assert config.patch_size == 8
        assert config.embed_dim == 128
        assert config.encoder_depth == 2
        assert config.use_cls_token is True

    def test_config_to_dict(self):
        """Test config serialization to dict."""
        config = MAEConfig(img_size=32, embed_dim=256)
        d = config.to_dict()

        assert d["img_size"] == 32
        assert d["embed_dim"] == 256
        assert isinstance(d, dict)

    def test_config_from_dict(self):
        """Test config deserialization from dict."""
        d = {"img_size": 48, "patch_size": 4, "embed_dim": 192}
        config = MAEConfig.from_dict(d)

        assert config.img_size == 48
        assert config.patch_size == 4
        assert config.embed_dim == 192
        # Check defaults are used for missing keys
        assert config.encoder_depth == 2

    def test_config_roundtrip(self):
        """Test config survives dict serialization roundtrip."""
        original = MAEConfig(
            img_size=32,
            patch_size=4,
            embed_dim=96,
            encoder_depth=4,
            encoder_heads=6,
        )

        d = original.to_dict()
        restored = MAEConfig.from_dict(d)

        assert original == restored


class TestMaskedAutoencoderConfig:
    """Tests for MaskedAutoencoder config methods."""

    @pytest.fixture
    def mae_model(self):
        """Create a small MAE for testing."""
        return MaskedAutoencoder(
            img_size=32,
            patch_size=4,
            in_chans=1,
            embed_dim=64,
            encoder_depth=2,
            encoder_heads=2,
            decoder_embed_dim=32,
            decoder_depth=1,
        )

    def test_get_config(self, mae_model):
        """Test get_config returns valid MAEConfig."""
        config = mae_model.get_config()

        assert isinstance(config, MAEConfig)
        assert config.img_size == 32
        assert config.patch_size == 4
        assert config.embed_dim == 64
        assert config.encoder_depth == 2

    def test_from_config(self, mae_model):
        """Test from_config creates equivalent model."""
        config = mae_model.get_config()
        new_model = MaskedAutoencoder.from_config(config)

        # Check architecture matches
        assert new_model.img_size == mae_model.img_size
        assert new_model.patch_size == mae_model.patch_size
        assert new_model.embed_dim == mae_model.embed_dim
        assert len(new_model.encoder.blocks) == len(mae_model.encoder.blocks)

    def test_model_config_roundtrip(self, mae_model):
        """Test model can be recreated from its own config."""
        config = mae_model.get_config()
        new_model = MaskedAutoencoder.from_config(config)

        # Verify forward pass works
        x = torch.randn(2, 1, 32, 32)
        out1 = mae_model(x)
        out2 = new_model(x)

        # Shapes should match
        assert out1["pred"].shape == out2["pred"].shape
        assert out1["mask"].shape == out2["mask"].shape


class TestMAETaskCheckpoint:
    """Tests for MAETask checkpoint saving with config."""

    @pytest.fixture
    def mae_task(self):
        """Create MAETask for testing."""
        model = MaskedAutoencoder(
            img_size=32,
            patch_size=4,
            in_chans=1,
            embed_dim=64,
            encoder_depth=2,
            encoder_heads=2,
        )
        return MAETask(model=model, learning_rate=1e-4)

    def test_task_saves_config_in_hparams(self, mae_task):
        """Test MAETask stores model config in hyperparameters."""
        assert "mae_config" in mae_task.hparams
        assert isinstance(mae_task.hparams["mae_config"], dict)
        assert mae_task.hparams["mae_config"]["img_size"] == 32
        assert mae_task.hparams["mae_config"]["embed_dim"] == 64

    def test_checkpoint_contains_config(self, mae_task):
        """Test saved checkpoint contains model config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.ckpt"

            # Simulate Lightning checkpoint save
            checkpoint = {
                "state_dict": mae_task.state_dict(),
                "hyper_parameters": dict(mae_task.hparams),
            }
            torch.save(checkpoint, ckpt_path)

            # Load and verify
            loaded = torch.load(ckpt_path, weights_only=False)
            assert "mae_config" in loaded["hyper_parameters"]

    def test_load_mae_from_checkpoint_with_config(self, mae_task):
        """Test load_mae_from_checkpoint uses stored config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.ckpt"

            # Save checkpoint
            checkpoint = {
                "state_dict": mae_task.state_dict(),
                "hyper_parameters": dict(mae_task.hparams),
            }
            torch.save(checkpoint, ckpt_path)

            # Load model
            loaded_model = load_mae_from_checkpoint(str(ckpt_path))

            # Verify architecture matches
            assert loaded_model.img_size == 32
            assert loaded_model.embed_dim == 64
            assert len(loaded_model.encoder.blocks) == 2


class TestMAEFeatureExtractorCheckpoint:
    """Tests for MAEFeatureExtractor checkpoint loading."""

    @pytest.fixture
    def checkpoint_path(self):
        """Create a checkpoint for testing."""
        model = MaskedAutoencoder(
            img_size=32,
            patch_size=4,
            in_chans=1,
            embed_dim=64,
            encoder_depth=2,
            encoder_heads=2,
        )
        task = MAETask(model=model)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.ckpt"
            checkpoint = {
                "state_dict": task.state_dict(),
                "hyper_parameters": dict(task.hparams),
            }
            torch.save(checkpoint, ckpt_path)
            yield str(ckpt_path)

    def test_from_checkpoint_loads_config(self, checkpoint_path):
        """Test MAEFeatureExtractor.from_checkpoint uses stored config."""
        extractor = MAEFeatureExtractor.from_checkpoint(
            checkpoint_path, pooling="cls", device="cpu"
        )

        assert extractor.config.img_size == 32
        assert extractor.config.embed_dim == 64
        assert extractor.feature_dim == 64

    def test_from_checkpoint_extracts_features(self, checkpoint_path):
        """Test feature extraction works after checkpoint loading."""
        extractor = MAEFeatureExtractor.from_checkpoint(
            checkpoint_path, pooling="cls", device="cpu"
        )

        x = torch.randn(4, 1, 32, 32)
        features = extractor.extract(x)

        assert features.shape == (4, 64)

    def test_pooling_options(self, checkpoint_path):
        """Test different pooling options work correctly."""
        # CLS pooling
        extractor_cls = MAEFeatureExtractor.from_checkpoint(
            checkpoint_path, pooling="cls", device="cpu"
        )
        assert extractor_cls.feature_dim == 64

        # Mean pooling
        extractor_mean = MAEFeatureExtractor.from_checkpoint(
            checkpoint_path, pooling="mean", device="cpu"
        )
        assert extractor_mean.feature_dim == 64

        # Both (concatenated)
        extractor_both = MAEFeatureExtractor.from_checkpoint(
            checkpoint_path, pooling="both", device="cpu"
        )
        assert extractor_both.feature_dim == 128

    def test_constructor_with_config(self):
        """Test MAEFeatureExtractor can be created with explicit config."""
        config = MAEConfig(img_size=48, patch_size=6, embed_dim=96)
        extractor = MAEFeatureExtractor(config=config, device="cpu")

        assert extractor.config == config
        assert extractor.feature_dim == 96

        x = torch.randn(2, 1, 48, 48)
        features = extractor.extract(x)
        assert features.shape == (2, 96)


class TestMultiViewMAECheckpoint:
    """Tests for MultiViewMAE checkpoint loading."""

    @pytest.fixture
    def multiview_checkpoint_path(self):
        """Create a multi-view checkpoint for testing."""
        base_mae = MaskedAutoencoder(
            img_size=32,
            patch_size=4,
            in_chans=1,
            embed_dim=64,
            encoder_depth=2,
            encoder_heads=2,
        )
        model = MultiViewMAE(base_mae=base_mae, share_masking=True)
        task = MAETask(model=model)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "multiview.ckpt"
            checkpoint = {
                "state_dict": task.state_dict(),
                "hyper_parameters": dict(task.hparams),
            }
            torch.save(checkpoint, ckpt_path)
            yield str(ckpt_path)

    def test_multiview_saves_config(self, multiview_checkpoint_path):
        """Test MultiViewMAE checkpoint contains config."""
        hparams = load_checkpoint_config(multiview_checkpoint_path)
        assert "mae_config" in hparams
        assert "is_multiview" in hparams
        assert hparams["is_multiview"] is True

    def test_multiview_get_config(self):
        """Test MultiViewMAE.get_config returns proper config."""
        base_mae = MaskedAutoencoder(img_size=32, patch_size=4)
        model = MultiViewMAE(base_mae=base_mae, share_masking=True)

        config = model.get_config()

        assert "mae_config" in config
        assert "share_masking" in config
        assert config["share_masking"] is True

    def test_multiview_from_config(self):
        """Test MultiViewMAE.from_config creates equivalent model."""
        base_mae = MaskedAutoencoder(img_size=32, patch_size=4)
        original = MultiViewMAE(base_mae=base_mae, share_masking=True)

        config = original.get_config()
        restored = MultiViewMAE.from_config(config)

        assert restored.share_masking == original.share_masking
        assert restored.mae.img_size == original.mae.img_size


class TestModelLoadingUtilities:
    """Tests for model_loading.py utilities."""

    @pytest.fixture
    def checkpoint_path(self):
        """Create a checkpoint for testing utilities."""
        model = MaskedAutoencoder(img_size=32, patch_size=4)
        task = MAETask(model=model)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.ckpt"
            checkpoint = {
                "state_dict": task.state_dict(),
                "hyper_parameters": dict(task.hparams),
            }
            torch.save(checkpoint, ckpt_path)
            yield str(ckpt_path)

    def test_load_checkpoint_config(self, checkpoint_path):
        """Test load_checkpoint_config extracts hparams."""
        hparams = load_checkpoint_config(checkpoint_path)

        assert isinstance(hparams, dict)
        assert "mae_config" in hparams

    def test_get_checkpoint_model_config(self, checkpoint_path):
        """Test get_checkpoint_model_config extracts model config."""
        config = get_checkpoint_model_config(checkpoint_path, "mae_config")

        assert config is not None
        assert "img_size" in config

    def test_checkpoint_has_config(self, checkpoint_path):
        """Test checkpoint_has_config correctly detects config."""
        assert checkpoint_has_config(checkpoint_path, "mae_config") is True
        assert checkpoint_has_config(checkpoint_path, "nonexistent") is False

    def test_load_mae_model(self, checkpoint_path):
        """Test load_mae_model utility function."""
        model = load_mae_model(checkpoint_path)

        # Check type by class name (avoids import path issues)
        assert type(model).__name__ == "MaskedAutoencoder"
        assert model.img_size == 32


class TestWeightConsistency:
    """Tests verifying weights are correctly saved and loaded."""

    def test_weights_match_after_reload(self):
        """Test model weights match after save/load cycle."""
        model = MaskedAutoencoder(img_size=32, patch_size=4, embed_dim=64)
        task = MAETask(model=model)

        # Store original weights
        original_weights = {
            k: v.clone() for k, v in task.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.ckpt"
            checkpoint = {
                "state_dict": task.state_dict(),
                "hyper_parameters": dict(task.hparams),
            }
            torch.save(checkpoint, ckpt_path)

            # Load model
            loaded = load_mae_from_checkpoint(str(ckpt_path))

            # Compare weights
            for name, param in loaded.named_parameters():
                original_key = f"model.{name}"
                assert original_key in original_weights
                assert torch.allclose(param, original_weights[original_key])

    def test_features_match_after_reload(self):
        """Test extracted features match after save/load cycle."""
        model = MaskedAutoencoder(img_size=32, patch_size=4, embed_dim=64)
        task = MAETask(model=model)

        # Extract features with original model
        x = torch.randn(4, 1, 32, 32)
        task.model.eval()
        with torch.no_grad():
            original_features = task.model.encode(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.ckpt"
            checkpoint = {
                "state_dict": task.state_dict(),
                "hyper_parameters": dict(task.hparams),
            }
            torch.save(checkpoint, ckpt_path)

            # Load and extract features
            extractor = MAEFeatureExtractor.from_checkpoint(
                str(ckpt_path), device="cpu"
            )
            loaded_features = extractor.extract(x)

            # Features should match exactly
            assert torch.allclose(original_features, loaded_features, atol=1e-6)
