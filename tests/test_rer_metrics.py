"""Tests for RER metrics utilities."""

import pytest
import torch

from src.util.rer_metrics import (
    compute_rer,
    compute_dataset_chi,
    compute_class_chi,
    compute_chi_0,
    compute_chi_rand,
    estimate_noise_rate,
    compute_mislabel_threshold,
    detect_mislabels,
    compute_rer_statistics,
)


class TestComputeRER:
    """Tests for the compute_rer function."""

    def test_basic_rer_computation(self):
        """Test basic RER computation matches paper formula."""
        # Create simple test case
        # Sample 0: class 0, errors [1.0, 2.0, 3.0]
        # RER = 1.0 / min(2.0, 3.0) = 1.0 / 2.0 = 0.5
        all_errors = torch.tensor([
            [1.0, 2.0, 3.0],  # Sample 0
            [4.0, 1.0, 2.0],  # Sample 1 (class 1)
        ])
        labels = torch.tensor([0, 1])

        rer = compute_rer(all_errors, labels)

        assert rer.shape == (2,)
        assert torch.allclose(rer[0], torch.tensor(0.5), rtol=1e-5)
        assert torch.allclose(rer[1], torch.tensor(0.5), rtol=1e-5)

    def test_rer_shape(self):
        """Test RER output shape."""
        batch_size = 32
        num_classes = 10
        all_errors = torch.rand(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))

        rer = compute_rer(all_errors, labels)

        assert rer.shape == (batch_size,)

    def test_rer_positive(self):
        """Test RER is always positive."""
        batch_size = 100
        num_classes = 10
        all_errors = torch.rand(batch_size, num_classes) + 0.1  # Ensure positive
        labels = torch.randint(0, num_classes, (batch_size,))

        rer = compute_rer(all_errors, labels)

        assert (rer > 0).all()

    def test_rer_perfect_classifier(self):
        """Test RER for a perfect classifier (own error = 0)."""
        # When own class error is 0, RER should be 0
        all_errors = torch.tensor([
            [0.0, 1.0, 1.0],  # Sample from class 0
        ])
        labels = torch.tensor([0])

        rer = compute_rer(all_errors, labels)

        assert rer[0] == 0.0

    def test_rer_confused_sample(self):
        """Test RER for a confused sample (own error > min other)."""
        # When own class error > min other error, RER > 1
        all_errors = torch.tensor([
            [2.0, 1.0, 3.0],  # Sample from class 0, but class 1 has lower error
        ])
        labels = torch.tensor([0])

        rer = compute_rer(all_errors, labels)

        assert rer[0] > 1.0


class TestDatasetChi:
    """Tests for chi computation functions."""

    def test_compute_dataset_chi(self):
        """Test dataset chi is mean of RER values."""
        rer_values = torch.tensor([0.5, 1.0, 1.5, 2.0])

        chi = compute_dataset_chi(rer_values)

        assert chi == pytest.approx(1.25)

    def test_compute_class_chi(self):
        """Test per-class chi computation."""
        rer_values = torch.tensor([0.5, 1.0, 1.5, 2.0])
        labels = torch.tensor([0, 0, 1, 1])

        class_chi = compute_class_chi(rer_values, labels)

        assert 0 in class_chi
        assert 1 in class_chi
        assert class_chi[0] == pytest.approx(0.75)  # (0.5 + 1.0) / 2
        assert class_chi[1] == pytest.approx(1.75)  # (1.5 + 2.0) / 2


class TestNoiseEstimation:
    """Tests for noise rate estimation functions."""

    def test_compute_chi_0(self):
        """Test chi_0 computation."""
        # Simple case: 2 samples, 3 classes
        all_errors = torch.tensor([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ])
        labels = torch.tensor([0, 0])

        chi_0 = compute_chi_0(all_errors, labels)

        # Own error = 1.0, avg other = (2.0 + 3.0) / 2 = 2.5
        # chi_0 = mean(1.0 / 2.5) = 0.4
        assert chi_0 == pytest.approx(0.4, rel=1e-3)

    def test_compute_chi_rand(self):
        """Test chi_rand computation."""
        all_errors = torch.tensor([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ])
        labels = torch.tensor([0, 0])

        chi_rand = compute_chi_rand(all_errors, labels)

        # Best error = 1.0, avg other = 2.5
        # chi_rand = mean(1.0 / 2.5) = 0.4
        assert chi_rand == pytest.approx(0.4, rel=1e-3)

    def test_estimate_noise_rate_clean(self):
        """Test noise rate estimation on clean data (chi_0 ≈ chi_rand)."""
        chi_0 = 0.5
        chi_rand = 0.5

        eta = estimate_noise_rate(chi_0, chi_rand)

        # When chi_0 = chi_rand, eta should be ~0
        assert eta == pytest.approx(0.0, abs=1e-5)

    def test_estimate_noise_rate_noisy(self):
        """Test noise rate estimation on noisy data."""
        chi_0 = 0.7
        chi_rand = 0.5

        eta = estimate_noise_rate(chi_0, chi_rand)

        # eta = (0.7 - 0.5) / (1 - 0.5) = 0.4
        assert eta == pytest.approx(0.4, rel=1e-3)


class TestMislabelDetection:
    """Tests for mislabel detection functions."""

    def test_compute_mislabel_threshold(self):
        """Test mislabel threshold computation."""
        chi_0 = 0.5
        eta_est = 0.1

        threshold = compute_mislabel_threshold(chi_0, eta_est)

        # Should be positive
        assert threshold > 0

    def test_detect_mislabels_with_threshold(self):
        """Test mislabel detection with explicit threshold."""
        rer_values = torch.tensor([0.5, 1.5, 2.5, 0.8])
        threshold = 1.0

        mislabels = detect_mislabels(rer_values, threshold=threshold)

        assert mislabels.dtype == torch.bool
        assert mislabels.sum() == 2  # Two values > 1.0

    def test_detect_mislabels_auto_threshold(self):
        """Test mislabel detection with automatic threshold."""
        rer_values = torch.tensor([0.5, 1.5, 2.5, 0.8])
        chi_0 = 0.5
        eta_est = 0.1

        mislabels = detect_mislabels(rer_values, chi_0=chi_0, eta_est=eta_est)

        assert mislabels.dtype == torch.bool

    def test_detect_mislabels_requires_params(self):
        """Test that detect_mislabels requires either threshold or chi_0/eta_est."""
        rer_values = torch.tensor([0.5, 1.5, 2.5])

        with pytest.raises(ValueError):
            detect_mislabels(rer_values)


class TestRERStatistics:
    """Tests for compute_rer_statistics function."""

    def test_statistics_keys(self):
        """Test that statistics dict has expected keys."""
        rer_values = torch.tensor([0.5, 1.0, 1.5, 2.0])
        labels = torch.tensor([0, 0, 1, 1])

        stats = compute_rer_statistics(rer_values, labels)

        assert "chi" in stats
        assert "rer_mean" in stats
        assert "rer_std" in stats
        assert "rer_min" in stats
        assert "rer_max" in stats
        assert "rer_median" in stats
        assert "samples_above_1" in stats
        assert "frac_above_1" in stats

    def test_statistics_values(self):
        """Test that statistics values are correct."""
        rer_values = torch.tensor([0.5, 1.0, 1.5, 2.0])
        labels = torch.tensor([0, 0, 1, 1])

        stats = compute_rer_statistics(rer_values, labels)

        assert stats["chi"] == pytest.approx(1.25)
        assert stats["rer_mean"] == pytest.approx(1.25)
        assert stats["rer_min"] == pytest.approx(0.5)
        assert stats["rer_max"] == pytest.approx(2.0)
        assert stats["samples_above_1"] == 2
        assert stats["frac_above_1"] == pytest.approx(0.5)

    def test_statistics_per_class(self):
        """Test per-class statistics are included."""
        rer_values = torch.tensor([0.5, 1.0, 1.5, 2.0])
        labels = torch.tensor([0, 0, 1, 1])

        stats = compute_rer_statistics(rer_values, labels)

        assert "chi_class_0" in stats
        assert "chi_class_1" in stats
        assert "hardest_class" in stats
        assert "easiest_class" in stats
