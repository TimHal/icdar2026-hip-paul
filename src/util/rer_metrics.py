"""Reconstruction Error Ratio (RER) metrics and utilities.

Implements the RER framework from Marks et al. (2024) for:
- Dataset difficulty estimation (chi)
- Noise rate estimation
- Mislabel detection
"""

from __future__ import annotations

import torch
from typing import Optional


def compute_rer(
    all_errors: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute Reconstruction Error Ratio (RER) for samples.

    RER(x^c) = Delta^c(x^c) / min_{c' != c} Delta^{c'}(x^c)

    Where Delta^c(x) is the reconstruction error for sample x using class c's autoencoder.

    Args:
        all_errors: Reconstruction errors from all autoencoders [batch_size, num_classes]
        labels: Class labels [batch_size]
        eps: Small value to prevent division by zero

    Returns:
        RER values [batch_size]
    """
    batch_size = all_errors.shape[0]
    num_classes = all_errors.shape[1]
    device = all_errors.device

    # Get own class errors
    own_errors = all_errors[torch.arange(batch_size, device=device), labels.long()]

    # For min of other errors, set own class error to inf
    errors_masked = all_errors.clone()
    errors_masked[torch.arange(batch_size, device=device), labels.long()] = float("inf")
    min_other_errors = errors_masked.min(dim=1).values

    rer = own_errors / (min_other_errors + eps)
    return rer


def compute_dataset_chi(rer_values: torch.Tensor) -> float:
    """Compute chi (mean RER) for dataset difficulty estimation.

    Chi is the average RER across all samples, serving as a measure
    of classification difficulty (Equation 4 in paper).

    Args:
        rer_values: RER values for all samples [N]

    Returns:
        Chi value (mean RER)
    """
    return rer_values.mean().item()


def compute_class_chi(
    rer_values: torch.Tensor,
    labels: torch.Tensor,
) -> dict[int, float]:
    """Compute chi per class.

    Args:
        rer_values: RER values for all samples [N]
        labels: Class labels [N]

    Returns:
        Dictionary mapping class index to class-specific chi
    """
    class_chi = {}
    unique_classes = labels.unique()

    for c in unique_classes:
        mask = labels == c
        if mask.sum() > 0:
            class_chi[c.item()] = rer_values[mask].mean().item()

    return class_chi


def compute_chi_0(
    all_errors: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute chi_0 for noise rate estimation (Equation 7 in paper).

    chi_0 = E[Delta^c(x^c) / Delta_rand(x^c)]

    Where Delta_rand is the average reconstruction error from a random class.

    Args:
        all_errors: Reconstruction errors [N, num_classes]
        labels: Class labels [N]
        eps: Small value to prevent division by zero

    Returns:
        chi_0 value
    """
    batch_size = all_errors.shape[0]
    num_classes = all_errors.shape[1]
    device = all_errors.device

    # Get own class errors
    own_errors = all_errors[torch.arange(batch_size, device=device), labels.long()]

    # Compute average error from other classes (Delta_rand)
    # For each sample, average over all classes except own
    rand_errors = []
    for i in range(batch_size):
        c = labels[i].long()
        other_errors = torch.cat([all_errors[i, :c], all_errors[i, c+1:]])
        rand_errors.append(other_errors.mean())
    rand_errors = torch.stack(rand_errors)

    chi_0 = (own_errors / (rand_errors + eps)).mean().item()
    return chi_0


def compute_chi_rand(
    all_errors: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute chi_rand for noise rate estimation.

    chi_rand = E[Delta_best(x) / Delta_rand(x)]

    Where Delta_best is the minimum reconstruction error across all classes.

    Args:
        all_errors: Reconstruction errors [N, num_classes]
        labels: Class labels [N]
        eps: Small value to prevent division by zero

    Returns:
        chi_rand value
    """
    batch_size = all_errors.shape[0]
    num_classes = all_errors.shape[1]

    # Best error (min across all classes)
    best_errors = all_errors.min(dim=1).values

    # Average error from other classes
    rand_errors = []
    for i in range(batch_size):
        c = labels[i].long()
        other_errors = torch.cat([all_errors[i, :c], all_errors[i, c+1:]])
        rand_errors.append(other_errors.mean())
    rand_errors = torch.stack(rand_errors)

    chi_rand = (best_errors / (rand_errors + eps)).mean().item()
    return chi_rand


def estimate_noise_rate(
    chi_0: float,
    chi_rand: float,
    eps: float = 1e-8,
) -> float:
    """Estimate noise rate from RER metrics (Equation 8 in paper).

    eta ≈ (chi_0 - chi_rand) / (1 - chi_rand)

    Args:
        chi_0: chi_0 value from compute_chi_0()
        chi_rand: chi_rand value from compute_chi_rand()
        eps: Small value to prevent division by zero

    Returns:
        Estimated noise rate eta
    """
    return (chi_0 - chi_rand) / (1 - chi_rand + eps)


def compute_mislabel_threshold(
    chi_0: float,
    eta_est: float,
    gamma4: float = 1.01,
    gamma5: float = 1.5,
    gamma6: float = 13.8,
) -> float:
    """Compute mislabel detection threshold (Equation 9 in paper).

    chi* = gamma4 * chi_0^(-gamma5 / (1 + gamma6 * eta))

    Args:
        chi_0: chi_0 value
        eta_est: Estimated noise rate
        gamma4, gamma5, gamma6: Ansatz parameters from paper

    Returns:
        Threshold chi* for mislabel detection
    """
    exponent = -gamma5 / (1 + gamma6 * eta_est)
    return gamma4 * (chi_0 ** exponent)


def detect_mislabels(
    rer_values: torch.Tensor,
    threshold: Optional[float] = None,
    chi_0: Optional[float] = None,
    eta_est: Optional[float] = None,
) -> torch.Tensor:
    """Detect potentially mislabeled samples.

    Samples with RER > threshold are flagged as potentially mislabeled.

    Args:
        rer_values: RER values for all samples [N]
        threshold: Manual threshold (if not provided, computed from chi_0 and eta_est)
        chi_0: chi_0 value for automatic threshold computation
        eta_est: Estimated noise rate for automatic threshold computation

    Returns:
        Boolean mask indicating potentially mislabeled samples [N]
    """
    if threshold is None:
        if chi_0 is None or eta_est is None:
            raise ValueError("Either threshold or (chi_0, eta_est) must be provided")
        threshold = compute_mislabel_threshold(chi_0, eta_est)

    return rer_values > threshold


def compute_rer_statistics(
    rer_values: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """Compute comprehensive RER statistics.

    Args:
        rer_values: RER values for all samples [N]
        labels: Class labels [N]

    Returns:
        Dictionary with RER statistics
    """
    stats = {
        "chi": compute_dataset_chi(rer_values),
        "rer_mean": rer_values.mean().item(),
        "rer_std": rer_values.std().item(),
        "rer_min": rer_values.min().item(),
        "rer_max": rer_values.max().item(),
        "rer_median": rer_values.median().item(),
        "samples_above_1": (rer_values > 1.0).sum().item(),
        "frac_above_1": (rer_values > 1.0).float().mean().item(),
    }

    # Per-class statistics
    class_chi = compute_class_chi(rer_values, labels)
    for c, chi_c in class_chi.items():
        stats[f"chi_class_{c}"] = chi_c

    # Identify hardest and easiest classes
    if class_chi:
        hardest_class = max(class_chi, key=class_chi.get)
        easiest_class = min(class_chi, key=class_chi.get)
        stats["hardest_class"] = hardest_class
        stats["easiest_class"] = easiest_class
        stats["hardest_class_chi"] = class_chi[hardest_class]
        stats["easiest_class_chi"] = class_chi[easiest_class]

    return stats
