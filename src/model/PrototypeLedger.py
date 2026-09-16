"""Weak label ledger for self-supervised class-wise autoencoder training.

The ledger tracks sample-to-class assignments based on reconstruction errors,
enabling fully unsupervised training with prototype-guided initialization.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


@dataclass
class SampleAssignment:
    """Tracks the assignment state of a single sample."""

    sample_id: int
    assigned_class: int = -1  # -1 = unassigned
    reconstruction_error: float = float("inf")
    rer: float = float("inf")  # Reconstruction Error Ratio
    confidence: float = 0.0  # Inverse RER or similar
    epoch_assigned: int = -1  # When assignment was made
    is_prototype: bool = False  # Whether this is a prototype sample


class PrototypeLedger:
    """Tracks weak label assignments for all samples in self-supervised training.

    The ledger maintains a mapping from sample IDs to their current class
    assignments, reconstruction errors, and confidence scores. Assignments
    are updated based on reconstruction error from class-wise autoencoders.

    Args:
        num_classes: Number of classes (autoencoders)
        rer_threshold: Maximum RER for a sample to be assigned (default: 2.0)
        min_confidence: Minimum confidence for assignment (1/rer_threshold)
    """

    def __init__(
        self,
        num_classes: int,
        rer_threshold: float = 2.0,
        device: str = "cpu",
    ):
        self.num_classes = num_classes
        self.rer_threshold = rer_threshold
        self.device = device

        # Main storage: sample_id -> SampleAssignment
        self.assignments: dict[int, SampleAssignment] = {}

        # Prototype sample IDs (fixed assignments)
        self.prototype_ids: set[int] = set()

        # Statistics cache (updated after each epoch)
        self._stats_cache: Optional[dict] = None

    def register_prototype(
        self,
        sample_id: int,
        class_idx: int,
        reconstruction_error: float = 0.0,
    ) -> None:
        """Register a prototype sample with fixed class assignment.

        Prototypes are never reassigned during training.

        Args:
            sample_id: Sample ID to register
            class_idx: Class index for this prototype
            reconstruction_error: Initial reconstruction error
        """
        self.assignments[sample_id] = SampleAssignment(
            sample_id=sample_id,
            assigned_class=class_idx,
            reconstruction_error=reconstruction_error,
            rer=0.0,  # Prototypes have perfect RER by definition
            confidence=1.0,
            epoch_assigned=0,
            is_prototype=True,
        )
        self.prototype_ids.add(sample_id)
        self._stats_cache = None

    def update(
        self,
        sample_ids: Tensor,
        all_errors: Tensor,
        epoch: int,
        force_assignment: bool = False,
    ) -> Tensor:
        """Update assignments based on reconstruction errors.

        For each sample, assigns to the class with minimum reconstruction
        error if the RER is below threshold.

        Args:
            sample_ids: [batch] tensor of sample IDs
            all_errors: [batch, num_classes] reconstruction errors
            epoch: Current training epoch
            force_assignment: If True, assign even if RER > threshold

        Returns:
            assigned_classes: [batch] tensor of class assignments (-1 = unassigned)
        """
        batch_size = sample_ids.shape[0]
        device = all_errors.device

        # Find minimum error class for each sample
        min_errors, min_classes = all_errors.min(dim=1)

        # Compute RER: own_error / min_other_error
        # Create mask for own class
        batch_indices = torch.arange(batch_size, device=device)
        other_errors = all_errors.clone()
        other_errors[batch_indices, min_classes] = float("inf")
        min_other_errors = other_errors.min(dim=1).values

        # Handle case where there's only one class
        min_other_errors = torch.where(
            min_other_errors == float("inf"),
            torch.ones_like(min_other_errors),
            min_other_errors,
        )

        rer_values = min_errors / (min_other_errors + 1e-8)
        confidences = 1.0 / (rer_values + 1e-8)

        # Determine which samples meet threshold
        if force_assignment:
            meets_threshold = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            meets_threshold = rer_values < self.rer_threshold

        # Build output tensor
        assigned_classes = torch.full(
            (batch_size,), -1, dtype=torch.long, device=device
        )
        assigned_classes[meets_threshold] = min_classes[meets_threshold]

        # Update internal state
        for i in range(batch_size):
            sid = sample_ids[i].item()

            # Skip prototypes - they have fixed assignments
            if sid in self.prototype_ids:
                assigned_classes[i] = self.assignments[sid].assigned_class
                continue

            # Update or create assignment
            if meets_threshold[i]:
                self.assignments[sid] = SampleAssignment(
                    sample_id=sid,
                    assigned_class=min_classes[i].item(),
                    reconstruction_error=min_errors[i].item(),
                    rer=rer_values[i].item(),
                    confidence=confidences[i].item(),
                    epoch_assigned=epoch,
                    is_prototype=False,
                )
            elif sid not in self.assignments:
                # Create unassigned entry
                self.assignments[sid] = SampleAssignment(
                    sample_id=sid,
                    assigned_class=-1,
                    reconstruction_error=min_errors[i].item(),
                    rer=rer_values[i].item(),
                    confidence=confidences[i].item(),
                    epoch_assigned=-1,
                    is_prototype=False,
                )

        self._stats_cache = None
        return assigned_classes

    def get_top_samples_per_class(
        self,
        n: int,
        exclude_prototypes: bool = False,
    ) -> dict[int, list[int]]:
        """Get top N samples per class by lowest RER.

        Args:
            n: Number of samples to return per class
            exclude_prototypes: If True, don't include prototype samples

        Returns:
            {class_idx: [sample_ids]} dictionary
        """
        # Group assignments by class
        class_samples: dict[int, list[tuple[int, float]]] = {
            c: [] for c in range(self.num_classes)
        }

        for sid, assignment in self.assignments.items():
            if assignment.assigned_class < 0:
                continue
            if exclude_prototypes and assignment.is_prototype:
                continue

            class_samples[assignment.assigned_class].append(
                (sid, assignment.rer)
            )

        # Sort by RER and take top N
        result = {}
        for class_idx, samples in class_samples.items():
            sorted_samples = sorted(samples, key=lambda x: x[1])
            result[class_idx] = [sid for sid, _ in sorted_samples[:n]]

        return result

    def get_samples_for_class(self, class_idx: int) -> list[int]:
        """Get all sample IDs assigned to a class."""
        return [
            sid for sid, a in self.assignments.items()
            if a.assigned_class == class_idx
        ]

    def get_class_assignments_tensor(self, sample_ids: Tensor) -> Tensor:
        """Get class assignments for given sample IDs.

        Args:
            sample_ids: [N] tensor of sample IDs

        Returns:
            [N] tensor of class assignments (-1 = unassigned)
        """
        device = sample_ids.device
        result = torch.full((len(sample_ids),), -1, dtype=torch.long, device=device)

        for i, sid in enumerate(sample_ids.tolist()):
            if sid in self.assignments:
                result[i] = self.assignments[sid].assigned_class

        return result

    def get_all_assignments(self) -> tuple[Tensor, Tensor]:
        """Get all assignments as tensors.

        Returns:
            sample_ids: [N] tensor
            assigned_classes: [N] tensor
        """
        if not self.assignments:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        sids = []
        classes = []
        for sid, assignment in self.assignments.items():
            sids.append(sid)
            classes.append(assignment.assigned_class)

        return torch.tensor(sids, dtype=torch.long), torch.tensor(classes, dtype=torch.long)

    def compute_assignment_stats(self) -> dict:
        """Compute statistics about current assignments.

        Returns:
            Dictionary with assignment statistics
        """
        if self._stats_cache is not None:
            return self._stats_cache

        stats = {
            "total_samples": len(self.assignments),
            "assigned_samples": 0,
            "unassigned_samples": 0,
            "prototype_count": len(self.prototype_ids),
        }

        # Per-class stats
        class_counts = {c: 0 for c in range(self.num_classes)}
        class_rer_sum = {c: 0.0 for c in range(self.num_classes)}

        all_rer = []

        for assignment in self.assignments.values():
            if assignment.assigned_class >= 0:
                stats["assigned_samples"] += 1
                class_counts[assignment.assigned_class] += 1
                class_rer_sum[assignment.assigned_class] += assignment.rer
                all_rer.append(assignment.rer)
            else:
                stats["unassigned_samples"] += 1

        # Add per-class stats
        for c in range(self.num_classes):
            stats[f"samples_class_{c}"] = class_counts[c]
            if class_counts[c] > 0:
                stats[f"avg_rer_class_{c}"] = class_rer_sum[c] / class_counts[c]
            else:
                stats[f"avg_rer_class_{c}"] = 0.0

        # Overall RER stats
        if all_rer:
            rer_tensor = torch.tensor(all_rer)
            stats["rer_mean"] = rer_tensor.mean().item()
            stats["rer_std"] = rer_tensor.std().item()
            stats["rer_min"] = rer_tensor.min().item()
            stats["rer_max"] = rer_tensor.max().item()
        else:
            stats["rer_mean"] = 0.0
            stats["rer_std"] = 0.0
            stats["rer_min"] = 0.0
            stats["rer_max"] = 0.0

        self._stats_cache = stats
        return stats

    def get_assignment_agreement(self, true_labels: Tensor, sample_ids: Tensor) -> float:
        """Compute agreement between weak assignments and true labels.

        Args:
            true_labels: [N] tensor of true class labels
            sample_ids: [N] tensor of sample IDs

        Returns:
            Fraction of assigned samples matching true labels
        """
        assigned = self.get_class_assignments_tensor(sample_ids)

        # Only compare assigned samples
        mask = assigned >= 0
        if not mask.any():
            return 0.0

        matches = (assigned[mask] == true_labels[mask]).float()
        return matches.mean().item()

    def reset(self, keep_prototypes: bool = True) -> None:
        """Reset all assignments.

        Args:
            keep_prototypes: If True, keep prototype assignments
        """
        if keep_prototypes:
            self.assignments = {
                sid: a for sid, a in self.assignments.items()
                if a.is_prototype
            }
        else:
            self.assignments.clear()
            self.prototype_ids.clear()

        self._stats_cache = None

    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            "num_classes": self.num_classes,
            "rer_threshold": self.rer_threshold,
            "assignments": {
                sid: {
                    "sample_id": a.sample_id,
                    "assigned_class": a.assigned_class,
                    "reconstruction_error": a.reconstruction_error,
                    "rer": a.rer,
                    "confidence": a.confidence,
                    "epoch_assigned": a.epoch_assigned,
                    "is_prototype": a.is_prototype,
                }
                for sid, a in self.assignments.items()
            },
            "prototype_ids": list(self.prototype_ids),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self.num_classes = state_dict["num_classes"]
        self.rer_threshold = state_dict["rer_threshold"]
        self.prototype_ids = set(state_dict["prototype_ids"])

        self.assignments = {}
        for sid, data in state_dict["assignments"].items():
            self.assignments[int(sid)] = SampleAssignment(**data)

        self._stats_cache = None

    def compute_candidate_rer(
        self,
        sample_ids: Tensor,
        all_errors: Tensor,
    ) -> dict[int, tuple[int, float, float]]:
        """Compute RER for samples without modifying assignments.

        Args:
            sample_ids: [batch] tensor of sample IDs
            all_errors: [batch, num_classes] reconstruction errors

        Returns:
            {sample_id: (best_class, rer_value, reconstruction_error)}
        """
        batch_size = sample_ids.shape[0]
        device = all_errors.device

        # Find minimum error class for each sample
        min_errors, min_classes = all_errors.min(dim=1)

        # Compute RER: own_error / min_other_error
        batch_indices = torch.arange(batch_size, device=device)
        other_errors = all_errors.clone()
        other_errors[batch_indices, min_classes] = float("inf")
        min_other_errors = other_errors.min(dim=1).values

        # Handle case where there's only one class
        min_other_errors = torch.where(
            min_other_errors == float("inf"),
            torch.ones_like(min_other_errors),
            min_other_errors,
        )

        rer_values = min_errors / (min_other_errors + 1e-8)

        # Build result dictionary
        result = {}
        for i in range(batch_size):
            sid = sample_ids[i].item()
            result[sid] = (
                min_classes[i].item(),
                rer_values[i].item(),
                min_errors[i].item(),
            )

        return result

    def assign_top_k_per_class(
        self,
        candidates: dict[int, tuple[int, float, float]],
        k: int,
        epoch: int,
    ) -> int:
        """Assign K best (lowest RER) unassigned samples per class.

        Args:
            candidates: {sample_id: (best_class, rer, error)} from compute_candidate_rer
            k: Number of samples to assign per class
            epoch: Current epoch for tracking

        Returns:
            Number of new assignments made.
        """
        # Group unassigned candidates by their best class
        class_candidates: dict[int, list[tuple[int, float, float]]] = {
            c: [] for c in range(self.num_classes)
        }

        for sid, (best_class, rer, error) in candidates.items():
            # Skip already assigned samples and prototypes
            if sid in self.assignments and self.assignments[sid].assigned_class >= 0:
                continue
            class_candidates[best_class].append((sid, rer, error))

        # Sort each class by RER and take top K
        new_assignments = 0
        for class_idx, samples in class_candidates.items():
            sorted_samples = sorted(samples, key=lambda x: x[1])  # Sort by RER
            for sid, rer, error in sorted_samples[:k]:
                confidence = 1.0 / (rer + 1e-8)
                self.assignments[sid] = SampleAssignment(
                    sample_id=sid,
                    assigned_class=class_idx,
                    reconstruction_error=error,
                    rer=rer,
                    confidence=confidence,
                    epoch_assigned=epoch,
                    is_prototype=False,
                )
                new_assignments += 1

        self._stats_cache = None
        return new_assignments

    def assign_below_threshold(
        self,
        candidates: dict[int, tuple[int, float, float]],
        threshold: float,
        epoch: int,
    ) -> int:
        """Assign all unassigned samples with RER below threshold.

        Args:
            candidates: {sample_id: (best_class, rer, error)} from compute_candidate_rer
            threshold: Maximum RER for assignment
            epoch: Current epoch for tracking

        Returns:
            Number of new assignments made.
        """
        new_assignments = 0
        for sid, (best_class, rer, error) in candidates.items():
            # Skip already assigned samples and prototypes
            if sid in self.assignments and self.assignments[sid].assigned_class >= 0:
                continue

            if rer < threshold:
                confidence = 1.0 / (rer + 1e-8)
                self.assignments[sid] = SampleAssignment(
                    sample_id=sid,
                    assigned_class=best_class,
                    reconstruction_error=error,
                    rer=rer,
                    confidence=confidence,
                    epoch_assigned=epoch,
                    is_prototype=False,
                )
                new_assignments += 1

        self._stats_cache = None
        return new_assignments

    def update_rer_values(
        self,
        candidates: dict[int, tuple[int, float, float]],
    ) -> None:
        """Update RER values for assigned samples based on new errors.

        Does not change assignments, only updates the stored RER/error values.

        Args:
            candidates: {sample_id: (best_class, rer, error)} from compute_candidate_rer
        """
        for sid, (_, rer, error) in candidates.items():
            if sid in self.assignments and not self.assignments[sid].is_prototype:
                self.assignments[sid].rer = rer
                self.assignments[sid].reconstruction_error = error
                self.assignments[sid].confidence = 1.0 / (rer + 1e-8)

        self._stats_cache = None

    def prune_worst_k_per_class(self, k: int) -> int:
        """Unassign K samples with highest RER per class.

        Never prunes prototypes.

        Args:
            k: Number of worst samples to remove per class

        Returns:
            Number of samples pruned.
        """
        # Group non-prototype assigned samples by class
        class_samples: dict[int, list[tuple[int, float]]] = {
            c: [] for c in range(self.num_classes)
        }

        for sid, assignment in self.assignments.items():
            if assignment.is_prototype:
                continue
            if assignment.assigned_class < 0:
                continue
            class_samples[assignment.assigned_class].append((sid, assignment.rer))

        # Sort by RER descending and prune worst K
        pruned = 0
        for class_idx, samples in class_samples.items():
            sorted_samples = sorted(samples, key=lambda x: x[1], reverse=True)
            for sid, _ in sorted_samples[:k]:
                self.assignments[sid].assigned_class = -1
                self.assignments[sid].epoch_assigned = -1
                pruned += 1

        self._stats_cache = None
        return pruned

    def prune_above_threshold(self, threshold: float) -> int:
        """Unassign all non-prototype samples with RER > threshold.

        Args:
            threshold: Maximum allowed RER

        Returns:
            Number of samples pruned.
        """
        pruned = 0
        for sid, assignment in self.assignments.items():
            if assignment.is_prototype:
                continue
            if assignment.assigned_class < 0:
                continue
            if assignment.rer > threshold:
                assignment.assigned_class = -1
                assignment.epoch_assigned = -1
                pruned += 1

        self._stats_cache = None
        return pruned

    def to_json_dict(self) -> dict:
        """Export ledger state as JSON-serializable dictionary.

        Returns:
            Dictionary with assignments grouped by class and metadata.
        """
        # Group by class
        class_assignments: dict[int, list[dict]] = {
            c: [] for c in range(self.num_classes)
        }
        unassigned: list[dict] = []

        for sid, a in self.assignments.items():
            entry = {
                "sample_id": a.sample_id,
                "rer": round(a.rer, 4),
                "reconstruction_error": round(a.reconstruction_error, 6),
                "confidence": round(a.confidence, 4),
                "epoch_assigned": a.epoch_assigned,
                "is_prototype": a.is_prototype,
            }
            if a.assigned_class >= 0:
                class_assignments[a.assigned_class].append(entry)
            else:
                unassigned.append(entry)

        # Sort each class by RER
        for c in class_assignments:
            class_assignments[c].sort(key=lambda x: x["rer"])

        # Compute stats
        stats = self.compute_assignment_stats()

        return {
            "num_classes": self.num_classes,
            "rer_threshold": self.rer_threshold,
            "total_samples": stats["total_samples"],
            "assigned_samples": stats["assigned_samples"],
            "unassigned_samples": stats["unassigned_samples"],
            "prototype_count": stats["prototype_count"],
            "per_class_stats": {
                c: {
                    "count": stats.get(f"samples_class_{c}", 0),
                    "avg_rer": round(stats.get(f"avg_rer_class_{c}", 0.0), 4),
                }
                for c in range(self.num_classes)
            },
            "assignments_by_class": {
                str(c): assignments for c, assignments in class_assignments.items()
            },
            "unassigned": unassigned,
        }
