"""Classification metrics utilities for evaluation and artifact logging.

Provides functions for computing F1 scores, accuracy, confusion matrices,
and logging them as MLFlow artifacts.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score


def compute_f1_scores(
    labels: Union[np.ndarray, torch.Tensor],
    predictions: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """Compute micro and macro F1 scores.

    Args:
        labels: Ground truth labels [N]
        predictions: Predicted labels [N]

    Returns:
        Dictionary with 'f1_micro' and 'f1_macro' keys
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)

    return {
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
    }


def compute_accuracy(
    labels: Union[np.ndarray, torch.Tensor],
    predictions: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute classification accuracy.

    Args:
        labels: Ground truth labels [N]
        predictions: Predicted labels [N]

    Returns:
        Accuracy as float
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    return float((labels == predictions).mean())


def compute_confusion_matrix(
    labels: Union[np.ndarray, torch.Tensor],
    predictions: Union[np.ndarray, torch.Tensor],
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        labels: Ground truth labels [N]
        predictions: Predicted labels [N]
        num_classes: Number of classes

    Returns:
        Confusion matrix as [num_classes, num_classes] array
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    return cm


def render_confusion_matrix_heatmap(
    confusion_mat: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> Image.Image:
    """Render confusion matrix using sklearn's ConfusionMatrixDisplay.

    Args:
        confusion_mat: Confusion matrix [num_classes, num_classes]
        class_names: List of class names/labels
        title: Plot title
        normalize: Whether to normalize by row (show percentages)

    Returns:
        PIL Image of the rendered confusion matrix
    """
    display_mat = confusion_mat.copy()
    if normalize:
        row_sums = confusion_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        display_mat = confusion_mat.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=display_mat,
        display_labels=class_names,
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f" if normalize else "d",
        colorbar=True,
    )

    ax.set_title(title)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)

    return img


def log_confusion_matrix_artifacts(
    logger,
    run_id: str,
    labels: Union[np.ndarray, torch.Tensor],
    predictions: Union[np.ndarray, torch.Tensor],
    num_classes: int,
    artifact_path: str = "confusion_matrices",
    prefix: str = "",
    metadata: Optional[Dict] = None,
) -> None:
    """Log confusion matrix artifacts (JSON + PNG) to MLFlow.

    Args:
        logger: PyTorch Lightning logger with MLFlow experiment
        run_id: MLFlow run ID
        labels: Ground truth labels [N]
        predictions: Predicted labels [N]
        num_classes: Number of classes
        artifact_path: Base artifact path in MLFlow
        prefix: Filename prefix (e.g., "test", "val_epoch_005")
        metadata: Optional metadata to include in JSON (e.g., F1 scores)
    """
    if not hasattr(logger, "experiment"):
        return

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    cm = compute_confusion_matrix(labels, predictions, num_classes)
    class_names = [str(i) for i in range(num_classes)]

    title = "Confusion Matrix"
    if prefix:
        title = f"Confusion Matrix - {prefix}"

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Save JSON
            json_filename = f"{prefix}.json" if prefix else "confusion_matrix.json"
            json_path = tmp_path / json_filename
            data = {
                "matrix": cm.tolist(),
                "class_names": class_names,
                "total_samples": int(cm.sum()),
            }
            if metadata:
                data.update(metadata)
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

            # Save PNG
            png_filename = f"{prefix}.png" if prefix else "confusion_matrix.png"
            png_path = tmp_path / png_filename
            img = render_confusion_matrix_heatmap(cm, class_names, title=title)
            img.save(png_path)

            # Log to MLFlow
            logger.experiment.log_artifact(
                run_id=run_id,
                local_path=str(json_path),
                artifact_path=artifact_path,
            )
            logger.experiment.log_artifact(
                run_id=run_id,
                local_path=str(png_path),
                artifact_path=artifact_path,
            )

    except Exception as e:
        print(f"Warning: Could not log confusion matrix artifacts: {e}")
