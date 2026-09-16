"""MLFlow utilities for experiment management."""

from pathlib import Path
import tempfile
from typing import Optional

import PIL.Image


def resolve_experiment_path(
    experiment_name: str,
    run_id: str,
    base_dir: str = "/data/thallybu/experiments",
) -> Path:
    """Resolve the full path for an experiment run.

    Args:
        experiment_name: Name of the MLFlow experiment
        run_id: Run identifier (usually timestamp-based)
        base_dir: Base directory for experiments

    Returns:
        Path to the experiment run directory
    """
    return Path(base_dir) / experiment_name / run_id


def log_image_artifact(
    logger,
    image: PIL.Image,
    filename: str,
    run_id: Optional[str] = None,
) -> None:
    """Log an image artifact to MLFlow.

    Args:
        logger: MLFlow logger instance
        image: PIL Image object to log
        filename: Name for the artifact file in MLFlow
        run_id: Run ID (optional, uses current run if not provided)
    """
    if logger is None or not hasattr(logger, "experiment"):
        return

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / filename
            artifact_path.parent.mkdir(parents=True, exist_ok=True)

            image.save(artifact_path)

            if run_id is None and hasattr(logger, "run_id"):
                run_id = logger.run_id

            if run_id:
                logger.experiment.log_artifact(
                    run_id=run_id,
                    local_path=str(artifact_path),
                    artifact_path=filename,
                )
    except Exception as e:
        print(f"Error logging image artifact: {e}")
