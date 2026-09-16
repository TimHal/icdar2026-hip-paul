"""MLFlow config saving callback for experiment reproducibility."""

import tempfile
from pathlib import Path

import lightning as L
from lightning.pytorch.cli import SaveConfigCallback


class SaveMLFlowConfigCallback(SaveConfigCallback):
    """Saves experiment configuration to MLFlow as an artifact.

    This callback extends Lightning's SaveConfigCallback to save the
    full experiment configuration (including model, data, trainer settings)
    to MLFlow for reproducibility.
    """

    def __init__(
        self,
        parser,
        config,
        config_filename="config.yaml",
        overwrite=False,
        multifile=False,
    ):
        """Initialize the callback.

        Args:
            parser: LightningArgumentParser instance
            config: Parsed configuration namespace
            config_filename: Name for the saved config file
            overwrite: Whether to overwrite existing config
            multifile: Whether to save as multiple files
        """
        super().__init__(
            parser,
            config,
            config_filename,
            overwrite,
            multifile,
            save_to_log_dir=False,
        )

    def save_config(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        stage: str,
    ) -> None:
        """Save configuration to MLFlow.

        Args:
            trainer: Lightning Trainer instance
            pl_module: Lightning Module being trained
            stage: Current training stage (fit, validate, test)
        """
        if not trainer.is_global_zero:
            return

        # Only save if we have an MLFlow logger
        if trainer.logger is None:
            return

        # Check if logger has MLFlow-style experiment/artifact logging
        if not hasattr(trainer.logger, "experiment") or not hasattr(
            trainer.logger.experiment, "log_artifact"
        ):
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            self.parser.save(
                self.config,
                config_path,
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )
            trainer.logger.experiment.log_artifact(
                local_path=str(config_path),
                run_id=trainer.logger.run_id,
            )
