"""ClasswiseAE Research CLI - Command Line Interface for RER experiments.

This CLI provides a flexible interface for running class-wise autoencoder
experiments with:
- Automatic checkpoint saving and loading
- MLFlow experiment tracking
- Configuration-driven experiments
- RER computation and logging
"""

import sys
from datetime import datetime
from typing import Optional

from lightning.pytorch.cli import LightningCLI
import torch

from core.callbacks.SaveConfigCallback import SaveMLFlowConfigCallback

# Constants
DEFAULT_EXPERIMENTS_ROOT = "/data/thallybu/experiments"
CUSTOM_COMMANDS = frozenset(["analyze_rer", "detect_mislabels", "study"])
RUN_ID_TIMESTAMP_FORMAT = "%Y-%m-%d_%H_%M"


class ClasswiseAECLI(LightningCLI):
    """Enhanced Lightning CLI for class-wise autoencoder experiments.

    Features:
    - Automatic checkpoint saving (best and last)
    - Automatic config saving
    - MLFlow logging integration
    - Learning rate monitoring
    - Optional early stopping
    - Automatic run_id generation with timestamp

    Usage:
        # Training
        python src/cli.py fit --config conf/experiment/mnist_dino_simple.yaml

        # Resume from checkpoint
        python src/cli.py fit --config conf/experiment/mnist_dino_simple.yaml \\
            --ckpt_path path/to/checkpoint.ckpt

        # Testing (computes full RER statistics)
        python src/cli.py test --config conf/experiment/mnist_dino_simple.yaml \\
            --ckpt_path path/to/checkpoint.ckpt

        # Validation
        python src/cli.py validate --config conf/experiment/mnist_dino_simple.yaml \\
            --ckpt_path path/to/checkpoint.ckpt

        # Custom run ID
        python src/cli.py fit --config conf/experiment/mnist_dino_simple.yaml \\
            --run_id my_custom_run
    """

    def __init__(self, *args, **kwargs):
        """Initialize the CLI with enhanced defaults."""
        self._check_custom_commands()
        self._set_default_kwargs(kwargs)
        super().__init__(*args, **kwargs)

    def _check_custom_commands(self) -> None:
        """Check for and handle custom commands."""
        if len(sys.argv) > 1 and sys.argv[1] in CUSTOM_COMMANDS:
            if sys.argv[1] == "study":
                self._run_study()
            else:
                print(f"Custom command '{sys.argv[1]}' detected but not yet implemented.")
                sys.exit(1)

    @staticmethod
    def _run_study() -> None:
        """Run an Optuna hyperparameter study and exit."""
        from study import parse_study_args, run_study

        base_path, study_path, n_trials = parse_study_args()
        run_study(base_path, study_path, n_trials_override=n_trials)
        sys.exit(0)

    def _set_default_kwargs(self, kwargs: dict) -> None:
        """Set default keyword arguments if not provided."""
        kwargs.setdefault("save_config_callback", SaveMLFlowConfigCallback)
        kwargs.setdefault(
            "parser_kwargs",
            {"parser_mode": "omegaconf", "error_handler": None},
        )

    def add_arguments_to_parser(self, parser) -> None:
        """Add custom arguments for checkpointing, logging, and run management."""
        self._add_checkpoint_arguments(parser)
        self._add_early_stopping_arguments(parser)
        self._add_experiment_arguments(parser)

    def _add_checkpoint_arguments(self, parser) -> None:
        """Add checkpoint-related CLI arguments."""
        parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default=None,
            help=(
                "Directory to save checkpoints. "
                f"Defaults to {DEFAULT_EXPERIMENTS_ROOT}/<experiment_name>/<run_id>"
            ),
        )
        parser.add_argument(
            "--checkpoint_monitor",
            type=str,
            default="val/loss",
            help="Metric to monitor for checkpointing",
        )
        parser.add_argument(
            "--checkpoint_mode",
            type=str,
            default="min",
            help="Mode for checkpoint monitoring (min or max)",
        )

    def _add_early_stopping_arguments(self, parser) -> None:
        """Add early stopping-related CLI arguments."""
        parser.add_argument(
            "--early_stopping",
            type=bool,
            default=False,
            help="Enable early stopping",
        )
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=10,
            help="Patience for early stopping",
        )

    def _add_experiment_arguments(self, parser) -> None:
        """Add experiment management CLI arguments."""
        parser.add_argument(
            "--experiment_name",
            type=Optional[str],
            default=None,
            help="MLFlow experiment name (overrides config)",
        )
        parser.add_argument(
            "--run_id",
            type=Optional[str],
            default=None,
            help=(
                "Custom run ID. Defaults to <experiment_name>_<timestamp> "
                f"format ({RUN_ID_TIMESTAMP_FORMAT})"
            ),
        )

    def before_instantiate_classes(self) -> None:
        """Set up callbacks and experiment metadata before instantiation."""
        config = self._get_subcommand_config()
        callbacks = self._ensure_callbacks_list(config)

        # Generate experiment identifiers
        experiment_name = self._resolve_experiment_name(config)
        run_id = self._resolve_run_id(experiment_name)

        # Configure MLFlow logger
        self._configure_mlflow_logger(config, run_id)

        # Add standard callbacks
        self._add_lr_monitor_callback(callbacks, config)
        self._add_early_stopping_callback(callbacks)

        config["trainer"]["callbacks"] = callbacks

    def _get_subcommand_config(self) -> dict:
        """Get the configuration for the current subcommand."""
        return self.config[self.config["subcommand"]]

    def _ensure_callbacks_list(self, config: dict) -> list:
        """Ensure trainer callbacks is an initialized list."""
        if config["trainer"].get("callbacks") is None:
            config["trainer"]["callbacks"] = []
        return config["trainer"]["callbacks"]

    def _resolve_experiment_name(self, config: dict) -> str:
        """Resolve experiment name from CLI args or config.

        Priority: CLI argument > logger config > default
        """
        # CLI argument takes precedence
        if self.config.get("experiment_name"):
            return self.config["experiment_name"]

        # Extract from MLFlow logger config
        logger_config = config["trainer"].get("logger")
        if isinstance(logger_config, dict):
            experiment_name = logger_config.get("init_args", {}).get("experiment_name")
            if experiment_name:
                return experiment_name

        return "ClasswiseAE"

    def _resolve_run_id(self, experiment_name: str) -> str:
        """Resolve run_id from CLI args or generate timestamp-based one."""
        if self.config.get("run_id"):
            return self.config["run_id"]

        timestamp = datetime.now().strftime(RUN_ID_TIMESTAMP_FORMAT)
        return f"{experiment_name}_{timestamp}"

    def _resolve_checkpoint_dir(self, experiment_name: str, run_id: str) -> str:
        """Resolve checkpoint directory from CLI args or generate default."""
        if self.config.get("checkpoint_dir"):
            return self.config["checkpoint_dir"]

        return f"{DEFAULT_EXPERIMENTS_ROOT}/{experiment_name}/{run_id}"

    def _configure_mlflow_logger(self, config: dict, run_id: str) -> None:
        """Configure MLFlow logger with run_name if present."""
        logger_config = config["trainer"].get("logger")
        if not isinstance(logger_config, dict):
            return

        if "MLFlowLogger" not in str(logger_config.get("class_path", "")):
            return

        logger_config.setdefault("init_args", {})
        logger_config["init_args"]["run_name"] = run_id
        logger_config["init_args"]["save_dir"] = DEFAULT_EXPERIMENTS_ROOT

    def _add_checkpoint_callback(self, callbacks: list, checkpoint_dir: str) -> None:
        """Add ModelCheckpoint callback."""
        monitor = self.config.get("checkpoint_monitor", "val/loss")
        mode = self.config.get("checkpoint_mode", "min")

        callbacks.append(
            {
                "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                "init_args": {
                    "dirpath": checkpoint_dir,
                    "filename": "best-{epoch:02d}-{val/loss:.4f}",
                    "monitor": monitor,
                    "mode": mode,
                    "save_top_k": 1,
                    "save_last": True,
                    "verbose": True,
                },
            }
        )

    def _add_lr_monitor_callback(self, callbacks: list, config: dict) -> None:
        """Add LearningRateMonitor callback if logger is enabled."""
        logger_config = config["trainer"].get("logger")
        # Skip if logger is disabled (False or None)
        if logger_config is False or logger_config is None:
            return

        callbacks.append(
            {
                "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                "init_args": {"logging_interval": "step"},
            }
        )

    def _add_early_stopping_callback(self, callbacks: list) -> None:
        """Add EarlyStopping callback if enabled."""
        if not self.config.get("early_stopping", False):
            return

        monitor = self.config.get("checkpoint_monitor", "val/loss")
        mode = self.config.get("checkpoint_mode", "min")
        patience = self.config.get("early_stopping_patience", 10)

        callbacks.append(
            {
                "class_path": "lightning.pytorch.callbacks.EarlyStopping",
                "init_args": {
                    "monitor": monitor,
                    "patience": patience,
                    "mode": mode,
                    "verbose": True,
                },
            }
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli = ClasswiseAECLI()
