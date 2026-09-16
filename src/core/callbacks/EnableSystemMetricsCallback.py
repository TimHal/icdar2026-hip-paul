"""Callback to enable MLflow system metrics logging."""

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor


class EnableSystemMetricsCallback(Callback):
    """Enable MLflow system metrics logging (CPU, GPU, memory, network, disk).

    This callback manually starts MLflow's SystemMetricsMonitor since Lightning's
    MLFlowLogger uses MlflowClient.create_run() which doesn't support the
    log_system_metrics parameter.
    """

    def __init__(self, sampling_interval=10):
        """Initialize callback.

        Args:
            sampling_interval: Seconds between metric samples (default: 10)
        """
        super().__init__()
        self._monitor = None
        self._sampling_interval = sampling_interval

    def on_train_start(self, trainer, pl_module):
        """Start system metrics monitoring when training begins."""
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        run_id = trainer.logger.run_id
        if run_id and not self._monitor:
            self._monitor = SystemMetricsMonitor(
                run_id=run_id,
                sampling_interval=self._sampling_interval,
            )
            self._monitor.start()

    def on_train_end(self, trainer, pl_module):
        """Stop system metrics monitoring when training ends."""
        if self._monitor:
            self._monitor.finish()
            self._monitor = None
