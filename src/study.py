"""Optuna hyperparameter optimization for ClasswiseAE experiments.

Runs an Optuna study using a base experiment config and a study config
that defines the search space, sampler, pruner, and optimization settings.

Usage:
    PYTHONPATH=src python src/cli.py study \
        --config conf/experiment/mnist_dino_simple.yaml \
        --study_config conf/study/example_mnist_dino.yaml

    # Continue an existing study with more trials:
    PYTHONPATH=src python src/cli.py study \
        --config conf/experiment/mnist_dino_simple.yaml \
        --study_config conf/study/example_mnist_dino.yaml \
        --n_trials 20
"""

import argparse
import copy
import gc
import os
import resource
import sys
import traceback
import tempfile
from pathlib import Path

import optuna
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from omegaconf import OmegaConf
from optuna.integration import PyTorchLightningPruningCallback


def _deep_set(d: dict, dotpath: str, value) -> None:
    """Set a value in a nested dict using dot-separated keys."""
    keys = dotpath.split(".")
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value


def _suggest_param(trial: optuna.Trial, name: str, spec: dict):
    """Suggest a parameter value from an Optuna trial based on the spec."""
    ptype = spec["type"]
    if ptype == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    elif ptype == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
    elif ptype == "categorical":
        choices = spec["choices"]
        idx = trial.suggest_categorical(name, list(range(len(choices))))
        return choices[idx]
    else:
        raise ValueError(f"Unknown search space type: {ptype}")


def _instantiate_class(class_path: str, init_args: dict):
    """Instantiate a class from a class_path string and init_args dict."""
    import importlib

    module_name, cls_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls(**init_args)


def _resolve_objectives(study_settings: dict) -> tuple[list[str], list[str]]:
    """Resolve metric(s) and direction(s) from study config.

    Single-objective: metric + direction
    Multi-objective:  metrics + directions
    """
    if "metrics" in study_settings and "directions" in study_settings:
        return study_settings["metrics"], study_settings["directions"]
    return [study_settings["metric"]], [study_settings["direction"]]


def _resolve_categorical(key: str, value, search_space: dict):
    """Resolve a categorical index back to its actual value."""
    if key in search_space and search_space[key]["type"] == "categorical":
        return search_space[key]["choices"][value]
    return value


def _resolve_trial_params(params: dict, search_space: dict, groups: list) -> dict[str, object]:
    """Resolve all trial params (including groups) to concrete config overrides.

    Returns a flat dict of {dotpath: value} suitable for CLI override args.
    Group indices are expanded into their constituent key-value pairs.
    """
    resolved = {}
    for key, value in params.items():
        if key.startswith("_group_"):
            idx = int(key.split("_")[-1])
            group_choice = groups[idx][value]
            resolved.update(group_choice)
        else:
            resolved[key] = _resolve_categorical(key, value, search_space)
    return resolved


def _build_callbacks(
    study_cfg: dict, trial: optuna.Trial, metrics: list[str], directions: list[str]
) -> list:
    """Build pruning + optional early stopping callbacks."""
    monitor = metrics[0]
    callbacks = [PyTorchLightningPruningCallback(trial, monitor=monitor)]

    es_cfg = study_cfg["study"].get("early_stopping")
    if es_cfg:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                patience=es_cfg.get("patience", 10),
                mode="min" if directions[0] == "minimize" else "max",
                verbose=False,
            )
        )
    return callbacks


def _build_mlflow_logger(trainer_cfg: dict, study_name: str, trial_number: int):
    """Build an MLFlowLogger for a trial, re-using the experiment config's settings.

    Returns False if the base config has no MLFlowLogger configured.
    """
    logger_cfg = trainer_cfg.get("logger")
    if not isinstance(logger_cfg, dict) or "MLFlowLogger" not in logger_cfg.get("class_path", ""):
        return False

    from lightning.pytorch.loggers import MLFlowLogger

    init_args = dict(logger_cfg.get("init_args", {}))
    init_args["run_name"] = f"{study_name}_trial_{trial_number:04d}"
    init_args["log_model"] = False
    return MLFlowLogger(**init_args)


def _build_storage(study_cfg: dict) -> optuna.storages.JournalStorage | None:
    """Create JournalStorage for persistent/distributed studies.

    Uses append-only journal files which handle concurrent multi-process
    writes safely (unlike SQLite which crashes under contention).

    Returns None if no storage_dir is configured (in-memory study).
    """
    storage_dir = study_cfg["study"].get("storage_dir")
    if not storage_dir:
        return None

    study_name = study_cfg["study"].get("name", "classwise_ae_hpo")
    os.makedirs(storage_dir, exist_ok=True)
    journal_path = os.path.join(storage_dir, f"{study_name}.journal")
    lock_path = journal_path + ".lock"
    return optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            journal_path,
            lock_obj=optuna.storages.journal.JournalFileOpenLock(lock_path),
        )
    )


def _build_grid_search_space(study_cfg: dict) -> dict[str, list]:
    """Build Optuna GridSampler search space from the study config.

    For grid search, every param must be categorical (explicit value lists).
    Groups are also included as categorical dimensions.
    Returns {param_name: [value_indices]} for GridSampler.
    """
    grid = {}
    for name, spec in study_cfg["search_space"].items():
        if spec["type"] != "categorical":
            raise ValueError(
                f"Grid search requires all params to be categorical, "
                f"but '{name}' is type '{spec['type']}'. "
                f"Convert it to categorical with explicit choices."
            )
        grid[name] = list(range(len(spec["choices"])))
    for i, group_choices in enumerate(study_cfg.get("groups", [])):
        grid[f"_group_{i}"] = list(range(len(group_choices)))
    return grid


def _build_sampler(study_cfg: dict) -> optuna.samplers.BaseSampler:
    """Build an Optuna sampler from the study config."""
    sampler_name = study_cfg["study"].get("sampler", "TPESampler")

    if sampler_name == "GridSampler":
        grid_space = _build_grid_search_space(study_cfg)
        return optuna.samplers.GridSampler(grid_space)

    sampler_map = {
        "TPESampler": optuna.samplers.TPESampler,
        "RandomSampler": optuna.samplers.RandomSampler,
        "CmaEsSampler": optuna.samplers.CmaEsSampler,
        "NSGAIISampler": optuna.samplers.NSGAIISampler,
    }
    sampler_cls = sampler_map.get(sampler_name)
    if sampler_cls is None:
        raise ValueError(
            f"Unknown sampler: {sampler_name}. Choose from: {list(sampler_map.keys()) + ['GridSampler']}"
        )
    return sampler_cls()


def _build_pruner(study_cfg: dict) -> optuna.pruners.BasePruner:
    """Build an Optuna pruner from the study config."""
    pruner_cfg = study_cfg["study"].get("pruner")
    if pruner_cfg is None:
        return optuna.pruners.MedianPruner()

    pruner_map = {
        "MedianPruner": optuna.pruners.MedianPruner,
        "PercentilePruner": optuna.pruners.PercentilePruner,
        "NopPruner": optuna.pruners.NopPruner,
    }
    pruner_cls = pruner_map.get(pruner_cfg["type"])
    if pruner_cls is None:
        raise ValueError(
            f"Unknown pruner: {pruner_cfg['type']}. Choose from: {list(pruner_map.keys())}"
        )

    kwargs = {k: v for k, v in pruner_cfg.items() if k != "type"}
    return pruner_cls(**kwargs)


def _init_mlflow_run(trainer, cfg: dict) -> str | None:
    """Pre-initialize the MLflow run and log the config artifact.

    Must be called BEFORE trainer.fit() to ensure the run is active
    when Lightning logs metrics and the task logs artifacts (e.g. confusion matrices).

    Returns the run_id, or None if the logger is not MLFlowLogger.
    """
    from lightning.pytorch.loggers import MLFlowLogger

    if not isinstance(trainer.logger, MLFlowLogger):
        return None

    # Force lazy run creation and capture run_id before fit
    run_id = trainer.logger.run_id

    # Log full resolved config as artifact
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "config.yaml"
        config_path.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg)))
        trainer.logger.experiment.log_artifact(local_path=str(config_path), run_id=run_id)

    return run_id


def _log_trial_to_mlflow(
    trainer, trial: optuna.Trial, study_name: str, run_id: str | None, error: str | None = None
) -> None:
    """Log Optuna params, tags, and optional error to MLflow."""
    from lightning.pytorch.loggers import MLFlowLogger

    if run_id is None or not isinstance(trainer.logger, MLFlowLogger):
        return

    mlf = trainer.logger.experiment

    # Log Optuna-suggested params with hpo/ prefix
    for param_name, param_value in trial.params.items():
        mlf.log_param(run_id, f"hpo/{param_name}", param_value)
    mlf.set_tag(run_id, "hpo_study", study_name)
    mlf.set_tag(run_id, "hpo_trial", str(trial.number))

    if error:
        mlf.set_tag(run_id, "hpo_status", "FAILED")
        with tempfile.TemporaryDirectory() as tmp_dir:
            error_path = Path(tmp_dir) / "trial_error.txt"
            error_path.write_text(error)
            mlf.log_artifact(local_path=str(error_path), run_id=run_id)
    else:
        mlf.set_tag(run_id, "hpo_status", "OK")


def run_study(
    base_config_path: str, study_config_path: str, n_trials_override: int | None = None
) -> None:
    """Run an Optuna hyperparameter optimization study.

    Supports single-objective and multi-objective optimization.
    Uses JournalStorage for safe concurrent multi-process execution.
    Automatically continues an existing study if one exists with the same name.
    """
    # Raise soft FD limit to hard limit — some environments default to 1024
    # which is too low for multi-trial studies with dataloader workers.
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

    base_cfg: dict = OmegaConf.to_container(OmegaConf.load(base_config_path), resolve=True)  # type: ignore[assignment]
    study_cfg: dict = OmegaConf.to_container(OmegaConf.load(study_config_path), resolve=True)  # type: ignore[assignment]

    if "pin_memory" in base_cfg.get("data", {}).get("init_args", {}):
        print(
            "Warning: 'pin_memory' is set in the base config. Make sure this is intentional for HPO studies, as it can lead to resource exhaustion if not managed carefully."
        )
        print(
            "Consider setting 'pin_memory' to False in the base config and enabling it in the datamodule init_args if needed."
        )
        print("DONT IGNORE THIS WARNING! This caused me lots of troubles and headaches!!!")

    study_settings = study_cfg["study"]
    search_space = study_cfg["search_space"]
    groups = study_cfg.get("groups", [])
    metrics, directions = _resolve_objectives(study_settings)
    multi_objective = len(metrics) > 1
    n_trials = (
        n_trials_override if n_trials_override is not None else int(study_settings["n_trials"])
    )

    study_name = study_settings.get("name", "classwise_ae_hpo")
    storage = _build_storage(study_cfg)

    create_kwargs: dict = dict(
        study_name=study_name,
        sampler=_build_sampler(study_cfg),
        pruner=_build_pruner(study_cfg),
        storage=storage,
        load_if_exists=True,
    )
    if multi_objective:
        create_kwargs["directions"] = directions
    else:
        create_kwargs["direction"] = directions[0]

    study = optuna.create_study(**create_kwargs)

    # Report status
    existing_trials = len(study.trials)
    is_continuation = existing_trials > 0
    storage_dir = study_settings.get("storage_dir")
    storage_info = (
        os.path.join(storage_dir, f"{study_name}.journal") if storage is not None else "in-memory"
    )

    if is_continuation:
        completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        print(
            f"Continuing study: {study_name} ({existing_trials} existing trials, {completed} completed)"
        )
    else:
        print(f"Creating study: {study_name}")
    print(f"  Direction(s): {directions}")
    print(f"  Metric(s): {metrics}")
    print(f"  Trials: {n_trials} (this worker)")
    print(f"  Storage: {storage_info}")
    space_names = list(search_space.keys()) + [f"_group_{i}" for i in range(len(groups))]
    print(f"  Search space: {space_names}")
    if groups:
        for i, g in enumerate(groups):
            print(f"    _group_{i}: {len(g)} choices")
    print()

    if n_trials == 0:
        print("n_trials=0, study created/loaded. Exiting.")
        return

    conditions: dict = study_cfg.get("conditions", {})

    def objective(trial: optuna.Trial) -> float | list[float]:
        cfg = copy.deepcopy(base_cfg)

        # Suggest and apply hyperparameters, respecting conditional dependencies.
        # A param with a condition is only suggested when its parent param has
        # already been resolved to the required value in this trial.
        resolved: dict = {}
        for param_path, spec in search_space.items():
            if param_path in conditions:
                cond = conditions[param_path]
                if resolved.get(cond["when"]) != cond["equals"]:
                    continue  # leave base config value intact
            value = _suggest_param(trial, param_path, spec)
            _deep_set(cfg, param_path, value)
            resolved[param_path] = value

        # Apply group selections (atomic multi-param choices)
        for i, group_choices in enumerate(groups):
            group_name = f"_group_{i}"
            idx = trial.suggest_categorical(group_name, list(range(len(group_choices))))
            for key, val in group_choices[idx].items():
                _deep_set(cfg, key, val)
                resolved[key] = val

        seed = cfg.get("seed_everything", 42)
        seed_everything(seed, workers=True)

        model = _instantiate_class(cfg["model"]["class_path"], cfg["model"].get("init_args", {}))
        datamodule = _instantiate_class(cfg["data"]["class_path"], cfg["data"].get("init_args", {}))

        # Build trainer kwargs
        trainer_cfg = cfg.get("trainer", {})
        safe_keys = [
            "max_epochs",
            "accelerator",
            "devices",
            "precision",
            "log_every_n_steps",
            "check_val_every_n_epoch",
            "gradient_clip_val",
        ]
        trainer_kwargs = {k: trainer_cfg[k] for k in safe_keys if k in trainer_cfg}
        trainer_kwargs["callbacks"] = _build_callbacks(study_cfg, trial, metrics, directions)
        trainer_kwargs["enable_checkpointing"] = False
        trainer_kwargs["logger"] = _build_mlflow_logger(trainer_cfg, study_name, trial.number)

        trainer = Trainer(**trainer_kwargs)
        result = None

        # Pre-initialize MLflow run and log config BEFORE fit.
        # This ensures the run is active when the task logs artifacts
        # (confusion matrices, etc.) and the config is saved reliably.
        run_id = _init_mlflow_run(trainer, cfg)

        try:
            try:
                trainer.fit(model, datamodule=datamodule)
            except optuna.TrialPruned:
                _log_trial_to_mlflow(trainer, trial, study_name, run_id)
                raise
            except Exception as e:
                error_msg = traceback.format_exc()
                print(f"Trial {trial.number} FAILED: {e}")
                trial.set_user_attr("error", error_msg)
                _log_trial_to_mlflow(trainer, trial, study_name, run_id, error=error_msg)
                raise optuna.TrialPruned() from None

            _log_trial_to_mlflow(trainer, trial, study_name, run_id)

            # Extract metrics before cleanup
            callback_metrics = trainer.callback_metrics
            values = []
            for m in metrics:
                if m not in callback_metrics:
                    raise ValueError(
                        f"Metric '{m}' not found. Available: {list(callback_metrics.keys())}"
                    )
                values.append(callback_metrics[m].item())

            result = values if multi_objective else values[0]
        finally:
            # Clean up resources to prevent file descriptor exhaustion.
            # Each trial creates MLflow HTTP connections, dataloader worker
            # processes (persistent_workers=True), and GPU allocations.
            # Without cleanup these accumulate across sequential trials.
            try:
                if hasattr(trainer, "logger") and hasattr(trainer.logger, "finalize"):
                    trainer.logger.finalize("success")
            except Exception:
                pass
            del trainer, model, datamodule
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result

    study.optimize(objective, n_trials=n_trials)

    # Print results
    print("\n" + "=" * 60)
    print("STUDY COMPLETE")
    print("=" * 60)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(
        f"Total trials: {len(study.trials)} ({len(completed)} completed, {len(pruned)} pruned, {len(failed)} failed)"
    )

    if not completed:
        print("No completed trials.")
        return

    if multi_objective:
        pareto = study.best_trials
        print(f"\nPareto-optimal trials: {len(pareto)}")
        for t in pareto:
            vals = ", ".join(f"{m}={v:.6f}" for m, v in zip(metrics, t.values, strict=True))
            print(f"  Trial #{t.number}: {vals}")
            for key, value in _resolve_trial_params(t.params, search_space, groups).items():
                print(f"    {key}: {value}")
    else:
        resolved = _resolve_trial_params(study.best_params, search_space, groups)
        print(f"\nBest trial: #{study.best_trial.number}")
        print(f"Best value ({metrics[0]}): {study.best_value:.6f}")
        print("Best params:")
        for key, value in resolved.items():
            print(f"  {key}: {value}")

        print("\nTo train with the best params, run:")
        override_args = [f"  --{key}={value}" for key, value in resolved.items()]
        print(f"  PYTHONPATH=src python src/cli.py fit --config {base_config_path} \\")
        print(" \\\n".join(override_args))


def parse_study_args() -> tuple[str, str, int | None]:
    """Parse CLI arguments for the study command."""
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter study")
    parser.add_argument("--config", required=True, help="Base experiment config path")
    parser.add_argument("--study_config", required=True, help="Study config path")
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Override number of trials (use to extend an existing study)",
    )
    argv = [a for a in sys.argv[1:] if a != "study"]
    args = parser.parse_args(argv)
    return args.config, args.study_config, args.n_trials


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    base_path, study_path, n_trials = parse_study_args()
    run_study(base_path, study_path, n_trials_override=n_trials)
