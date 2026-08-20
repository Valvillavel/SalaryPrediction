"""Utilidades de MLflow compartidas por todos los stages del pipeline."""

import subprocess
import mlflow
from src import config


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode().strip()
    except Exception:
        return "unknown"


def start_run(run_name):
    """Configura el tracking y abre un run de MLflow con tags básicos.

    Uso:
        with start_run("train_rf"):
            ...
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    run = mlflow.start_run(run_name=run_name)
    mlflow.set_tags({
        "git_commit": _git_commit(),
        "dvc_stage": run_name,
    })
    return run
