"""Encapsula el tracking de experimentos con MLflow (parámetros, métricas, artefactos)."""

import subprocess
from typing import Any, Dict

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from aml_prediction.tracking.artifact_generator import ArtifactGenerator


class MLflowExperimentTracker:
    """Registra una corrida de entrenamiento completa en MLflow."""

    def __init__(self, experiment_name: str, tracking_uri: str = "sqlite:///mlflow.db"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    @staticmethod
    def get_git_commit_hash() -> str:
        """Obtiene el hash corto del commit actual de Git.

        :return: hash corto del commit (7 caracteres) o "unknown" si falla
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return "unknown"

    def log_run(
        self,
        model,
        X_train, y_train, X_test, y_test,
        params: Dict[str, Any],
        data_version: str = "v1.0",
        run_name: str = "run",
        dvc_stage: str = "train",
        register_as: str = None,
    ):
        """Entrena, evalúa y registra todo en MLflow.

        :param model: instancia de un modelo sklearn sin entrenar
        :param X_train: features de entrenamiento (ya transformadas)
        :param y_train: etiquetas de entrenamiento (0/1)
        :param X_test: features de test (ya transformadas)
        :param y_test: etiquetas de test (0/1)
        :param params: hiperparámetros a asignar con model.set_params(**params)
        :param data_version: hash de dvc/git para trazabilidad
        :param run_name: nombre de la corrida en MLflow
        :param dvc_stage: nombre del stage de DVC que ejecuta este script
        :param register_as: si se pasa, registra el modelo en el Model Registry con ese nombre
        :return: run_id de MLflow
        """
        with mlflow.start_run(run_name=run_name) as run:
            # Tags para trazabilidad
            git_commit = self.get_git_commit_hash()
            mlflow.set_tags({
                "project": "aml_prediction",
                "framework": "scikit-learn",
                "data_version": data_version,
                "git_commit": git_commit,
                "dvc_stage": dvc_stage,
            })

            # Log de parámetros
            mlflow.log_params(params)
            model.set_params(**params)
            model.fit(X_train, y_train)

            # Predicción y métricas
            preds = model.predict(X_test)
            mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
            mlflow.log_metric("f1_score", f1_score(y_test, preds, pos_label=1))
            mlflow.log_metric("precision", precision_score(y_test, preds, pos_label=1, zero_division=0))
            mlflow.log_metric("recall", recall_score(y_test, preds, pos_label=1, zero_division=0))

            # Artefactos: matriz de confusión
            cm_path = ArtifactGenerator.plot_confusion_matrix(y_test, preds)
            mlflow.log_artifact(cm_path, artifact_path="charts")

            # Artefactos: curva ROC (solo si el modelo tiene predict_proba)
            if hasattr(model, "predict_proba"):
                roc_path = ArtifactGenerator.plot_roc_curve(model, X_test, y_test)
                mlflow.log_artifact(roc_path, artifact_path="charts")

            # Log del modelo
            input_example = X_test[:5] if hasattr(X_test, "__getitem__") else None
            if register_as:
                mlflow.sklearn.log_model(
                    sk_model=model, artifact_path="model",
                    input_example=input_example, registered_model_name=register_as,
                )
            else:
                mlflow.sklearn.log_model(sk_model=model, artifact_path="model", input_example=input_example)

            print(f"🔬 [MLflow] run_id={run.info.run_id} | run_name={run_name} | git={git_commit}")
            return run.info.run_id
