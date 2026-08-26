"""Carga el modelo y el preprocesador de producción desde el MLflow Model Registry."""

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from pathlib import Path
import os

MODEL_NAME = "aml_prediction_Model"
# Obtener el path absoluto al mlflow.db en el directorio aml_project
# Convertir a formato POSIX para SQLite (con / en lugar de \)
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mlflow.db")).replace("\\", "/")
TRACKING_URI = f"sqlite:///{DB_PATH}"

mlflow.set_tracking_uri(TRACKING_URI)


def load_model():
    """Carga la versión en Production del modelo registrado.

    :return: modelo cargado como pyfunc, o None si no hay versión en Production
    """
    try:
        model_uri = f"models:/{MODEL_NAME}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        print(f"[model_loader] No se pudo cargar el modelo: {e}")
        return None


def load_preprocessor(run_id: str):
    """Descarga el preprocessor.pkl logueado como artefacto en el mismo run del modelo.

    :param run_id: run_id de MLflow de donde se registró el modelo
    :return: el ColumnTransformer ya ajustado, o None si no se encuentra
    """
    import joblib

    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="preprocessor/preprocessor.pkl"
        )
        return joblib.load(local_path)
    except Exception as e:
        print(f"[model_loader] No se pudo cargar el preprocessor: {e}")
        return None


def get_model_metadata():
    """Obtiene versión y run_id de la versión en Production.

    :return: dict con 'version' y 'run_id'
    """
    client = MlflowClient(tracking_uri=TRACKING_URI)
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            return {"version": "Desconocida", "run_id": "Desconocido"}
        v = versions[0]
        return {"version": v.version, "run_id": v.run_id}
    except Exception as e:
        print(f"[model_loader] No se pudo obtener metadata: {e}")
        return {"version": "Desconocida", "run_id": "Desconocido"}
