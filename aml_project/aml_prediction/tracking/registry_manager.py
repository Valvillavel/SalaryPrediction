"""Gestiona el ciclo de vida de modelos en el MLflow Model Registry."""

from mlflow.tracking import MlflowClient


class ModelRegistryManager:
    """Promueve versiones de modelo entre Staging, Production y Archived."""

    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db"):
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def transition_model_stage(self, model_name: str, version: int, stage: str):
        """Promueve un modelo registrado a un nuevo stage.

        :param model_name: nombre con el que se registró el modelo
        :param version: número de versión a promover
        :param stage: 'Staging', 'Production' o 'Archived'
        """
        self.client.transition_model_version_stage(
            name=model_name, version=version, stage=stage,
            archive_existing_versions=True,
        )
        print(f"📦 [Registry] '{model_name}' v{version} -> {stage}")

    def load_production_model(self, model_name: str):
        """Carga el modelo actualmente en Production para inferencia."""
        import mlflow.pyfunc
        return mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
