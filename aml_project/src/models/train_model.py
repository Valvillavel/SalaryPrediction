"""Entrena el modelo de salary, registra métricas para DVC Y para MLflow."""

import json
import os

import pandas as pd
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from aml_prediction.features import build_preprocessor, balance_with_smote
from aml_prediction.tracking.experiment_tracker import MLflowExperimentTracker


class ModelTrainer:
    """Orquesta entrenamiento, evaluación (para DVC) y tracking (para MLflow)."""

    def __init__(self, params_path: str, data_path: str, metrics_dir: str):
        self.params = self._load_params(params_path)
        self.data_path = data_path
        self.metrics_dir = metrics_dir
        self.model = None

    def _load_params(self, params_path: str) -> dict:
        with open(params_path, "r") as f:
            return yaml.safe_load(f)

    def prepare_data(self):
        """Carga y prepara los datos para entrenamiento.
        
        Convierte las etiquetas a 0/1 para compatibilidad con MLflow (ROC curve).
        """
        df = pd.read_csv(self.data_path)
        target_col = self.params["train"]["target_col"]

        X = df.drop(columns=[target_col])
        # Convertir y a 0/1 (necesario para ROC curve y métricas consistentes)
        y = df[target_col].map({"<=50K": 0, ">50K": 1})

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.params["prepare"]["split_ratio"],
            random_state=self.params["prepare"]["random_state"],
            stratify=y,
        )

        # Preprocesamiento
        preprocessor = build_preprocessor()
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        # OPCIÓN 1: Usar SMOTE para balancear (método actual)
        X_train_bal, y_train_bal = balance_with_smote(X_train_t, y_train)
        
        # OPCIÓN 2: Usar class_weight="balanced" en vez de SMOTE
        # Descomentá las siguientes líneas y comentá la línea de SMOTE de arriba:
        # X_train_bal = X_train_t
        # y_train_bal = y_train
        # Y agregá class_weight="balanced" en los parámetros del modelo (ver abajo)

        # Guardar preprocessor para DVC
        os.makedirs("models", exist_ok=True)
        joblib.dump(preprocessor, "models/preprocessor.pkl")

        return X_train_bal, X_test_t, y_train_bal, y_test

    def run(self):
        """Ejecuta el pipeline completo: prepara datos, entrena y registra en DVC + MLflow."""
        X_train, X_test, y_train, y_test = self.prepare_data()

        # --- 1. Entrenamiento para DVC (genera metrics/eval.json) ---
        self.model = RandomForestClassifier(
            n_estimators=self.params["train"]["n_estimators"],
            max_depth=self.params["train"]["max_depth"],
            random_state=self.params["prepare"]["random_state"],
            n_jobs=-1,
            # class_weight="balanced",  # Descomentá si usás OPCIÓN 2 (sin SMOTE)
        )
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)

        # Métricas para DVC
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1_score": float(f1_score(y_test, preds, zero_division=0)),
        }
        os.makedirs(self.metrics_dir, exist_ok=True)
        with open(os.path.join(self.metrics_dir, "eval.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        pd.DataFrame({"actual": y_test, "predicted": preds}).to_csv(
            os.path.join(self.metrics_dir, "plots.csv"), index=False
        )
        print(f"✅ [DVC] Métricas guardadas en {self.metrics_dir}/eval.json")

        # Guardar modelo para DVC
        joblib.dump(self.model, "models/modelo_salary_rf.pkl")

        # --- 2. Registro en MLflow (con artefactos visuales y Model Registry) ---
        tracker = MLflowExperimentTracker(experiment_name="aml_prediction")
        tracker.log_run(
            model=RandomForestClassifier(
                random_state=self.params["prepare"]["random_state"],
                n_jobs=-1,
                # class_weight="balanced",  # Descomentá si usás OPCIÓN 2
            ),
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            params={
                "n_estimators": self.params["train"]["n_estimators"],
                "max_depth": self.params["train"]["max_depth"],
                "split_ratio": self.params["prepare"]["split_ratio"],
                "random_state": self.params["prepare"]["random_state"],
            },
            run_name="train",  # Nombre del stage de DVC
            dvc_stage="train",  # Tag para trazabilidad con DVC
            register_as="aml_prediction_Model",  # Registrar en Model Registry
        )


if __name__ == "__main__":
    trainer = ModelTrainer("params.yaml", "data/processed/clean_salary.csv", "metrics")
    trainer.run()
