import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from keras import layers, callbacks, Sequential, metrics, models
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src import config
from src.models.logisticRegressionModel import LogisticRegressionModel
from src.models.randomForestModel import RandomForestModel
from src.models.neuralNetworkModel import NeuralNetworkModel
from src.models.randomForestTuner import RandomForestTuner
from src.models.modelEvaluator import ModelEvaluator
from src.visualization.plots import PlotHelper


class MLflowExperimentManager:
    """Gestiona experimentos de MLflow para el proyecto."""

    def __init__(self, experiment_name: str = "Salary_Prediction_Experiments", 
                 tracking_uri: str = "sqlite:///mlflow.db"):
        """
        Inicializa el manager de experimentos.
        
        Args:
            experiment_name: Nombre del experimento en MLflow
            tracking_uri: URI para el tracking de MLflow (local o remoto)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.plotter = PlotHelper(config.FIGURES_DIR)
        self.evaluator = ModelEvaluator(
            pos_label=config.POSITIVE_LABEL, 
            plotter=self.plotter
        )
        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        print(f"MLflow configurado: {self.experiment_name}")
        print(f"Tracking URI: {self.tracking_uri}")

    def get_data_version(self) -> str:
        """Obtiene la versión del dataset desde DVC/Git."""
        try:
            import subprocess
            git_hash = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], 
                capture_output=True, text=True
            ).stdout.strip()
            return git_hash if git_hash else "v1.0.0-local"
        except Exception:
            return "local-no-git"

    def run_logistic_regression_experiments(self, X_train, X_test, y_train, y_test, y_train_bin, y_test_bin):
        """Ejecuta experimentos de Logistic Regression."""
        
        print("\n" + "="*60)
        print("🧪 EXPERIMENTO 1: Logistic Regression (Base)")
        print("="*60)
        
        lr_base = LogisticRegressionModel(random_state=42)
        
        with mlflow.start_run(run_name="LogisticRegression_Exp1_Base") as run:
            mlflow.set_tags({
                "model_type": "Logistic Regression",
                "experiment_type": "Base",
                "developer": "TuNombre",
                "data_version": self.get_data_version()
            })

            params = {
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": 42,
                "solver": "lbfgs"
            }
            mlflow.log_params(params)
            
            # Entrenar
            lr_base.fit(X_train, y_train)
            
            # Predecir
            y_pred = lr_base.predict(X_test)
            y_proba = lr_base.predict_proba(X_test)
            
            # Métricas
            metrics = self.evaluator.evaluate("Logistic Regression", y_test, y_pred, y_test_bin, y_proba)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name.lower().replace("-", "_"), value)
            
            # Loggear modelo
            input_example = X_test[:5]
            mlflow.sklearn.log_model(
                sk_model=lr_base.model,
                artifact_path="model",
                input_example=input_example,
                registered_model_name="LogisticRegression_Model"
            )
            
            print(f"✅ Experimento 1 completado - Run ID: {run.info.run_id}")
            print(f"📊 Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1-Score']:.4f}")
        
        # Experimento 2: Logistic Regression con parámetros modificados
        print("\n" + "="*60)
        print("🧪 EXPERIMENTO 2: Logistic Regression (Tuned)")
        print("="*60)
        
        lr_tuned = LogisticRegressionModel(random_state=99)
        
        with mlflow.start_run(run_name="LogisticRegression_Exp2_Tuned") as run:
            mlflow.set_tags({
                "model_type": "Logistic Regression",
                "experiment_type": "Tuned",
                "developer": "TuNombre",
                "data_version": self.get_data_version()
            })
            
            params = {
                "max_iter": 2000,
                "class_weight": "balanced",
                "random_state": 99,
                "solver": "liblinear",
                "penalty": "l2",
                "C": 0.1
            }
            mlflow.log_params(params)
            
            # Entrenar con parámetros diferentes
            lr_tuned.model.set_params(**{k: v for k, v in params.items() if hasattr(lr_tuned.model, k)})
            lr_tuned.fit(X_train, y_train)
            
            # Predecir
            y_pred = lr_tuned.predict(X_test)
            y_proba = lr_tuned.predict_proba(X_test)
            
            # Métricas
            metrics = self.evaluator.evaluate("Logistic Regression Tuned", y_test, y_pred, y_test_bin, y_proba)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name.lower().replace("-", "_"), value)
            
            # Loggear modelo
            input_example = X_test[:5]
            mlflow.sklearn.log_model(
                sk_model=lr_tuned.model,
                artifact_path="model",
                input_example=input_example,
                registered_model_name="LogisticRegression_Model"
            )
            
            print(f"✅ Experimento 2 completado - Run ID: {run.info.run_id}")
            print(f"📊 Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1-Score']:.4f}")

    def run_random_forest_experiments(self, X_train, X_test, y_train, y_test, y_train_bin, y_test_bin):
        """Ejecuta experimentos de Random Forest."""
        
        # Experimento 1: Random Forest con parámetros base
        print("\n" + "="*60)
        print("🧪 EXPERIMENTO 1: Random Forest (Base)")
        print("="*60)
        
        rf_base = RandomForestModel(random_state=42)
        
        with mlflow.start_run(run_name="RandomForest_Exp1_Base") as run:
            mlflow.set_tags({
                "model_type": "Random Forest",
                "experiment_type": "Base",
                "developer": "TuNombre",
                "data_version": self.get_data_version()
            })
            
            params = {
                "n_estimators": 100,
                "max_depth": 15,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1
            }
            mlflow.log_params(params)
            
            rf_base.fit(X_train, y_train)
            
            y_pred = rf_base.predict(X_test)
            y_proba = rf_base.predict_proba(X_test)
            
            metrics = self.evaluator.evaluate("Random Forest", y_test, y_pred, y_test_bin, y_proba)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name.lower().replace("-", "_"), value)
            
            input_example = X_test[:5]
            mlflow.sklearn.log_model(
                sk_model=rf_base.model,
                artifact_path="model",
                input_example=input_example,
                registered_model_name="RandomForest_Model"
            )
            
            print(f"✅ Experimento 1 completado - Run ID: {run.info.run_id}")
            print(f"📊 Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1-Score']:.4f}")
        
        # Experimento 2: Random Forest con parámetros modificados
        print("\n" + "="*60)
        print("🧪 EXPERIMENTO 2: Random Forest (Tuned)")
        print("="*60)
        
        rf_tuned = RandomForestModel(random_state=99)
        
        with mlflow.start_run(run_name="RandomForest_Exp2_Tuned") as run:
            mlflow.set_tags({
                "model_type": "Random Forest",
                "experiment_type": "Tuned",
                "developer": "TuNombre",
                "data_version": self.get_data_version()
            })
            
            params = {
                "n_estimators": 300,
                "max_depth": 20,
                "class_weight": "balanced_subsample",
                "random_state": 99,
                "n_jobs": -1,
                "min_samples_split": 5,
                "min_samples_leaf": 2
            }
            mlflow.log_params(params)
            
            # Aplicar parámetros
            rf_tuned.model.set_params(**params)
            rf_tuned.fit(X_train, y_train)
            
            y_pred = rf_tuned.predict(X_test)
            y_proba = rf_tuned.predict_proba(X_test)
            
            metrics = self.evaluator.evaluate("Random Forest Tuned", y_test, y_pred, y_test_bin, y_proba)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name.lower().replace("-", "_"), value)
            
            input_example = X_test[:5]
            mlflow.sklearn.log_model(
                sk_model=rf_tuned.model,
                artifact_path="model",
                input_example=input_example,
                registered_model_name="RandomForest_Model"
            )
            
            print(f"✅ Experimento 2 completado - Run ID: {run.info.run_id}")
            print(f"📊 Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1-Score']:.4f}")

    def run_neural_network_experiments(self, X_train, X_test, y_train, y_test, y_train_bin, y_test_bin):
        """Ejecuta experimentos de Neural Network."""
        
        # Experimento 1: Neural Network con parámetros base
        print("\n" + "="*60)
        print("🧪 EXPERIMENTO 1: Neural Network (Base)")
        print("="*60)
        
        nn_base = NeuralNetworkModel(input_dim=X_train.shape[1])
        
        with mlflow.start_run(run_name="NeuralNetwork_Exp1_Base") as run:
            mlflow.set_tags({
                "model_type": "Neural Network",
                "experiment_type": "Base",
                "developer": "TuNombre",
                "data_version": self.get_data_version()
            })
            
            params = {
                "epochs": 100,
                "batch_size": 128,
                "optimizer": "adam",
                "hidden_layers": [64, 32],
                "dropout": 0.3
            }
            mlflow.log_params(params)
            
            nn_base.fit(X_train, y_train_bin, epochs=100, batch_size=128)
            
            y_pred = nn_base.predict(X_test, threshold=0.5)
            y_pred = [' >50K' if p == 1 else ' <=50K' for p in y_pred]
            y_proba = nn_base.predict_proba(X_test)
            
            eval_metrics  = self.evaluator.evaluate("Neural Network", y_test, y_pred, y_test_bin, y_proba)
            for metric_name, value in eval_metrics.items():
                mlflow.log_metric(metric_name.lower().replace("-", "_"), value)
            
            # Loggear modelo Keras como artefacto
            mlflow.tensorflow.log_model(
                nn_base.model,
                artifact_path="model",
                registered_model_name="NeuralNetwork_Model"
            )
            
            print(f"✅ Experimento 1 completado - Run ID: {run.info.run_id}")
            print(f"📊 Accuracy: {eval_metrics['Accuracy']:.4f}, F1: {eval_metrics['F1-Score']:.4f}")
        
        # Experimento 2: Neural Network con parámetros modificados
        print("\n" + "="*60)
        print("🧪 EXPERIMENTO 2: Neural Network (Tuned)")
        print("="*60)
        
        nn_tuned = NeuralNetworkModel(input_dim=X_train.shape[1])
        
        with mlflow.start_run(run_name="NeuralNetwork_Exp2_Tuned") as run:
            mlflow.set_tags({
                "model_type": "Neural Network",
                "experiment_type": "Tuned",
                "developer": "TuNombre",
                "data_version": self.get_data_version()
            })
            
            params = {
                "epochs": 150,
                "batch_size": 64,
                "optimizer": "adam",
                "hidden_layers": [128, 64, 32],
                "dropout": 0.2,
                "learning_rate": 0.001
            }
            mlflow.log_params(params)
            
            # Modificar arquitectura de la NN para el segundo experimento
            nn_tuned.model = Sequential([
                layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
                layers.Dropout(0.2),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(32, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"),
            ])
            nn_tuned.model.compile(
                optimizer="adam", loss="binary_crossentropy",
                metrics=["accuracy", metrics.Precision(), metrics.Recall(), metrics.AUC(name="auc")]
            )
            
            nn_tuned.fit(X_train, y_train_bin, epochs=150, batch_size=64)
            
            y_pred = nn_tuned.predict(X_test, threshold=0.5)
            y_pred_str = [' >50K' if p == 1 else ' <=50K' for p in y_pred]
            y_proba = nn_tuned.predict_proba(X_test)
            
            eval_metrics  = self.evaluator.evaluate("Neural Network Tuned", y_test, y_pred_str, y_test_bin, y_proba)
            for metric_name, value in eval_metrics.items():
                mlflow.log_metric(metric_name.lower().replace("-", "_"), value)
            
            mlflow.tensorflow.log_model(
                nn_tuned.model,
                artifact_path="model",
                registered_model_name="NeuralNetwork_Model"
            )
            
            print(f"✅ Experimento 2 completado - Run ID: {run.info.run_id}")
            print(f"📊 Accuracy: {eval_metrics['Accuracy']:.4f}, F1: {eval_metrics['F1-Score']:.4f}")

    def run_all_experiments(self, X_train, X_test, y_train, y_test, y_train_bin, y_test_bin):
        """Ejecuta todos los experimentos de todos los modelos."""
        
        print("\n" + "="*60)
        print("🚀 INICIANDO EXPERIMENTOS MLflow")
        print("="*60)
        print(f"📊 Experiment Name: {self.experiment_name}")
        print(f"📁 Data version: {self.get_data_version()}")
        print(f"📊 X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print("="*60 + "\n")
        
        # Logistic Regression
        self.run_logistic_regression_experiments(X_train, X_test, y_train, y_test, y_train_bin, y_test_bin)
        
        # Random Forest
        self.run_random_forest_experiments(X_train, X_test, y_train, y_test, y_train_bin, y_test_bin)
        
        # Neural Network
        self.run_neural_network_experiments(X_train, X_test, y_train, y_test, y_train_bin, y_test_bin)
        
        print("\n" + "="*60)
        print("✅ ¡TODOS LOS EXPERIMENTOS COMPLETADOS!")
        print("="*60)
        print("📊 Para ver los resultados, ejecuta: mlflow ui")
        print("🌐 Abre en tu navegador: http://localhost:5000")
        print("="*60)


def main():
    """Función principal para ejecutar los experimentos."""
    
    print("="*60)
    print("🔧 CARGANDO DATOS Y PREPROCESADORES")
    print("="*60)
    
    import joblib
    from src.data.dataLoader import DataLoader
    from src.features.salaryPreprocessor import SalaryPreprocessor
    from src.config import RAW_DATA_PATH, PROCESSED_DATA_DIR
    
    # Cargar datos raw
    loader = DataLoader(RAW_DATA_PATH)
    df = loader.load()
    
    # Preprocesar
    preprocessor = SalaryPreprocessor(target_col=config.TARGET_COLUMN)
    X_train, X_test, y_train, y_test = preprocessor.split(df)
    X_train_p, X_test_p = preprocessor.fit_transform(X_train, X_test)
    
    # Convertir labels a binario
    y_train_bin = (y_train == config.POSITIVE_LABEL).astype(int)
    y_test_bin = (y_test == config.POSITIVE_LABEL).astype(int)
    
    print(f"✅ Datos cargados correctamente")
    print(f"📊 X_train: {X_train_p.shape}, X_test: {X_test_p.shape}")
    print(f"📊 y_train: {y_train.shape[0]}, y_test: {y_test.shape[0]}")
    
    # Ejecutar experimentos
    manager = MLflowExperimentManager(
        experiment_name="Salary_Prediction_Experiments",
        tracking_uri="sqlite:///mlflow.db"
    )
    
    manager.run_all_experiments(X_train_p, X_test_p, y_train, y_test, y_train_bin, y_test_bin)


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parents[2])
    main()