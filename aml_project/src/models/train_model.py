"""Script de entrenamiento: capa fina sobre aml_prediction.modeling.train."""

import json
import os

import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from aml_prediction.features import build_preprocessor, balance_with_smote
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


def main():
    """Entrena el modelo con los hiperparámetros de params.yaml y registra métricas."""
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # Configurar MLflow
    mlflow.set_experiment("salary-classification")
    
    with mlflow.start_run():
        target_col = params["train"]["target_col"]
        df = pd.read_csv("data/processed/clean_salary.csv")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params["prepare"]["split_ratio"],
            stratify=y, random_state=params["prepare"]["random_state"],
        )

        preprocessor = build_preprocessor()
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        X_train_bal, y_train_bal = balance_with_smote(X_train_t, y_train)

        # Log parameters to MLflow
        mlflow.log_param("n_estimators", params["train"]["n_estimators"])
        mlflow.log_param("max_depth", params["train"]["max_depth"])
        mlflow.log_param("split_ratio", params["prepare"]["split_ratio"])
        mlflow.log_param("random_state", params["prepare"]["random_state"])

        clf = RandomForestClassifier(
            n_estimators=params["train"]["n_estimators"],
            max_depth=params["train"]["max_depth"],
            random_state=params["prepare"]["random_state"],
            n_jobs=-1,
        )
        clf.fit(X_train_bal, y_train_bal)

        preds = clf.predict(X_test_t)
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, pos_label=">50K", zero_division=0)),
            "recall": float(recall_score(y_test, preds, pos_label=">50K", zero_division=0)),
            "f1_score": float(f1_score(y_test, preds, pos_label=">50K", zero_division=0)),
        }

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(clf, "model")

        # DVC outputs (se mantienen para compatibilidad con dvc.yaml)
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/eval.json", "w") as f:
            json.dump(metrics, f, indent=4)

        pd.DataFrame({"actual": y_test.values, "predicted": preds}).to_csv(
            "metrics/plots.csv", index=False
        )

        os.makedirs("models", exist_ok=True)
        joblib.dump(clf, "models/modelo_salary_rf.pkl")
        joblib.dump(preprocessor, "models/preprocessor.pkl")

        print("✅ Entrenamiento completado. Métricas en metrics/eval.json")
        print(f"🔬 MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
