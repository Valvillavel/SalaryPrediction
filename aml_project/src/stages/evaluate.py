import json
import joblib
import mlflow
import pandas as pd
from keras import models
from src import config
from src.models.modelEvaluator import ModelEvaluator
from src.visualization.plots import PlotHelper
from src.tracking import start_run


def main():
    X_test = pd.read_csv(config.PROCESSED_DATA_DIR / "X_test_processed.csv")
    y_test = pd.read_csv(config.PROCESSED_DATA_DIR / "y_test.csv").squeeze()
    y_test_bin = (y_test == config.POSITIVE_LABEL).astype(int)

    plotter = PlotHelper(config.FIGURES_DIR)
    evaluator = ModelEvaluator(
        pos_label=config.POSITIVE_LABEL, plotter=plotter)

    logreg = joblib.load(config.MODELS_DIR / "logreg_model.pkl")
    rf = joblib.load(config.MODELS_DIR / "rf_model.pkl")
    nn = models.load_model(config.MODELS_DIR / "neural_network_model.keras")

    with start_run("evaluate"):
        for name, y_pred, y_proba in [
            ("logreg", logreg.predict(X_test),
             logreg.predict_proba(X_test)[:, 1]),
            ("rf", rf.predict(X_test), rf.predict_proba(X_test)[:, 1]),
        ]:
            metrics = evaluator.evaluate(
                name, y_test, y_pred, y_test_bin, y_proba)
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items()})

        nn_proba = nn.predict(X_test.values).ravel()
        nn_pred = [config.POSITIVE_LABEL if p >=
                   0.5 else " <=50K" for p in nn_proba]
        nn_metrics = evaluator.evaluate(
            "nn", y_test, nn_pred, y_test_bin, nn_proba)
        mlflow.log_metrics({f"nn_{k}": v for k, v in nn_metrics.items()})

        metrics_out = {name: m for name, m in evaluator.results.items()}
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2)

        # reports/metrics.json ya lo consume `dvc metrics diff`;
        # aquí además lo guardamos como artefacto de MLflow junto con las figuras
        mlflow.log_artifact("reports/metrics.json")
        mlflow.log_artifacts(str(config.FIGURES_DIR), artifact_path="figures")

        # Registrar el mejor modelo (según AUC-ROC) en el Model Registry
        best_name = max(evaluator.results,
                        key=lambda n: evaluator.results[n]["AUC-ROC"])
        print(f"Mejor modelo: {best_name}")
        # el registro real se hace apuntando al run/artifact correspondiente,
        # ver sección 7 (Model Registry)


if __name__ == "__main__":
    main()
