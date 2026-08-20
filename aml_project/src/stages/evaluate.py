import json
import joblib
import pandas as pd
from keras import models
from src import config
from src.models.modelEvaluator import ModelEvaluator
from src.visualization.plots import PlotHelper


def main():
    X_test = pd.read_csv(config.PROCESSED_DATA_DIR / "X_test_processed.csv")
    y_test = pd.read_csv(config.PROCESSED_DATA_DIR / "y_test.csv").squeeze()
    y_test_bin = (y_test == config.POSITIVE_LABEL).astype(int)

    plotter = PlotHelper(config.FIGURES_DIR)
    evaluator = ModelEvaluator(
        pos_label=config.POSITIVE_LABEL, plotter=plotter)

    logreg = joblib.load(config.MODELS_DIR / "logreg_model.pkl")
    rf = joblib.load(config.MODELS_DIR / "rf_model.pkl")
    nn = models.load_model(
        config.MODELS_DIR / "neural_network_model.keras")

    for name, y_pred, y_proba in [
        ("Logistic Regression", logreg.predict(
            X_test), logreg.predict_proba(X_test)[:, 1]),
        ("Random Forest", rf.predict(X_test), rf.predict_proba(X_test)[:, 1]),
    ]:
        evaluator.evaluate(name, y_test, y_pred, y_test_bin, y_proba)

    nn_proba = nn.predict(X_test.values).ravel()
    nn_pred = [config.POSITIVE_LABEL if p >=
               0.5 else " <=50K" for p in nn_proba]
    evaluator.evaluate("Neural Network", y_test, nn_pred, y_test_bin, nn_proba)

    # DVC metrics: un JSON plano por modelo, para dvc metrics diff / dvc plots
    metrics_out = {name: m for name, m in evaluator.results.items()}
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)


if __name__ == "__main__":
    main()
