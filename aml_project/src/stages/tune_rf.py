import yaml
import mlflow
import pandas as pd
from src import config
from src.models.randomForestTuner import RandomForestTuner
from src.tracking import start_run


def main():
    params = yaml.safe_load(open("params.yaml"))["tuner"]

    X_train = pd.read_csv(config.PROCESSED_DATA_DIR / "X_train_processed.csv")
    y_train = pd.read_csv(config.PROCESSED_DATA_DIR / "y_train.csv").squeeze()

    with start_run("tune_rf"):
        mlflow.log_params(params)  # n_iter, cv_folds

        tuner = RandomForestTuner(
            n_iter=params["n_iter"], cv_folds=params["cv_folds"])
        tuner.fit(X_train, y_train)

        # Run anidado por cada combinación probada en el RandomizedSearchCV
        cv_results = tuner.search.cv_results_
        for i in range(len(cv_results["params"])):
            with mlflow.start_run(run_name=f"trial_{i}", nested=True):
                mlflow.log_params(cv_results["params"][i])
                mlflow.log_metric("mean_test_roc_auc",
                                  cv_results["mean_test_score"][i])
                mlflow.log_metric("mean_train_roc_auc",
                                  cv_results["mean_train_score"][i])

        # Resumen del mejor resultado en el run padre
        mlflow.log_params(
            {f"best_{k}": v for k, v in tuner.best_params.items()})
        mlflow.log_metric("best_cv_roc_auc", tuner.best_score)

        tuner.save_best_model(config.MODELS_DIR / "random_forest_tuned.pkl")
        tuner.save_search(config.MODELS_DIR / "random_search_rf.pkl")
        mlflow.sklearn.log_model(tuner.best_model_, artifact_path="model")


if __name__ == "__main__":
    main()
