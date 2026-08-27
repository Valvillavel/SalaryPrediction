import yaml
import mlflow
import pandas as pd
from src import config
from src.models.randomForestModel import RandomForestModel
from src.tracking import start_run


def main():
    params = yaml.safe_load(open("params.yaml"))["random_forest"]

    X_train = pd.read_csv(config.PROCESSED_DATA_DIR / "X_train_processed.csv")
    y_train = pd.read_csv(config.PROCESSED_DATA_DIR / "y_train.csv").squeeze()

    with start_run("train_rf"):
        mlflow.sklearn.autolog(log_models=False)
        # n_estimators, max_depth, ... desde params.yaml
        mlflow.log_params(params)

        model = RandomForestModel(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=params["class_weight"],
        )
        model.fit(X_train, y_train)
        model.save(config.MODELS_DIR / "rf_model.pkl")

        mlflow.sklearn.log_model(model.model, artifact_path="model")


if __name__ == "__main__":
    main()
