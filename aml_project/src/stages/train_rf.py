import mlflow
import pandas as pd
from src import config
from src.models.randomForestModel import RandomForestModel
from src.tracking import start_run


def main():
    X_train = pd.read_csv(config.PROCESSED_DATA_DIR / "X_train_processed.csv")
    y_train = pd.read_csv(config.PROCESSED_DATA_DIR / "y_train.csv").squeeze()

    with start_run("train_rf"):
        mlflow.sklearn.autolog(log_models=False)

        model = RandomForestModel()
        model.fit(X_train, y_train)
        model.save(config.MODELS_DIR / "rf_model.pkl")

        mlflow.sklearn.log_model(model.model, artifact_path="model")


if __name__ == "__main__":
    main()
