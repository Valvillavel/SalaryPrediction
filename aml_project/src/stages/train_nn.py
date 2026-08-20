import yaml
import mlflow
import pandas as pd
from src import config
from src.models.neuralNetworkModel import NeuralNetworkModel
from src.tracking import start_run


def main():
    params = yaml.safe_load(open("params.yaml"))["neural_network"]

    X_train = pd.read_csv(config.PROCESSED_DATA_DIR / "X_train_processed.csv")
    y_train = pd.read_csv(config.PROCESSED_DATA_DIR / "y_train.csv").squeeze()
    y_train_bin = (y_train == config.POSITIVE_LABEL).astype(int)

    with start_run("train_nn"):
        mlflow.log_params(params)  # epochs, batch_size (vienen de params.yaml)

        nn = NeuralNetworkModel(input_dim=X_train.shape[1])
        nn.fit(X_train.values, y_train_bin.values,
               epochs=params["epochs"], batch_size=params["batch_size"])

        # Curvas de entrenamiento, época a época
        h = nn.history.history
        for epoch in range(len(h["loss"])):
            mlflow.log_metrics({
                "loss": h["loss"][epoch],
                "val_loss": h["val_loss"][epoch],
                "auc": h["auc"][epoch],
                "val_auc": h["val_auc"][epoch],
            }, step=epoch)

        nn.save(config.MODELS_DIR / "neural_network_model.keras")
        mlflow.keras.log_model(nn.model, artifact_path="model")


if __name__ == "__main__":
    main()
