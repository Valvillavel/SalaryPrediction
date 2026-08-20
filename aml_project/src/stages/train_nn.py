import yaml
import pandas as pd
from src import config
from src.models.neuralNetworkModel import NeuralNetworkModel


def main():
    params = yaml.safe_load(open("params.yaml"))["neural_network"]

    X_train = pd.read_csv(config.PROCESSED_DATA_DIR / "X_train_processed.csv")
    y_train = pd.read_csv(config.PROCESSED_DATA_DIR / "y_train.csv").squeeze()
    y_train_bin = (y_train == config.POSITIVE_LABEL).astype(int)

    nn = NeuralNetworkModel(input_dim=X_train.shape[1])
    nn.fit(X_train.values, y_train_bin.values,
           epochs=params["epochs"], batch_size=params["batch_size"])
    nn.save(config.MODELS_DIR / "neural_network_model.keras")


if __name__ == "__main__":
    main()
