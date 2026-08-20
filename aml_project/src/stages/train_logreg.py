import pandas as pd
from src import config
from src.models.logisticRegressionModel import LogisticRegressionModel


def main():
    X_train = pd.read_csv(config.PROCESSED_DATA_DIR / "X_train_processed.csv")
    y_train = pd.read_csv(config.PROCESSED_DATA_DIR / "y_train.csv").squeeze()

    model = LogisticRegressionModel()
    model.fit(X_train, y_train)
    model.save(config.MODELS_DIR / "logreg_model.pkl")


if __name__ == "__main__":
    main()
