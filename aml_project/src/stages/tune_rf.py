import yaml
import pandas as pd
from src import config
from src.models.randomForestTuner import RandomForestTuner


def main():
    params = yaml.safe_load(open("params.yaml"))["tuner"]

    X_train = pd.read_csv(config.PROCESSED_DATA_DIR / "X_train_processed.csv")
    y_train = pd.read_csv(config.PROCESSED_DATA_DIR / "y_train.csv").squeeze()

    tuner = RandomForestTuner(
        n_iter=params["n_iter"], cv_folds=params["cv_folds"])
    tuner.fit(X_train, y_train)
    tuner.save_best_model(config.MODELS_DIR / "random_forest_tuned.pkl")
    tuner.save_search(config.MODELS_DIR / "random_search_rf.pkl")


if __name__ == "__main__":
    main()
