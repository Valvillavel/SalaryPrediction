import yaml
from src import config
from src.data.dataLoader import DataLoader
from src.features.salaryPreprocessor import SalaryPreprocessor


def main():
    params = yaml.safe_load(open("params.yaml"))["preprocessing"]

    df = DataLoader(config.RAW_DATA_PATH).load()
    prep = SalaryPreprocessor(
        target_col=config.TARGET_COLUMN,
        test_size=params["test_size"],
        random_state=params["random_state"],
    )
    X_train, X_test, y_train, y_test = prep.split(df)
    X_train_p, X_test_p = prep.fit_transform(X_train, X_test)

    prep.save(config.MODELS_DIR / "preprocessor.pkl")
    prep.save_processed_datasets(X_train_p, X_test_p, y_train, y_test,
                                 config.PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()
