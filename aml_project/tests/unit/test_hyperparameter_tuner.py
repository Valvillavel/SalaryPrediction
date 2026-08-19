import pytest
from src.features.salaryPreprocessor import SalaryPreprocessor
from src.models.randomForestTuner import RandomForestTuner


@pytest.mark.slow
def test_tuner_fits_and_exposes_best_params(sample_df):
    prep = SalaryPreprocessor(target_col="salary")
    X_train, X_test, y_train, y_test = prep.split(sample_df)
    X_train_p, _ = prep.fit_transform(X_train, X_test)

    # valores reducidos solo para test
    tuner = RandomForestTuner(n_iter=3, cv_folds=2)
    tuner.fit(X_train_p, y_train)

    assert tuner.best_model_ is not None
    assert isinstance(tuner.best_params, dict)
    assert 0 <= tuner.best_score <= 1


def test_tuner_save_best_model(tmp_path, sample_df):
    prep = SalaryPreprocessor(target_col="salary")
    X_train, X_test, y_train, _ = prep.split(sample_df)
    X_train_p, _ = prep.fit_transform(X_train, X_test)

    tuner = RandomForestTuner(n_iter=2, cv_folds=2)
    tuner.fit(X_train_p, y_train)

    path = tmp_path / "rf_tuned.pkl"
    tuner.save_best_model(path)
    assert path.exists()
