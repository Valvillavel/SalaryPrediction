import numpy as np
from src.features.salaryPreprocessor import SalaryPreprocessor
from src.models.randomForestModel import RandomForestModel


def _get_processed_data(sample_df):
    prep = SalaryPreprocessor(target_col="salary")
    X_train, X_test, y_train, y_test = prep.split(sample_df)
    X_train_p, X_test_p = prep.fit_transform(X_train, X_test)
    return X_train_p, X_test_p, y_train, y_test


def test_fit_predict_shapes(sample_df):
    X_train_p, X_test_p, y_train, y_test = _get_processed_data(sample_df)
    model = RandomForestModel()
    model.fit(X_train_p, y_train)
    preds = model.predict(X_test_p)
    assert len(preds) == len(y_test)


def test_predict_proba_between_0_and_1(sample_df):
    X_train_p, X_test_p, y_train, _ = _get_processed_data(sample_df)
    model = RandomForestModel()
    model.fit(X_train_p, y_train)
    proba = model.predict_proba(X_test_p)
    assert np.all((proba >= 0) & (proba <= 1))


def test_save_and_load(sample_df, tmp_path):
    X_train_p, X_test_p, y_train, _ = _get_processed_data(sample_df)
    model = RandomForestModel()
    model.fit(X_train_p, y_train)
    path = tmp_path / "rf.pkl"
    model.save(path)

    reloaded = RandomForestModel()
    reloaded.load(path)
    preds_original = model.predict(X_test_p)
    preds_reloaded = reloaded.predict(X_test_p)
    assert (preds_original == preds_reloaded).all()
