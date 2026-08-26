import numpy as np
from src.features.salaryPreprocessor import SalaryPreprocessor


def test_split_is_stratified(sample_df):
    prep = SalaryPreprocessor(
        target_col="salary", test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = prep.split(sample_df)
    original_ratio = sample_df["salary"].value_counts(normalize=True)
    train_ratio = y_train.value_counts(normalize=True)
    # tolerancia amplia por el tamaño pequeño de la fixture
    for label in original_ratio.index:
        assert abs(original_ratio[label] - train_ratio[label]) < 0.2


def test_fit_transform_no_nulls_and_correct_shape(sample_df):
    prep = SalaryPreprocessor(target_col="salary")
    X_train, X_test, y_train, y_test = prep.split(sample_df)
    X_train_p, X_test_p = prep.fit_transform(X_train, X_test)
    assert np.isnan(X_train_p).sum() == 0
    assert np.isnan(X_test_p).sum() == 0
    assert X_train_p.shape[0] == len(y_train)
    assert X_test_p.shape[1] == X_train_p.shape[1]


def test_no_data_leakage_preprocessor_fitted_only_on_train(sample_df, mocker):
    prep = SalaryPreprocessor(target_col="salary")
    X_train, X_test, _, _ = prep.split(sample_df)
    spy = mocker.spy(prep, "_build_transformer")
    prep.fit_transform(X_train, X_test)
    spy.assert_called_once()  # el ColumnTransformer se construye y ajusta una sola vez


def test_save_and_load_roundtrip(sample_df, tmp_path):
    prep = SalaryPreprocessor(target_col="salary")
    X_train, X_test, _, _ = prep.split(sample_df)
    prep.fit_transform(X_train, X_test)
    path = tmp_path / "preprocessor.pkl"
    prep.save(path)

    prep2 = SalaryPreprocessor(target_col="salary")
    loaded = prep2.load(path)
    assert loaded is not None
    assert hasattr(loaded, "transform")
