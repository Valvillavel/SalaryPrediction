import numpy as np
import pytest
from src.models.neuralNetworkModel import NeuralNetworkModel


def test_model_architecture_output_shape(sample_df):
    input_dim = 10
    nn = NeuralNetworkModel(input_dim=input_dim)
    assert nn.model.input_shape == (None, input_dim)
    assert nn.model.output_shape == (None, 1)


@pytest.mark.slow
def test_fit_predict_with_few_epochs(sample_df):
    from src.features.salaryPreprocessor import SalaryPreprocessor
    prep = SalaryPreprocessor(target_col="salary")
    X_train, X_test, y_train, y_test = prep.split(sample_df)
    X_train_p, X_test_p = prep.fit_transform(X_train, X_test)
    y_train_bin = (y_train == " >50K").astype(int).values

    nn = NeuralNetworkModel(input_dim=X_train_p.shape[1])
    nn.fit(X_train_p, y_train_bin, epochs=2, batch_size=8)
    proba = nn.predict_proba(X_test_p)
    assert np.all((proba >= 0) & (proba <= 1))


def test_save_and_load(tmp_path):
    nn = NeuralNetworkModel(input_dim=5)
    path = tmp_path / "nn_model.keras"
    nn.save(path)

    reloaded = NeuralNetworkModel(input_dim=5)
    reloaded.load(path)
    assert reloaded.model is not None
