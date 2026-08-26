import pytest
from src.pipeline import SalaryPredictionPipeline


@pytest.mark.slow
@pytest.mark.integration
def test_full_pipeline_runs_and_creates_artifacts(monkeypatch, raw_csv_in_tmp, tmp_project_dirs):
    # Redirige las rutas de config a los directorios temporales
    from src import config
    monkeypatch.setattr(config, "RAW_DATA_PATH", raw_csv_in_tmp)
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR",
                        tmp_project_dirs["processed"])
    monkeypatch.setattr(config, "MODELS_DIR", tmp_project_dirs["models"])
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_project_dirs["figures"])

    pipeline = SalaryPredictionPipeline()
    pipeline.run()

    assert (tmp_project_dirs["models"] / "preprocessor.pkl").exists()
    assert (tmp_project_dirs["models"] / "neural_network_model.keras").exists()
    assert (tmp_project_dirs["models"] / "random_forest_tuned.pkl").exists()
    assert (tmp_project_dirs["processed"] / "X_train_processed.csv").exists()
    assert (tmp_project_dirs["processed"] / "y_test.csv").exists()

    comparison = pipeline.evaluator.comparison_table()
    assert set(comparison.index) == {
        "Logistic Regression", "Random Forest", "Neural Network"}
