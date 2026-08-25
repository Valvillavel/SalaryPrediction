import pandas as pd
from src.data.dataLoader import DataLoader


def test_load_returns_dataframe(sample_csv_path):
    loader = DataLoader(sample_csv_path)
    df = loader.load()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_has_expected_columns(sample_csv_path):
    df = DataLoader(sample_csv_path).load()
    expected = {"age", "workclass", "education", "salary", "hours-per-week"}
    assert expected.issubset(df.columns)


def test_missing_summary_is_series(sample_csv_path):
    loader = DataLoader(sample_csv_path)
    loader.load()
    summary = loader.get_missing_summary()
    assert isinstance(summary, pd.Series)


def test_load_raises_on_missing_file(tmp_path):
    loader = DataLoader(tmp_path / "no_existe.csv")
    import pytest
    with pytest.raises(FileNotFoundError):
        loader.load()
