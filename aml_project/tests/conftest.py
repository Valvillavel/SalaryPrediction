import pytest
import pandas as pd
from pathlib import Path
import shutil
import matplotlib
matplotlib.use("Agg")


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_csv_path():
    return FIXTURES_DIR / "sample_salary.csv"


@pytest.fixture
def sample_df(sample_csv_path):
    return pd.read_csv(sample_csv_path)


@pytest.fixture
def tmp_project_dirs(tmp_path):
    """Simula data/raw, data/processed, models y reports/figures en un directorio temporal."""
    dirs = {
        "raw": tmp_path / "data" / "raw",
        "processed": tmp_path / "data" / "processed",
        "models": tmp_path / "models",
        "figures": tmp_path / "reports" / "figures",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


@pytest.fixture
def raw_csv_in_tmp(tmp_project_dirs, sample_csv_path):
    dest = tmp_project_dirs["raw"] / "salary.csv"
    shutil.copy(sample_csv_path, dest)
    return dest
