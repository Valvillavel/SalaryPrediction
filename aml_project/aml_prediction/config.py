"""Configuración global del proyecto: rutas y constantes."""

from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJ_ROOT / "models"
FIGURES_DIR = PROJ_ROOT / "reports" / "figures"

RAW_SALARY_CSV = RAW_DATA_DIR / "salary.csv"

TARGET_COLUMN = "salary"
CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
NUMERIC_COLUMNS = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week",
]
RANDOM_STATE = 42