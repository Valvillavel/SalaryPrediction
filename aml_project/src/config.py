"""Configuración global del proyecto: rutas y constantes."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

RAW_DATA_PATH = RAW_DATA_DIR / "salary.csv"
TARGET_COLUMN = "salary"
POSITIVE_LABEL = " >50K"
RANDOM_STATE = 42
