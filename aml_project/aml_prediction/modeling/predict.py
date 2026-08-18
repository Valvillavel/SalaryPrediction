"""Inferencia con el modelo ya entrenado."""

import joblib

from aml_prediction import config


def load_artifacts():
    """Carga el modelo y el preprocesador guardados en disco."""
    model = joblib.load(config.MODELS_DIR / "modelo_salary_rf.pkl")
    preprocessor = joblib.load(config.MODELS_DIR / "preprocessor.pkl")
    return model, preprocessor


def predict(df_nuevo):
    """Predice sobre un DataFrame nuevo con las mismas columnas de entrada.

    :param df_nuevo: DataFrame con las columnas originales (sin la columna salary)
    :return: array con las predicciones
    """
    model, preprocessor = load_artifacts()
    X = preprocessor.transform(df_nuevo)
    return model.predict(X)