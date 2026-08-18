"""Entrenamiento del modelo de clasificación de salario."""

import joblib
from sklearn.ensemble import RandomForestClassifier

from aml_prediction import config
from aml_prediction.dataset import SalaryDataset
from aml_prediction.features import build_preprocessor, balance_with_smote


def run_training():
    """Ejecuta el pipeline completo: carga, limpieza, split, encoding, balanceo y entrenamiento.

    :return: modelo entrenado, preprocesador ajustado, y (X_test, y_test) para evaluar
    """
    ds = SalaryDataset().load().clean()
    X_train, X_test, y_train, y_test = ds.train_test_split()

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    X_train_bal, y_train_bal = balance_with_smote(X_train_t, y_train)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train_bal, y_train_bal)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODELS_DIR / "modelo_salary_rf.pkl")
    joblib.dump(preprocessor, config.MODELS_DIR / "preprocessor.pkl")

    return model, preprocessor, X_test_t, y_test


if __name__ == "__main__":
    run_training()