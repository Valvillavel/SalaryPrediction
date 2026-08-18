"""Ingeniería de variables: encoding, escalado y balanceo de clases."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer

from aml_prediction import config


def build_preprocessor():
    """Construye el ColumnTransformer para variables numéricas y categóricas.

    :return: sklearn ColumnTransformer sin ajustar (fit)
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, config.NUMERIC_COLUMNS),
        ("cat", categorical_pipe, config.CATEGORICAL_COLUMNS),
    ])


def balance_with_smote(X_train, y_train):
    """Aplica SMOTE solo sobre el conjunto de entrenamiento, ya transformado a numérico.

    :param X_train: matriz numérica (después del ColumnTransformer)
    :param y_train: etiquetas de entrenamiento
    :return: X_train balanceado, y_train balanceado
    """
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=config.RANDOM_STATE)
    return smote.fit_resample(X_train, y_train)