import pandas as pd

from aml_prediction import config


class SalaryDataset:
    """Encapsula la carga y limpieza del dataset de salarios.

    :param csv_path: ruta al archivo csv crudo
    """

    def __init__(self, csv_path=config.RAW_SALARY_CSV):
        self._csv_path = csv_path
        self._df = None

    @property
    def data(self):
        """Devuelve el DataFrame cargado (y limpio, si ya se llamó a clean())."""
        if self._df is None:
            raise ValueError("Los datos aún no fueron cargados. Llamá a load() primero.")
        return self._df

    def load(self):
        """Carga el csv crudo en memoria.

        :return: self, para poder encadenar .load().clean()
        """
        self._df = pd.read_csv(self._csv_path)
        return self

    def clean(self):
        """Limpia el dataset: quita espacios, normaliza '?' a NaN y elimina duplicados.

        :return: self
        """
        df = self._df.copy()

        # Los strings vienen con un espacio al inicio: ' State-gov' -> 'State-gov'
        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].apply(lambda col: col.str.strip())

        # Los faltantes están codificados como '?'
        df[obj_cols] = df[obj_cols].replace("?", pd.NA)

        n_antes = df.shape[0]
        df = df.drop_duplicates()
        print(f"Duplicados eliminados: {n_antes - df.shape[0]}")

        self._df = df
        return self

    def train_test_split(self, test_size=0.2):
        """Separa en train/test de forma estratificada por la variable objetivo.

        :param test_size: proporción del set de prueba
        :return: X_train, X_test, y_train, y_test
        """
        from sklearn.model_selection import train_test_split

        X = self._df.drop(columns=[config.TARGET_COLUMN])
        y = self._df[config.TARGET_COLUMN]
        return train_test_split(
            X, y, test_size=test_size, stratify=y,
            random_state=config.RANDOM_STATE,
        )