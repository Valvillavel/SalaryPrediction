"""Responsabilidad única: leer el CSV crudo y exponer un `DataFrame` limpio de responsabilidades de análisis o modelado."""

import pandas as pd


class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.filepath)
        return self.df

    def get_missing_summary(self) -> pd.Series:
        return self.df.isnull().sum()

    def get_info(self):
        return self.df.info()
