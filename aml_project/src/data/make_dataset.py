"""Script de preparación de datos: capa fina sobre aml_prediction.dataset."""

import os
import yaml

from aml_prediction.dataset import SalaryDataset
from aml_prediction import config


def process_data(output_path="data/processed/clean_salary.csv"):
    """Carga y limpia el dataset crudo usando SalaryDataset, y guarda el resultado.

    :param output_path: ruta donde se guarda el csv procesado
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ds = SalaryDataset(csv_path=config.RAW_SALARY_CSV).load().clean()
    ds.data.to_csv(output_path, index=False)
    print(f"✅ Datos procesados en: {output_path}")


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)  # por si en el futuro necesitás algún param acá
    process_data()
