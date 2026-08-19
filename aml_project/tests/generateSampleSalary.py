"""
Genera tests/fixtures/sample_salary.csv: un dataset pequeño, reproducible
y balanceado (misma estructura que salary.csv) para usar en las pruebas
unitarias/integración con pytest.

Uso:
    python generate_sample_salary.py
    python generate_sample_salary.py --n-per-class 25 --output tests/fixtures/sample_salary.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42

WORKCLASS = ["Private", "Self-emp-not-inc",
             "Local-gov", "State-gov", "Federal-gov"]
EDUCATION = [
    ("HS-grad", 9), ("Some-college", 10), ("Bachelors", 13),
    ("Masters", 14), ("Assoc-voc", 11), ("Doctorate", 16),
]
MARITAL_STATUS = ["Married-civ-spouse", "Never-married", "Divorced", "Widowed"]
OCCUPATION = [
    "Exec-managerial", "Craft-repair", "Sales", "Prof-specialty",
    "Adm-clerical", "Machine-op-inspct", "Other-service",
]
RELATIONSHIP = ["Husband", "Not-in-family", "Wife", "Own-child", "Unmarried"]
RACE = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
SEX = ["Male", "Female"]
NATIVE_COUNTRY = ["United-States", "Mexico", "Germany", "India", "Philippines"]


def _generate_class(rng: np.random.Generator, n: int, high_income: bool) -> pd.DataFrame:
    """Genera n filas sesgadas hacia una de las dos clases de salario,
    para que las variables numéricas sean mínimamente informativas
    (evita que los modelos de prueba entrenen sobre puro ruido)."""
    if high_income:
        age = rng.integers(35, 60, size=n)
        education_idx = rng.choice(
            [2, 3, 4, 5], size=n)  # sesgo a más educación
        hours = rng.integers(40, 60, size=n)
        capital_gain = rng.choice([0, 0, 0, 5000, 15000], size=n)
        salary = " >50K"
    else:
        age = rng.integers(18, 45, size=n)
        education_idx = rng.choice([0, 1, 2, 4], size=n)
        hours = rng.integers(20, 45, size=n)
        capital_gain = np.zeros(n, dtype=int)
        salary = " <=50K"

    education_pairs = [EDUCATION[i] for i in education_idx]

    return pd.DataFrame({
        "age": age,
        "workclass": rng.choice(WORKCLASS, size=n),
        "fnlwgt": rng.integers(19000, 400000, size=n),
        "education": [e[0] for e in education_pairs],
        "education-num": [e[1] for e in education_pairs],
        "marital-status": rng.choice(MARITAL_STATUS, size=n),
        "occupation": rng.choice(OCCUPATION, size=n),
        "relationship": rng.choice(RELATIONSHIP, size=n),
        "race": rng.choice(RACE, size=n),
        "sex": rng.choice(SEX, size=n),
        "capital-gain": capital_gain,
        "capital-loss": np.zeros(n, dtype=int),
        "hours-per-week": hours,
        "native-country": rng.choice(NATIVE_COUNTRY, size=n, p=[0.8, 0.05, 0.05, 0.05, 0.05]),
        "salary": salary,
    })


def generate_sample_salary(n_per_class: int = 25, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Genera un dataset balanceado (n_per_class filas por cada clase de salario)
    y lo devuelve mezclado (shuffle) con índice reiniciado."""
    rng = np.random.default_rng(random_state)

    df_high = _generate_class(rng, n_per_class, high_income=True)
    df_low = _generate_class(rng, n_per_class, high_income=False)

    df = pd.concat([df_high, df_low], ignore_index=True)

    # introduce algunos nulos controlados, útil para probar DataLoader/imputación
    n_nulls = max(1, len(df) // 15)
    null_idx = rng.choice(df.index, size=n_nulls, replace=False)
    df.loc[null_idx, "workclass"] = np.nan

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-per-class", type=int, default=25,
        help="Filas por clase de salario (total = 2 * n-per-class). Default: 25",
    )
    parser.add_argument(
        "--output", type=str, default="tests/fixtures/sample_salary.csv",
        help="Ruta de salida del CSV. Default: tests/fixtures/sample_salary.csv",
    )
    parser.add_argument(
        "--random-state", type=int, default=RANDOM_STATE,
        help="Semilla para reproducibilidad. Default: 42",
    )
    args = parser.parse_args()

    df = generate_sample_salary(
        n_per_class=args.n_per_class, random_state=args.random_state)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generado: {output_path} ({len(df)} filas)")
    print(df["salary"].value_counts())


if __name__ == "__main__":
    main()
